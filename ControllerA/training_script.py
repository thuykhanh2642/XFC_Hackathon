# training_script_patched.py
# Trains the GAFuzzyController (hybrid architecture) using GA
#type: ignore
import sys
sys.path.append('.')

import json
import os
import random
import time
import multiprocessing
from datetime import timedelta
from pathlib import Path
import copy
from deap import base, creator, tools
from kesslergame import KesslerGame, GraphicsType
from HybridController2 import GAFuzzyController
from scenarios import training_set, validation_set


# Fitness config ───────────────────────────────────────────────────────────
# Rebalanced to reduce all-or-nothing failure on hard scenarios while still
# valuing survival and clean play.
W_HIT       = 1.20
W_ACC       = 0.40
W_DEATH     = 0.45
W_SURVIVAL  = 0.75
W_MINES     = 0.05
W_TIME      = 0.35   # partial credit for surviving longer, when available

SCENARIO_WEIGHTS = {
    # Easier maps should not dominate the search.
    "One Asteroid Still": 0.60,
    "One Asteroid Slow Horizontal": 0.75,
    "Two Asteroids Still": 0.80,
    "Three Asteroids Row": 0.90,
    "Sniper Practice (Large Arena)": 0.90,

    # Medium / general maps.
    "Stock Scenario": 1.10,
    "Donut Ring": 1.15,
    "Donut Ring (Closing In, Large Asteroids)": 1.15,
    "Vertical Wall Left (Big Moving Right)": 1.20,
    "Asteroid Rain": 1.10,
    "Crossing Lanes": 1.20,
    "Giants with Kamikaze": 1.25,
    "Spiral Swarm": 1.15,
    "Four Corner Assault": 1.25,

    # Hard holdouts should matter a lot.
    "Cross (Rotating Look, CW)": 2.00,
    "Moving Maze (Rightward Tunnel)": 2.50,
}


game_settings = {
    'perf_tracker':          False,
    'graphics_type':         GraphicsType.NoGraphics,
    'realtime_multiplier':   0,
    'graphics_obj':          None,
    'frequency':             30,
    'time_limit':            30,
    'competition_safe_mode': False,
}

VALIDATION_LOG_EVERY = 5

# Shared curriculum state for multiprocessing workers
_WORKER_SHARED_STATE = None


# Fitness helpers ─────────────────────────────────────────────────────────

def scenario_weight(name):
    return float(SCENARIO_WEIGHTS.get(name, 1.0))


def survival_fraction(team):
    """Best-effort extraction of partial survival signal from result fields."""
    candidate_attrs = (
        "time_alive",
        "survival_time",
        "time_survived",
        "sim_time_alive",
        "seconds_alive",
    )
    for attr in candidate_attrs:
        value = getattr(team, attr, None)
        if value is not None:
            try:
                return float(max(0.0, min(float(value) / float(game_settings['time_limit']), 1.0)))
            except Exception:
                pass
    return 0.0


def raw_score_team(team):
    """Single-scenario scalar objective before scenario weighting."""
    mines_used = float(getattr(team, "mines_used", 0.0) or 0.0)
    score = (
        W_HIT * float(team.fraction_total_asteroids_hit)
        + W_ACC * float(team.accuracy)
        - W_DEATH * float(team.deaths)
        - W_MINES * mines_used
        + W_TIME * survival_fraction(team)
    )
    if int(team.deaths) == 0:
        score += W_SURVIVAL
    return score


def weighted_score_team(team, scenario_name):
    return scenario_weight(scenario_name) * raw_score_team(team)


def init_worker(shared_state):
    global _WORKER_SHARED_STATE
    _WORKER_SHARED_STATE = shared_state


def _scenario_map():
    return {s.name: s for s in training_set}


def _tiered_training_sets():
    by_name = _scenario_map()
    # Note: Cross and Moving Maze are validation-only, not in training_set.
    hard_holdout = []

    tier1_names = [
        "One Asteroid Still",
        "One Asteroid Slow Horizontal",
        "Two Asteroids Still",
        "Three Asteroids Row",
    ]

    tier2_names = [
        "Stock Scenario",
        "Donut Ring",
        "Donut Ring (Closing In, Large Asteroids)",
        "Vertical Wall Left (Big Moving Right)",
    ]

    tier3_names = [
        "Asteroid Rain",
        "Crossing Lanes",
        "Giants with Kamikaze",
        "Spiral Swarm",
        "Four Corner Assault",
    ]

    def present(names):
        return [by_name[n] for n in names if n in by_name]

    tier1 = present(tier1_names)
    tier2 = present(tier2_names)
    tier3 = present(tier3_names)
    hard = present(hard_holdout)

    already = {s.name for s in (tier1 + tier2 + tier3 + hard)}
    leftovers = [s for s in training_set if s.name not in already]

    # Unlisted scenarios are treated as medium complexity rather than hard holdouts.
    tier3 = tier3 + leftovers

    return tier1, tier2, tier3, hard


def get_training_set_for_generation(generation):
    tier1, tier2, tier3, hard = _tiered_training_sets()

    # Accumulating curriculum: do not drop earlier skills, just add complexity.
    if generation < 5:
        active = tier1                          # 4 easy (still/slow targets)
    elif generation < 12:
        active = tier1 + tier2                  # +4 medium (stock, donut, wall, closing)
    else:
        active = tier1 + tier2 + tier3          # +5 hard (rain, lanes, giants, spiral, corners) = all 12+

    # Robust fallback in case names differ from expectations.
    return active if active else list(training_set)


def get_active_training_set():
    generation = int(getattr(_WORKER_SHARED_STATE, 'generation', 0) or 0)
    return get_training_set_for_generation(generation)


def evaluate_on_scenarios(individual, scenarios=None):
    controller = GAFuzzyController(chromosome=individual)
    game = KesslerGame(settings=game_settings)
    total_score = 0.0
    total_weight = 0.0

    active_scenarios = get_active_training_set() if scenarios is None else scenarios
    for scenario in active_scenarios:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        team = result.teams[0]
        total_score += weighted_score_team(team, scenario.name)
        total_weight += scenario_weight(scenario.name)

    return total_score / max(total_weight, 1e-9)


def evaluate_scenarios_detailed(individual, scenarios):
    controller = GAFuzzyController(chromosome=individual)
    game = KesslerGame(settings=game_settings)
    details = {}

    for scenario in scenarios:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        team = result.teams[0]
        raw = raw_score_team(team)
        weight = scenario_weight(scenario.name)
        details[scenario.name] = {
            "raw": raw,
            "weight": weight,
            "weighted": raw * weight,
        }

    return details


def Fitness(individual):
    return evaluate_on_scenarios(individual, None),


def validate(individual, detailed=False, quiet=True):
    if not detailed:
        return evaluate_on_scenarios(individual, validation_set)

    details = evaluate_scenarios_detailed(individual, validation_set)
    total_weighted = sum(v["weighted"] for v in details.values())
    total_weight = sum(v["weight"] for v in details.values())
    total = total_weighted / max(total_weight, 1e-9)
    if not quiet:
        print("    Validation breakdown:")
        for name, info in details.items():
            print(
                f"      {name}: raw={info['raw']:.4f}  "
                f"w={info['weight']:.2f}  weighted={info['weighted']:.4f}"
            )
        print(f"      weighted total: {total:.4f}")
    return total, details


# Helpers ─────────────────────────────────────────────────────────────────

def clear_solution_history(output_dir: Path):
    if not output_dir.exists():
        return
    for p in output_dir.glob("gen_*.json"):
        p.unlink()


def save_best(individual, generation, out_dir, pop_size, cxpb, mutpb, sigma,
              validation_score=None, validation_details=None):
    result = {
        "generation":       generation,
        "fitness":          list(individual.fitness.values),
        "validation_score": validation_score,
        "validation_details": validation_details,
        "genome":           list(individual),
        "n_genes":          len(individual),
        "population_size":  pop_size,
        "cxpb":             cxpb,
        "mutpb":            mutpb,
        "sigma":            sigma,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"gen_{generation:04d}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with (out_dir.parent / "best_solution.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def fmt_duration(seconds):
    return str(timedelta(seconds=int(seconds)))


def clamp_individual(individual):
    for i in range(len(individual)):
        individual[i] = float(min(max(individual[i], 0.0), 1.0))


def mean_gene_std(population):
    if not population:
        return 0.0
    n = len(population[0])
    per_gene = []
    for i in range(n):
        vals = [ind[i] for ind in population]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        per_gene.append(var ** 0.5)
    return sum(per_gene) / len(per_gene)


def compute_mutation_params(generation, best_history, validation_history):
    """Increase mutation when training OR validation plateaus."""
    cur_mutpb = MUTPB
    cur_sigma = SIGMA

    if generation >= 10 and len(best_history) >= 10:
        recent_best = max(best_history[-10:])
        older_best = max(best_history[:-10], default=recent_best)
        train_plateau = recent_best <= older_best * 1.001
        if train_plateau:
            cur_mutpb = max(cur_mutpb, min(0.45, MUTPB * 1.8))
            cur_sigma = max(cur_sigma, min(0.35, SIGMA * 1.5))

    if generation >= 8 and len(validation_history) >= 8:
        recent_val = max(validation_history[-8:])
        older_val = max(validation_history[:-8], default=recent_val)
        val_plateau = recent_val <= older_val + 1e-9
        if val_plateau:
            cur_mutpb = max(cur_mutpb, 0.40)
            cur_sigma = max(cur_sigma, 0.30)

    return cur_mutpb, cur_sigma


# GA config ───────────────────────────────────────────────────────────────

N_GENES             = 50
POP_SIZE            = 60
MAX_GEN             = 600
CXPB                = 0.5
MUTPB               = 0.25
SIGMA               = 0.20
INDPB               = 0.20
ELITE_FRAC          = 0.08
EARLY_STOP_PATIENCE = 80
DIVERSITY_FLOOR     = 0.04
RANDOM_INJECT_FRAC  = 0.25
N_WORKERS           = max(1, multiprocessing.cpu_count() - 1)


# Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  GAFuzzyController (HYBRID) — TRAINING")
    print("=" * 72)
    print(f"  Population : {POP_SIZE}  |  Max gens: {MAX_GEN}")
    print(f"  Train set  : {len(training_set)} total (smooth curriculum)  |  Validation: {len(validation_set)}")
    print(f"  Workers    : {N_WORKERS}")
    print("  Fitness    : weighted(mean) of [1.2*hit + 0.4*acc - 0.45*deaths - 0.05*mines + 0.75 clean + time]")
    print("  Extras     : elitism + adaptive mutation + diversity injection + smoother curriculum")
    print("  Saves to   : best_solution.json")
    print("=" * 72)
    print()

    for t in ["FitnessMax", "Individual"]:
        if hasattr(creator, t):
            delattr(creator, t)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_flt", random.uniform, 0.0, 1.0)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_flt, N_GENES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", Fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("clone", copy.deepcopy)

    manager = multiprocessing.Manager()
    shared_state = manager.Namespace()
    shared_state.generation = 0
    pool = multiprocessing.Pool(processes=N_WORKERS, initializer=init_worker, initargs=(shared_state,))
    toolbox.register("map", pool.map)

    out_dir = Path(os.path.dirname(__file__), "solution_history")
    clear_solution_history(out_dir)

    pop = toolbox.population(n=POP_SIZE)

    # Warm-start from previous best if available, but keep diversity higher.
    best_path = Path(os.path.dirname(__file__), 'best_solution.json')
    if best_path.exists():
        try:
            with best_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            best_genome = [float(x) for x in data.get('genome', [])]
            if len(best_genome) == N_GENES:
                print('Seeding population with previous best genome (reduced bias)')
                seed_count = min(3, POP_SIZE)
                for i in range(seed_count):
                    ind = creator.Individual(best_genome[:])
                    if i > 0:
                        tools.mutGaussian(ind, mu=0.0, sigma=0.10, indpb=0.20)
                        clamp_individual(ind)
                    pop[i] = ind
        except Exception as e:
            print(f'Warm start skipped: {e}')

    initial_active = get_training_set_for_generation(0)
    print(f"Evaluating initial population ({POP_SIZE} individuals in parallel)...")
    print(f"Curriculum phase 0 uses {len(initial_active)} training scenarios")
    t0 = time.perf_counter()
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    gen0_time = time.perf_counter() - t0
    training_start = time.perf_counter()

    hof = tools.HallOfFame(1)
    hof.update(pop)
    best_validation, best_val_details = validate(hof[0], detailed=True, quiet=False)
    save_best(hof[0], 0, out_dir, POP_SIZE, CXPB, MUTPB, SIGMA, best_validation, best_val_details)

    fits = [ind.fitness.values[0] for ind in pop]
    best_history = [hof[0].fitness.values[0]]
    validation_history = [best_validation]
    no_improve_gens = 0

    print(f"Done in {gen0_time:.1f}s")
    print()
    print(f"{'Gen':>5}  {'best':>8}  {'val':>8}  {'max':>8}  {'mean':>8}  {'gstd':>7}  "
          f"{'mutpb':>6}  {'sigma':>6}  {'evals':>6}  {'gen_t':>7}  {'elapsed':>9}  {'ETA':>9}")
    print("-" * 112)
    print(f"{'0':>5}  {hof[0].fitness.values[0]:>8.4f}  {best_validation:>8.4f}  "
          f"{max(fits):>8.4f}  {sum(fits)/len(fits):>8.4f}  {mean_gene_std(pop):>7.4f}  "
          f"{MUTPB:>6.2f}  {SIGMA:>6.2f}  {'—':>6}  {gen0_time:>6.1f}s  "
          f"{fmt_duration(0):>9}  {'—':>9}")

    gen_times = []
    last_active_count = len(initial_active)

    try:
        for g in range(1, MAX_GEN + 1):
            shared_state.generation = g
            t_gen = time.perf_counter()
            cur_mutpb, cur_sigma = compute_mutation_params(g, best_history, validation_history)
            elite_size = max(1, int(round(ELITE_FRAC * POP_SIZE)))

            elites = [toolbox.clone(ind) for ind in tools.selBest(pop, elite_size)]
            offspring = [toolbox.clone(ind) for ind in toolbox.select(pop, POP_SIZE - elite_size)]
            for ind in offspring:
                if hasattr(ind.fitness, "values"):
                    del ind.fitness.values

            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(c1, c2)
                    if hasattr(c1.fitness, "values"):
                        del c1.fitness.values
                    if hasattr(c2.fitness, "values"):
                        del c2.fitness.values

            for mutant in offspring:
                if random.random() < cur_mutpb:
                    tools.mutGaussian(mutant, mu=0.0, sigma=cur_sigma, indpb=INDPB)
                    clamp_individual(mutant)
                    if hasattr(mutant.fitness, "values"):
                        del mutant.fitness.values

            # Inject randomness if the population has collapsed too much.
            gene_std = mean_gene_std(pop)
            if g >= 6 and gene_std < DIVERSITY_FLOOR:
                inject_n = max(1, int(round(RANDOM_INJECT_FRAC * POP_SIZE)))
                for i in range(inject_n):
                    new_ind = toolbox.individual()
                    if i < len(offspring):
                        offspring[-(i + 1)] = new_ind
                print(f"--> Diversity injection at generation {g}: gene-std={gene_std:.4f}, injected {inject_n} random individuals")

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            if invalid:
                fitnesses = toolbox.map(toolbox.evaluate, invalid)
                for ind, fit in zip(invalid, fitnesses):
                    ind.fitness.values = fit

            pop[:] = elites + offspring
            hof.update(pop)
            best_ind = hof[0]

            should_log_validation = (g % VALIDATION_LOG_EVERY == 0)
            if should_log_validation:
                best_validation, best_val_details = validate(best_ind, detailed=True, quiet=False)
            else:
                best_validation = validate(best_ind)
                best_val_details = None

            save_best(best_ind, g, out_dir, POP_SIZE, CXPB, cur_mutpb, cur_sigma,
                      best_validation, best_val_details)

            gen_time = time.perf_counter() - t_gen
            gen_times.append(gen_time)
            avg_gen_time = sum(gen_times[-10:]) / len(gen_times[-10:])
            elapsed = time.perf_counter() - training_start
            eta = fmt_duration(avg_gen_time * (MAX_GEN - g))

            fits = [ind.fitness.values[0] for ind in pop]
            gen_max = max(fits)
            mean = sum(fits) / len(fits)
            best_now = best_ind.fitness.values[0]
            best_history.append(best_now)
            validation_history.append(best_validation)

            if best_validation > max(validation_history[:-1], default=float('-inf')) + 1e-9:
                no_improve_gens = 0
            else:
                no_improve_gens += 1

            active_len = len(get_training_set_for_generation(g))
            if active_len != last_active_count:
                print(f"--> Curriculum phase change at generation {g}: now using {active_len} training scenarios")
                last_active_count = active_len

            gene_std_after = mean_gene_std(pop)
            print(f"{g:>5}  {best_now:>8.4f}  {best_validation:>8.4f}  "
                  f"{gen_max:>8.4f}  {mean:>8.4f}  {gene_std_after:>7.4f}  "
                  f"{cur_mutpb:>6.2f}  {cur_sigma:>6.2f}  {len(invalid):>6}  {gen_time:>6.1f}s  "
                  f"{fmt_duration(elapsed):>9}  {eta:>9}")

            if no_improve_gens >= EARLY_STOP_PATIENCE:
                print()
                print(f"Early stopping at generation {g}: no validation improvement for "
                      f"{EARLY_STOP_PATIENCE} generations.")
                break
    finally:
        pool.close()
        pool.join()
        manager.shutdown()

    total = time.perf_counter() - training_start
    final_val, final_val_details = validate(hof[0], detailed=True, quiet=False)
    print()
    print("=" * 72)
    print("  TRAINING COMPLETE")
    print("=" * 72)
    print(f"  Total time       : {fmt_duration(total)}")
    print(f"  Best train score : {hof[0].fitness.values[0]:.4f}")
    print(f"  Best val score   : {final_val:.4f}")
    print("  Saved to         : best_solution.json")
    print("=" * 72)


if __name__ == "__main__":
    main()