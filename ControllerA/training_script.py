# training_script_patched.py
# Trains the GAFuzzyController (hybrid architecture) using GA
# Run:  python training_script_patched.py

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

    # Hard holdouts — architecturally unwinnable, keep low weight
    # so they don't mask progress on achievable scenarios.
    "Cross (Rotating Look, CW)": 0.30,
    "Moving Maze (Rightward Tunnel)": 0.30,
}


game_settings = {
    'perf_tracker':          False,
    'graphics_type':         GraphicsType.NoGraphics,
    'realtime_multiplier':   0,
    'graphics_obj':          None,
    'frequency':             30,
    'time_limit':            20,
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
    # No curriculum — train on all scenarios from the start.
    # Scenario weights already control relative importance.
    return list(training_set)


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


_best_saved_validation = float('-inf')

def save_best(individual, generation, out_dir, pop_size, cxpb, mutpb, sigma,
              validation_score=None, validation_details=None):
    global _best_saved_validation
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
    # Only overwrite best_solution.json when validation improves
    val = validation_score if validation_score is not None else float('-inf')
    if val >= _best_saved_validation:
        _best_saved_validation = val
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
POP_SIZE            = 40
MAX_GEN             = 400
CXPB                = 0.5
MUTPB               = 0.25
SIGMA               = 0.20
INDPB               = 0.20
ELITE_FRAC          = 0.08
EARLY_STOP_PATIENCE = 30
DIVERSITY_FLOOR     = 0.04
RANDOM_INJECT_FRAC  = 0.25
N_WORKERS           = max(1, multiprocessing.cpu_count() - 1)


# Multi-restart config
N_RESTARTS          = 10
GENS_PER_RESTART    = 25


# Main ────────────────────────────────────────────────────────────────────

def run_one_restart(restart_id, toolbox, shared_state, out_dir):
    """Run a short GA from random init, return (best_genome, best_val, best_train)."""
    pop = toolbox.population(n=POP_SIZE)

    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(1)
    hof.update(pop)

    best_val = validate(hof[0])
    best_val_genome = list(hof[0])
    best_train = hof[0].fitness.values[0]

    for g in range(1, GENS_PER_RESTART + 1):
        shared_state.generation = g
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
            if random.random() < MUTPB:
                tools.mutGaussian(mutant, mu=0.0, sigma=SIGMA, indpb=INDPB)
                clamp_individual(mutant)
                if hasattr(mutant.fitness, "values"):
                    del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        if invalid:
            fitnesses = toolbox.map(toolbox.evaluate, invalid)
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

        pop[:] = elites + offspring
        hof.update(pop)

        cur_val = validate(hof[0])
        cur_train = hof[0].fitness.values[0]

        if cur_val > best_val + 1e-9:
            best_val = cur_val
            best_val_genome = list(hof[0])
            best_train = cur_train

        fits = [ind.fitness.values[0] for ind in pop]
        print(f"  R{restart_id:>2} G{g:>3}  train={cur_train:>8.4f}  val={cur_val:>8.4f}  "
              f"best_val={best_val:>8.4f}  mean={sum(fits)/len(fits):>8.4f}  gstd={mean_gene_std(pop):>7.4f}")

    return best_val_genome, best_val, best_train


def main():
    print("=" * 72)
    print("  GAFuzzyController — MULTI-RESTART GA")
    print("=" * 72)
    print(f"  Restarts   : {N_RESTARTS}  |  Gens per restart: {GENS_PER_RESTART}")
    print(f"  Population : {POP_SIZE}  |  Train set: {len(training_set)}  |  Val set: {len(validation_set)}")
    print(f"  Workers    : {N_WORKERS}")
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

    global_best_val = float('-inf')
    global_best_genome = None
    global_best_train = None
    all_results = []

    training_start = time.perf_counter()

    try:
        for r in range(N_RESTARTS):
            print(f"\n{'='*60}")
            print(f"  RESTART {r+1}/{N_RESTARTS}")
            print(f"{'='*60}")
            t0 = time.perf_counter()

            genome, val, train = run_one_restart(r+1, toolbox, shared_state, out_dir)
            rt = time.perf_counter() - t0
            all_results.append({'restart': r+1, 'val': val, 'train': train, 'time': rt})

            if val > global_best_val + 1e-9:
                global_best_val = val
                global_best_genome = genome
                global_best_train = train

                # Save new global best
                ind = creator.Individual(genome)
                ind.fitness.values = (train,)
                val_detail_score, val_details = validate(ind, detailed=True, quiet=False)
                save_best(ind, r+1, out_dir, POP_SIZE, CXPB, MUTPB, SIGMA,
                          val_detail_score, val_details)
                print(f"  *** New global best: val={val:.4f}  train={train:.4f} ***")
            else:
                print(f"  Restart {r+1} best val={val:.4f} (global best still {global_best_val:.4f})")

            elapsed = time.perf_counter() - training_start
            print(f"  Time: {rt:.0f}s this restart | {fmt_duration(elapsed)} total")

    finally:
        pool.close()
        pool.join()
        manager.shutdown()

    total = time.perf_counter() - training_start
    print()
    print("=" * 72)
    print("  MULTI-RESTART TRAINING COMPLETE")
    print("=" * 72)
    print(f"  Total time         : {fmt_duration(total)}")
    print(f"  Restarts completed : {len(all_results)}")
    print(f"  Global best val    : {global_best_val:.4f}")
    print(f"  Global best train  : {global_best_train:.4f}")
    print("  Saved to           : best_solution.json")
    print()
    print("  Per-restart results:")
    for res in all_results:
        marker = " <-- BEST" if abs(res['val'] - global_best_val) < 1e-9 else ""
        print(f"    R{res['restart']:>2}: val={res['val']:>8.4f}  train={res['train']:>8.4f}  ({res['time']:.0f}s){marker}")
    print("=" * 72)


if __name__ == "__main__":
    main()