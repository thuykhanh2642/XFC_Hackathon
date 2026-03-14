# training_script.py
# Trains the GAFuzzyController (hybrid architecture) using GA
# Run:  python training_script.py

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


#Fitness config ────────────────────────────────────────────────────────────

W_HIT       = 1.50
W_ACC       = 0.35
W_DEATH     = 0.75
W_SURVIVAL  = 0.50
W_MINES     = 0.05

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


#Fitness helpers ───────────────────────────────────────────────────────────

def score_team(team):
    """Single scalar objective: clear asteroids, stay alive, avoid waste."""
    mines_used = float(getattr(team, "mines_used", 0.0) or 0.0)
    score = (
        W_HIT * float(team.fraction_total_asteroids_hit)
        + W_ACC * float(team.accuracy)
        - W_DEATH * float(team.deaths)
        - W_MINES * mines_used
    )
    if int(team.deaths) == 0:
        score += W_SURVIVAL
    return score


def evaluate_on_scenarios(individual, scenarios):
    controller = GAFuzzyController(chromosome=individual)
    game = KesslerGame(settings=game_settings)
    total_score = 0.0

    for scenario in scenarios:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        total_score += score_team(result.teams[0])

    return total_score


def evaluate_scenarios_detailed(individual, scenarios):
    controller = GAFuzzyController(chromosome=individual)
    game = KesslerGame(settings=game_settings)
    details = {}

    for scenario in scenarios:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        details[scenario.name] = score_team(result.teams[0])

    return details


def Fitness(individual):
    return evaluate_on_scenarios(individual, training_set),


def validate(individual, detailed=False, quiet=True):
    if not detailed:
        return evaluate_on_scenarios(individual, validation_set)

    details = evaluate_scenarios_detailed(individual, validation_set)
    total = sum(details.values())
    if not quiet:
        print("    Validation breakdown:")
        for name, score in details.items():
            print(f"      {name}: {score:.4f}")
        print(f"      total: {total:.4f}")
    return total, details


#Helpers ──

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


def compute_mutation_params(generation, best_history):
    """Increase mutation slightly if training plateaus."""
    if generation < 20 or len(best_history) < 20:
        return MUTPB, SIGMA

    recent_best = max(best_history[-20:])
    older_best = max(best_history[:-20], default=recent_best)
    plateau = recent_best <= older_best * 1.001
    if plateau:
        return min(0.35, MUTPB * 1.5), min(0.35, SIGMA * 1.25)
    return MUTPB, SIGMA


#GA config 

N_GENES             = 50
POP_SIZE            = 30
MAX_GEN             = 1000
CXPB                = 0.5
MUTPB               = 0.2
SIGMA               = 0.2
INDPB               = 0.05
ELITE_FRAC          = 0.10
EARLY_STOP_PATIENCE = 60
N_WORKERS           = max(1, multiprocessing.cpu_count() - 1)


#Main

def main():
    print("=" * 72)
    print("  GAFuzzyController (HYBRID) — TRAINING")
    print("=" * 72)
    print(f"  Population : {POP_SIZE}  |  Max gens: {MAX_GEN}")
    print(f"  Train set  : {len(training_set)}  |  Validation: {len(validation_set)}")
    print(f"  Workers    : {N_WORKERS}")
    print("  Fitness    : 1.5*hit + 0.35*acc - 0.75*deaths - 0.05*mines + clean-run bonus")
    print("  Extras     : elitism + adaptive mutation + validation early stopping")
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
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("clone", copy.deepcopy)
    pool = multiprocessing.Pool(processes=N_WORKERS)
    toolbox.register("map", pool.map)

    out_dir = Path(os.path.dirname(__file__), "solution_history")
    clear_solution_history(out_dir)

    pop = toolbox.population(n=POP_SIZE)

    print(f"Evaluating initial population ({POP_SIZE} individuals in parallel)...")
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
    print(f"{'Gen':>5}  {'best':>8}  {'val':>8}  {'max':>8}  {'mean':>8}  {'std':>7}  "
          f"{'mutpb':>6}  {'sigma':>6}  {'evals':>6}  {'gen_t':>7}  {'elapsed':>9}  {'ETA':>9}")
    print("-" * 112)
    print(f"{'0':>5}  {hof[0].fitness.values[0]:>8.4f}  {best_validation:>8.4f}  "
          f"{max(fits):>8.4f}  {sum(fits)/len(fits):>8.4f}  {'—':>7}  "
          f"{MUTPB:>6.2f}  {SIGMA:>6.2f}  {'—':>6}  {gen0_time:>6.1f}s  "
          f"{fmt_duration(0):>9}  {'—':>9}")

    gen_times = []

    try:
        for g in range(1, MAX_GEN + 1):
            t_gen = time.perf_counter()
            cur_mutpb, cur_sigma = compute_mutation_params(g, best_history)
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
            std = (sum(x**2 for x in fits) / len(fits) - mean**2) ** 0.5
            best_now = best_ind.fitness.values[0]
            best_history.append(best_now)
            validation_history.append(best_validation)

            if best_validation > max(validation_history[:-1], default=float('-inf')) + 1e-9:
                no_improve_gens = 0
            else:
                no_improve_gens += 1

            print(f"{g:>5}  {best_now:>8.4f}  {best_validation:>8.4f}  "
                  f"{gen_max:>8.4f}  {mean:>8.4f}  {std:>7.4f}  "
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
