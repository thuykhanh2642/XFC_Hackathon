# GA Training Script — HybridGA (parallel evaluation)

import sys
sys.path.append('.')

import json
import os
import random
import time
import multiprocessing
from datetime import timedelta
from pathlib import Path
from scenarios import training_set

from deap import base, creator, tools
from fitness_function import Fitness


def clear_solution_history(output_dir: Path):
    if not output_dir.exists():
        return
    for p in output_dir.glob("gen_*.json"):
        p.unlink()


def save_best(individual, generation, out_dir, pop_size, cxpb, mutpb):
    result = {
        "generation":      generation,
        "fitness":         list(individual.fitness.values),
        "genome":          list(individual),
        "n_genes":         len(individual),
        "population_size": pop_size,
        "cxpb":            cxpb,
        "mutpb":           mutpb,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"gen_{generation:04d}.json").open("w") as f:
        json.dump(result, f, indent=2)
    with (out_dir.parent / "best_solution.json").open("w") as f:
        json.dump(result, f, indent=2)


def fmt_duration(seconds):
    return str(timedelta(seconds=int(seconds)))


def print_header(pop_size, max_gen, n_scenarios, n_workers):
    print("=" * 70)
    print("  GA FUZZY CONTROLLER — TRAINING")
    print("=" * 70)
    print(f"  Population : {pop_size} individuals")
    print(f"  Max gens   : {max_gen}")
    print(f"  Scenarios  : {n_scenarios} per eval")
    print(f"  Workers    : {n_workers} parallel processes")
    print(f"  Speedup    : ~{n_workers}x vs sequential")
    print("=" * 70)
    print()


N_GENES  = 50
POP_SIZE = 20
MAX_GEN  = 1000
CXPB     = 0.5
MUTPB    = 0.2
SIGMA    = 0.15
INDPB    = 0.05

# Leave 1 CPU free for the OS — change this if you want more/fewer workers
N_WORKERS = max(1, multiprocessing.cpu_count() - 1)


def main():
    n_scenarios = len(training_set)

    #DEAP setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_flt",   random.uniform, 0.0, 1.0)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_flt, N_GENES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate",   Fitness)
    toolbox.register("mate",       tools.cxTwoPoint)
    toolbox.register("mutate",     tools.mutGaussian,
                     mu=0.0, sigma=SIGMA, indpb=INDPB)
    toolbox.register("select",     tools.selTournament, tournsize=3)

    out_dir = Path(os.path.dirname(__file__), "solution_history")
    clear_solution_history(out_dir)

    print_header(POP_SIZE, MAX_GEN, n_scenarios, N_WORKERS)

    # Parallel worker pool
    pool = multiprocessing.Pool(processes=N_WORKERS)
    toolbox.register("map", pool.map)

    # ── Initial population
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
    save_best(hof[0], 0, out_dir, POP_SIZE, CXPB, MUTPB)

    fits = [ind.fitness.values[0] for ind in pop]
    print(f"Done in {gen0_time:.1f}s")
    print()
    print(f"{'Gen':>5}  {'best':>8}  {'max':>8}  {'mean':>8}  {'std':>7}  "
          f"{'evals':>6}  {'gen_t':>7}  {'elapsed':>9}  {'ETA':>9}")
    print("-" * 80)
    print(f"{'0':>5}  {hof[0].fitness.values[0]:>8.4f}  "
          f"{max(fits):>8.4f}  {sum(fits)/len(fits):>8.4f}  "
          f"{'—':>7}  {'—':>6}  {gen0_time:>6.1f}s  "
          f"{fmt_duration(0):>9}  {'—':>9}")

    # Evolution loop
    gen_times = []

    for g in range(1, MAX_GEN + 1):
        t_gen = time.perf_counter()

        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                for i in range(len(mutant)):
                    mutant[i] = float(min(max(mutant[i], 0.0), 1.0))
                del mutant.fitness.values

        # Evaluate stale individuals — in parallel
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)
        save_best(hof[0], g, out_dir, POP_SIZE, CXPB, MUTPB)

        gen_time = time.perf_counter() - t_gen
        gen_times.append(gen_time)
        avg_gen_time = sum(gen_times[-10:]) / len(gen_times[-10:])
        elapsed = time.perf_counter() - training_start
        eta = fmt_duration(avg_gen_time * (MAX_GEN - g))

        fits = [ind.fitness.values[0] for ind in pop]
        mean = sum(fits) / len(fits)
        std  = (sum(x**2 for x in fits) / len(fits) - mean**2) ** 0.5

        print(f"{g:>5}  {hof[0].fitness.values[0]:>8.4f}  "
              f"{max(fits):>8.4f}  {mean:>8.4f}  {std:>7.4f}  "
              f"{len(invalid):>6}  {gen_time:>6.1f}s  "
              f"{fmt_duration(elapsed):>9}  {eta:>9}")

    pool.close()
    pool.join()

    total = time.perf_counter() - training_start
    print()
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time   : {fmt_duration(total)}")
    print(f"  Best fitness : {hof[0].fitness.values[0]:.4f}")
    print(f"  Saved to     : {(out_dir.parent / 'best_solution.json').resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()