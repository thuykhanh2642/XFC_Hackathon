# GA Training Script — GAFuzzyController
# Drop-in replacement for example_fuzzy_training_script.py.
# Run this from the project root:   python training_script.py
#
# Chromosome: 50 floats in [0.0, 1.0]
# Encoding breakdown -> see GAcontroller.py header comment.

import sys
sys.path.append('.')

import json
import os
import random
from pathlib import Path

from deap import base, creator, tools

from fitness_function import Fitness


#Helpers

def clear_solution_history(output_dir: Path):
    if not output_dir.exists():
        return
    for p in output_dir.glob("gen_*.json"):
        p.unlink()


def save_best(individual, generation, out_dir, pop_size, cxpb, mutpb):
    result = {
        "generation":     generation,
        "fitness":        list(individual.fitness.values),
        "genome":         list(individual),
        "n_genes":        len(individual),
        "population_size": pop_size,
        "cxpb":           cxpb,
        "mutpb":          mutpb,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gen_{generation:04d}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    # Always overwrite the "best so far" convenience file
    best_path = out_dir.parent / "best_solution.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


#GA setup

N_GENES      = 50
POP_SIZE     = 20
MAX_GEN      = 1000
CXPB         = 0.5
MUTPB        = 0.2
SIGMA        = 0.15   # slightly tighter Gaussian mutation than baseline
INDPB        = 0.05   # per-gene mutation probability


def main():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_flt",  random.uniform, 0.0, 1.0)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_flt, N_GENES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", Fitness)
    toolbox.register("mate",     tools.cxTwoPoint)
    toolbox.register("mutate",   tools.mutGaussian,
                     mu=0.0, sigma=SIGMA, indpb=INDPB)
    toolbox.register("select",   tools.selTournament, tournsize=3)

    #initialise population
    pop = toolbox.population(n=POP_SIZE)

    out_dir = Path(os.path.dirname(__file__), "solution_history")
    clear_solution_history(out_dir)

    print(f"Evaluating initial population ({POP_SIZE} individuals) …")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(1)
    hof.update(pop)
    save_best(hof[0], 0, out_dir, POP_SIZE, CXPB, MUTPB)

    fits = [ind.fitness.values[0] for ind in pop]
    print(f"Gen 0 — max={max(fits):.4f}  mean={sum(fits)/len(fits):.4f}")

    #evolution loop
    for g in range(1, MAX_GEN + 1):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        # crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # mutation + clamp to [0, 1]
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                for i in range(len(mutant)):
                    mutant[i] = float(min(max(mutant[i], 0.0), 1.0))
                del mutant.fitness.values

        # evaluate stale individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)
        save_best(hof[0], g, out_dir, POP_SIZE, CXPB, MUTPB)

        fits  = [ind.fitness.values[0] for ind in pop]
        mean  = sum(fits) / len(fits)
        std   = (sum(x**2 for x in fits) / len(fits) - mean**2) ** 0.5

        print(
            f"Gen {g:4d} — "
            f"best={hof[0].fitness.values[0]:.4f}  "
            f"max={max(fits):.4f}  mean={mean:.4f}  std={std:.4f}"
        )

    print("\n── Training complete ──")
    print(f"Best fitness : {hof[0].fitness.values[0]:.4f}")
    print(f"Best genome  : {list(hof[0])}")
    print(f"Checkpoints  : {out_dir.resolve()}")
    print(f"Best solution: {(out_dir.parent / 'best_solution.json').resolve()}")


if __name__ == "__main__":
    main()