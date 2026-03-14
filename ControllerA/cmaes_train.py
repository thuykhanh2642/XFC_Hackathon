import sys
sys.path.append('.')

import json
import os
import time
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor

import cma
from kesslergame import KesslerGame, GraphicsType
from HybridController2 import GAFuzzyController
from scenarios import training_set, validation_set


# Config

N_GENES = 50
MAX_ITERS = 200
SIGMA0 = 0.15
VALIDATION_LOG_EVERY = 5
EARLY_STOP_PATIENCE = 40
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

W_HIT = 1.50
W_ACC = 0.35
W_DEATH = 0.75
W_SURVIVAL = 0.50
W_MINES = 0.05

game_settings = {
    'perf_tracker': False,
    'graphics_type': GraphicsType.NoGraphics,
    'realtime_multiplier': 0,
    'graphics_obj': None,
    'frequency': 30,
    'time_limit': 30,
    'competition_safe_mode': False,
}



# Helpers

def fmt_duration(seconds):
    return str(timedelta(seconds=int(seconds)))


def clamp_genome(x):
    return [float(min(max(v, 0.0), 1.0)) for v in x]


def score_team(team):
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


def _evaluate_one(genome, scenarios):
    genome = clamp_genome(genome)
    controller = GAFuzzyController(chromosome=genome)
    game = KesslerGame(settings=game_settings)

    total_score = 0.0
    details = {}
    for scenario in scenarios:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        score = score_team(result.teams[0])
        total_score += score
        details[scenario.name] = score
    return total_score, details


def train_fitness(genome):
    total, _ = _evaluate_one(genome, training_set)
    return total


def validation_score(genome):
    total, _ = _evaluate_one(genome, validation_set)
    return total


def validation_score_detailed(genome):
    return _evaluate_one(genome, validation_set)


def save_best(genome, iteration, out_dir, sigma, train_score, val_score, val_details=None):
    result = {
        "method": "CMA-ES",
        "iteration": iteration,
        "train_score": train_score,
        "validation_score": val_score,
        "validation_details": val_details,
        "sigma": sigma,
        "genome": list(clamp_genome(genome)),
        "n_genes": len(genome),
        "train_scenarios": len(training_set),
        "validation_scenarios": len(validation_set),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"iter_{iteration:04d}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with (out_dir.parent / "best_solution_cmaes.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def load_initial_genome():
    cma_path = Path(os.path.dirname(__file__), "best_solution_cmaes.json")
    if cma_path.exists():
        with cma_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        genome = data.get("genome")
        if genome is not None and len(genome) == N_GENES:
            print("Loaded starting genome from best_solution_cmaes.json")
            return [float(x) for x in genome]

    ga_path = Path(os.path.dirname(__file__), "best_solution.json")
    if ga_path.exists():
        with ga_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        genome = data.get("genome")
        if genome is not None and len(genome) == N_GENES:
            print("Loaded starting genome from best_solution.json (GA warm start)")
            return [float(x) for x in genome]

    print("Using default CMA-ES start at 0.5")
    return [0.5] * N_GENES


def clear_solution_history(output_dir: Path):
    if not output_dir.exists():
        return
    for p in output_dir.glob("iter_*.json"):
        p.unlink()



# Main

def main():
    print("=" * 72)
    print("  GAFuzzyController — CMA-ES TRAINING")
    print("=" * 72)
    print(f"  Dimensions : {N_GENES}")
    print(f"  Max iters  : {MAX_ITERS}")
    print(f"  Sigma0     : {SIGMA0}")
    print(f"  Train set  : {len(training_set)}")
    print(f"  Val set    : {len(validation_set)}")
    print(f"  Workers    : {N_WORKERS}")
    print("  Saves to   : best_solution_cmaes.json")
    print("=" * 72)
    print()

    x0 = load_initial_genome()
    out_dir = Path(os.path.dirname(__file__), "solution_history_cmaes")
    clear_solution_history(out_dir)

    es = cma.CMAEvolutionStrategy(
        x0,
        SIGMA0,
        {
            "bounds": [0.0, 1.0],
            "verb_log": 0,
            "verb_disp": 0,
        }
    )

    best_genome = clamp_genome(x0)
    best_train = train_fitness(best_genome)
    best_val, best_val_details = validation_score_detailed(best_genome)
    save_best(best_genome, 0, out_dir, SIGMA0, best_train, best_val, best_val_details)

    validation_history = [best_val]
    no_improve_iters = 0
    start_time = time.perf_counter()

    print(f"{'Iter':>5}  {'train':>8}  {'val':>8}  {'sigma':>8}  {'pop':>5}  {'iter_t':>7}  {'elapsed':>9}")
    print("-" * 64)
    print(f"{0:>5}  {best_train:>8.4f}  {best_val:>8.4f}  {SIGMA0:>8.4f}  {es.popsize:>5}  {'—':>7}  {fmt_duration(0):>9}")

    for it in range(1, MAX_ITERS + 1):
        t0 = time.perf_counter()

        solutions = [clamp_genome(x) for x in es.ask()]

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            fitnesses = list(executor.map(train_fitness, solutions))

        losses = [-f for f in fitnesses]
        es.tell(solutions, losses)

        idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
        gen_best = solutions[idx]
        gen_best_train = fitnesses[idx]

        if gen_best_train > best_train:
            best_genome = gen_best
            best_train = gen_best_train

        if it % VALIDATION_LOG_EVERY == 0:
            best_val, best_val_details = validation_score_detailed(best_genome)
            print("    Validation breakdown:")
            for name, score in best_val_details.items():
                print(f"      {name}: {score:.4f}")
            print(f"      total: {best_val:.4f}")
        else:
            best_val = validation_score(best_genome)
            best_val_details = None

        save_best(best_genome, it, out_dir, es.sigma, best_train, best_val, best_val_details)

        if best_val > max(validation_history, default=float('-inf')) + 1e-9:
            no_improve_iters = 0
        else:
            no_improve_iters += 1
        validation_history.append(best_val)

        iter_time = time.perf_counter() - t0
        elapsed = time.perf_counter() - start_time
        print(f"{it:>5}  {best_train:>8.4f}  {best_val:>8.4f}  {es.sigma:>8.4f}  {es.popsize:>5}  {iter_time:>6.1f}s  {fmt_duration(elapsed):>9}")

        if no_improve_iters >= EARLY_STOP_PATIENCE:
            print()
            print(f"Early stopping at iteration {it}: no validation improvement for {EARLY_STOP_PATIENCE} iterations.")
            break

    final_val, final_val_details = validation_score_detailed(best_genome)
    save_best(best_genome, it, out_dir, es.sigma, best_train, final_val, final_val_details)

    print()
    print("=" * 72)
    print("  CMA-ES TRAINING COMPLETE")
    print("=" * 72)
    print(f"  Best train score : {best_train:.4f}")
    print(f"  Best val score   : {final_val:.4f}")
    print("  Saved to         : best_solution_cmaes.json")
    print("=" * 72)


if __name__ == "__main__":
    main()
