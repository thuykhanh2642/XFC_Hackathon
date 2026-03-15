import sys
sys.path.append('.')

import json
import os
import time
import random
from pathlib import Path
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor

import cma
from kesslergame import KesslerGame, GraphicsType
from HybridController2 import GAFuzzyController
from scenarios import training_set, validation_set

# CMA-ES fine-tuning trainer aligned with the patched GA objective.
# This version fixes two important issues:
#  1) champion tracking is based on VALIDATION, not only training
#  2) restarts can be cold/random rather than always recentering on the same basin

N_GENES = 50
MAX_ITERS = 120
SIGMA0 = 0.18
VALIDATION_LOG_EVERY = 5
EARLY_STOP_PATIENCE = 30
RESTART_PATIENCE = 10
MAX_RESTARTS = 4
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

W_HIT = 1.20
W_ACC = 0.40
W_DEATH = 0.45
W_SURVIVAL = 0.75
W_MINES = 0.05
W_TIME = 0.35

SCENARIO_WEIGHTS = {
    "One Asteroid Still": 0.60,
    "One Asteroid Slow Horizontal": 0.75,
    "Two Asteroids Still": 0.80,
    "Three Asteroids Row": 0.90,
    "Stock Scenario": 1.10,
    "Donut Ring": 1.15,
    "Donut Ring (Closing In, Large Asteroids)": 1.15,
    "Vertical Wall Left (Big Moving Right)": 1.20,
    "Asteroid Rain": 1.10,
    "Crossing Lanes": 1.20,
    "Giants with Kamikaze": 1.25,
    "Spiral Swarm": 1.15,
    "Four Corner Assault": 1.25,
    "Cross (Rotating Look, CW)": 0.30,
    "Moving Maze (Rightward Tunnel)": 0.30,
}

game_settings = {
    'perf_tracker': False,
    'graphics_type': GraphicsType.NoGraphics,
    'realtime_multiplier': 0,
    'graphics_obj': None,
    'frequency': 30,
    'time_limit': 20,
    'competition_safe_mode': False,
}


def fmt_duration(seconds):
    return str(timedelta(seconds=int(seconds)))


def clamp_genome(x):
    return [float(min(max(v, 0.0), 1.0)) for v in x]


def scenario_weight(name: str) -> float:
    return float(SCENARIO_WEIGHTS.get(name, 1.0))


def survival_fraction(team):
    candidate_attrs = (
        'time_alive', 'survival_time', 'time_survived', 'sim_time_alive', 'seconds_alive',
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
    mines_used = float(getattr(team, 'mines_used', 0.0) or 0.0)
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


def _scenario_map():
    return {s.name: s for s in training_set}


def _tiered_training_sets():
    by_name = _scenario_map()
    hard_holdout = []
    tier1_names = [
        'One Asteroid Still',
        'One Asteroid Slow Horizontal',
        'Two Asteroids Still',
        'Three Asteroids Row',
    ]
    tier2_names = [
        'Stock Scenario',
        'Donut Ring',
        'Donut Ring (Closing In, Large Asteroids)',
        'Vertical Wall Left (Big Moving Right)',
    ]
    tier3_names = [
        'Asteroid Rain',
        'Crossing Lanes',
        'Giants with Kamikaze',
        'Spiral Swarm',
        'Four Corner Assault',
    ]

    def present(names):
        return [by_name[n] for n in names if n in by_name]

    tier1 = present(tier1_names)
    tier2 = present(tier2_names)
    tier3 = present(tier3_names)
    hard = present(hard_holdout)
    already = {s.name for s in (tier1 + tier2 + tier3 + hard)}
    leftovers = [s for s in training_set if s.name not in already]
    tier3 = tier3 + leftovers
    return tier1, tier2, tier3, hard


def get_training_set_for_iteration(iteration: int):
    return list(training_set)


def _evaluate_one(genome, scenarios):
    genome = clamp_genome(genome)
    controller = GAFuzzyController(chromosome=genome)
    game = KesslerGame(settings=game_settings)

    total_score = 0.0
    total_weight = 0.0
    details = {}
    for scenario in scenarios:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        team = result.teams[0]
        raw = raw_score_team(team)
        weight = scenario_weight(scenario.name)
        weighted = raw * weight
        total_score += weighted
        total_weight += weight
        details[scenario.name] = {
            'raw': raw,
            'weight': weight,
            'weighted': weighted,
        }
    return total_score / max(total_weight, 1e-9), details


def train_fitness(args):
    genome, iteration = args
    active = get_training_set_for_iteration(int(iteration))
    total, _ = _evaluate_one(genome, active)
    return total


def validation_score_detailed(genome):
    return _evaluate_one(genome, validation_set)


def save_best(genome, iteration, out_dir, sigma, train_score, val_score,
              val_details=None, restarts=0, active_train_count=None, note=None):
    result = {
        'method': 'CMA-ES',
        'iteration': iteration,
        'train_score': train_score,
        'validation_score': val_score,
        'validation_details': val_details,
        'sigma': sigma,
        'restarts': restarts,
        'genome': list(clamp_genome(genome)),
        'n_genes': len(genome),
        'active_train_scenarios': active_train_count,
        'validation_scenarios': len(validation_set),
        'note': note,
        'objective': {
            'W_HIT': W_HIT,
            'W_ACC': W_ACC,
            'W_DEATH': W_DEATH,
            'W_SURVIVAL': W_SURVIVAL,
            'W_MINES': W_MINES,
            'W_TIME': W_TIME,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f'iter_{iteration:04d}.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    with (out_dir.parent / 'best_solution_cmaes.json').open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)


def _load_candidate(path: Path, score_key: str):
    if not path.exists():
        return None
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        genome = data.get('genome')
        if genome is None or len(genome) != N_GENES:
            return None
        score = float(data.get(score_key, float('-inf')))
        return {'path': path.name, 'genome': [float(x) for x in genome], 'score': score}
    except Exception:
        return None


def load_initial_genome():
    candidates = []
    ga = _load_candidate(Path(os.path.dirname(__file__), 'best_solution.json'), 'validation_score')
    if ga is not None:
        candidates.append(ga)
    cma_prev = _load_candidate(Path(os.path.dirname(__file__), 'best_solution_cmaes.json'), 'validation_score')
    if cma_prev is not None:
        candidates.append(cma_prev)

    if candidates:
        best = max(candidates, key=lambda x: x['score'])
        print(f"Loaded starting genome from {best['path']} (best saved validation = {best['score']:.4f})")
        return best['genome']

    print('Using default CMA-ES start at 0.5')
    return [0.5] * N_GENES


def clear_solution_history(output_dir: Path):
    if not output_dir.exists():
        return
    for p in output_dir.glob('iter_*.json'):
        p.unlink()


def make_es(x0, sigma0):
    return cma.CMAEvolutionStrategy(
        list(x0),
        float(sigma0),
        {
            'verb_log': 0,
            'verb_disp': 0,
            'bounds': [0.0, 1.0],
            'popsize': max(16, 4 + int(4 * (N_GENES ** 0.5))),
        },
    )


def mixed_restart_point(best_genome, blend=0.65):
    rnd = [random.random() for _ in range(N_GENES)]
    return clamp_genome([blend * b + (1.0 - blend) * r for b, r in zip(best_genome, rnd)])


def choose_restart_state(restart_count, best_val_genome, best_train_genome):
    # 1st restart: around validation-best, but much wider
    if restart_count == 1:
        return best_val_genome, 0.28, 'wide restart around validation-best'
    # 2nd restart: mixed point between best and random
    if restart_count == 2:
        return mixed_restart_point(best_val_genome, blend=0.60), 0.38, 'mixed restart (best/random)'
    # 3rd+ restart: fully random cold restart
    return [random.random() for _ in range(N_GENES)], 0.50, 'cold random restart'


def main():
    print('=' * 78)
    print('  GAFuzzyController — CMA-ES FINETUNING (VALIDATION-SAFE)')
    print('=' * 78)
    print(f'  Dimensions : {N_GENES}')
    print(f'  Max iters  : {MAX_ITERS}')
    print(f'  Sigma0     : {SIGMA0}')
    print(f'  Train set  : {len(training_set)} total (smooth curriculum)')
    print(f'  Val set    : {len(validation_set)}')
    print(f'  Workers    : {N_WORKERS}')
    print(f'  Restarts   : {MAX_RESTARTS} max, patience {RESTART_PATIENCE}')
    print('  Objective  : weighted mean of [1.2*hit + 0.4*acc - 0.45*deaths - 0.05*mines + 0.75 clean + time]')
    print('  Selection  : save champion by VALIDATION, not by train only')
    print('  Saves to   : best_solution_cmaes.json')
    print('=' * 78)
    print()

    x0 = load_initial_genome()
    out_dir = Path(os.path.dirname(__file__), 'solution_history_cmaes')
    clear_solution_history(out_dir)

    es = make_es(x0, SIGMA0)

    best_train_genome = clamp_genome(x0)
    best_train = train_fitness((best_train_genome, 0))

    best_val_genome = clamp_genome(x0)
    best_val, best_val_details = validation_score_detailed(best_val_genome)
    best_val_train = best_train

    save_best(best_val_genome, 0, out_dir, SIGMA0, best_val_train, best_val, best_val_details,
              restarts=0, active_train_count=len(get_training_set_for_iteration(0)),
              note='initial seed')

    validation_history = [best_val]
    no_val_improve_iters = 0
    no_train_improve_iters = 0
    restart_count = 0
    start_time = time.perf_counter()
    last_active_count = len(get_training_set_for_iteration(0))

    print(f"{'Iter':>5}  {'train*':>8}  {'val*':>8}  {'sigma':>8}  {'pop':>5}  {'rst':>3}  {'trainN':>6}  {'iter_t':>7}  {'elapsed':>9}")
    print('-' * 94)
    print(f"{0:>5}  {best_val_train:>8.4f}  {best_val:>8.4f}  {SIGMA0:>8.4f}  {es.popsize:>5}  {restart_count:>3}  {last_active_count:>6}  {'—':>7}  {fmt_duration(0):>9}")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        it = 0
        for it in range(1, MAX_ITERS + 1):
            t0 = time.perf_counter()
            active_train = get_training_set_for_iteration(it)
            active_count = len(active_train)
            if active_count != last_active_count:
                print(f"--> Curriculum phase change at iteration {it}: now using {active_count} training scenarios")
                last_active_count = active_count

            solutions = [clamp_genome(x) for x in es.ask()]
            fitnesses = list(executor.map(train_fitness, [(sol, it) for sol in solutions]))
            es.tell(solutions, [-f for f in fitnesses])

            idx = max(range(len(fitnesses)), key=lambda i: fitnesses[i])
            gen_best_train_genome = solutions[idx]
            gen_best_train = fitnesses[idx]

            if gen_best_train > best_train + 1e-9:
                best_train = gen_best_train
                best_train_genome = gen_best_train_genome
                no_train_improve_iters = 0
            else:
                no_train_improve_iters += 1

            # Validate the generation's train-best candidate and compare to current validation champion.
            cand_val, cand_val_details = validation_score_detailed(gen_best_train_genome)
            if cand_val > best_val + 1e-9:
                best_val = cand_val
                best_val_genome = gen_best_train_genome
                best_val_details = cand_val_details
                best_val_train = gen_best_train
                no_val_improve_iters = 0
            else:
                no_val_improve_iters += 1

            validation_history.append(best_val)

            iter_time = time.perf_counter() - t0
            elapsed = time.perf_counter() - start_time

            if it % VALIDATION_LOG_EVERY == 0:
                print('    Validation breakdown (current validation champion):')
                for name, info in best_val_details.items():
                    print(f"      {name}: raw={info['raw']:.4f}  w={info['weight']:.2f}  weighted={info['weighted']:.4f}")
                print(f'      weighted total: {best_val:.4f}')

            save_best(best_val_genome, it, out_dir, es.sigma, best_val_train, best_val, best_val_details,
                      restarts=restart_count, active_train_count=active_count,
                      note='validation champion')

            print(f"{it:>5}  {best_val_train:>8.4f}  {best_val:>8.4f}  {es.sigma:>8.4f}  {es.popsize:>5}  {restart_count:>3}  {active_count:>6}  {iter_time:>6.1f}s  {fmt_duration(elapsed):>9}")

            should_restart = (
                restart_count < MAX_RESTARTS and (
                    no_val_improve_iters >= RESTART_PATIENCE or
                    (no_train_improve_iters >= RESTART_PATIENCE and es.sigma <= 0.12)
                )
            )
            if should_restart:
                restart_count += 1
                x_restart, sigma_restart, label = choose_restart_state(
                    restart_count, best_val_genome, best_train_genome
                )
                no_train_improve_iters = 0
                no_val_improve_iters = 0
                print()
                print(
                    f'Restarting CMA-ES at iteration {it}: {label} '
                    f'with sigma={sigma_restart:.4f} (restart {restart_count}/{MAX_RESTARTS}).'
                )
                es = make_es(x_restart, sigma_restart)

            if no_val_improve_iters >= EARLY_STOP_PATIENCE:
                print()
                print(f'Early stopping at iteration {it}: no validation improvement for {EARLY_STOP_PATIENCE} iterations.')
                break

    final_val, final_val_details = validation_score_detailed(best_val_genome)
    save_best(best_val_genome, it, out_dir, es.sigma, best_val_train, final_val, final_val_details,
              restarts=restart_count, active_train_count=len(get_training_set_for_iteration(it)),
              note='final validation champion')

    print()
    print('=' * 78)
    print('  CMA-ES TRAINING COMPLETE')
    print('=' * 78)
    print(f'  Best train (champion) : {best_val_train:.4f}')
    print(f'  Best val score        : {final_val:.4f}')
    print(f'  Restarts used         : {restart_count}')
    print('  Saved to              : best_solution_cmaes.json')
    print('=' * 78)


if __name__ == '__main__':
    main()
