#type: ignore
import sys
sys.path.append('.')

import json
import os
from pathlib import Path
from kesslergame import KesslerGame, GraphicsType
from HybridController2 import GAFuzzyController
from scenarios import training_set, validation_set

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
    "Sniper Practice (Large Arena)": 0.90,
    "Stock Scenario": 1.10,
    "Donut Ring": 1.15,
    "Donut Ring (Closing In, Large Asteroids)": 1.15,
    "Vertical Wall Left (Big Moving Right)": 1.20,
    "Asteroid Rain": 1.10,
    "Crossing Lanes": 1.20,
    "Giants with Kamikaze": 1.25,
    "Spiral Swarm": 1.15,
    "Four Corner Assault": 1.25,
    "Cross (Rotating Look, CW)": 2.00,
    "Moving Maze (Rightward Tunnel)": 2.50,
}

game_settings = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.Tkinter,
    'realtime_multiplier': 1,
    'graphics_obj': None,
    'frequency': 30,
    'time_limit': 60,
    'competition_safe_mode': False,
}


def scenario_weight(name):
    return float(SCENARIO_WEIGHTS.get(name, 1.0))


def survival_fraction(team):
    candidate_attrs = (
        'time_alive', 'survival_time', 'time_survived', 'sim_time_alive', 'seconds_alive'
    )
    for attr in candidate_attrs:
        value = getattr(team, attr, None)
        if value is not None:
            try:
                return float(max(0.0, min(float(value) / 60.0, 1.0)))
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


def run_set(game, controller, scenarios, label):
    total_weighted = 0.0
    total_weight = 0.0
    print(f'\n=== {label} ===')
    for scenario in scenarios:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        t = result.teams[0]
        raw = raw_score_team(t)
        weight = scenario_weight(scenario.name)
        weighted = raw * weight
        total_weighted += weighted
        total_weight += weight
        mines_used = float(getattr(t, 'mines_used', 0.0) or 0.0)
        print(
            f"{scenario.name}: raw={raw:.4f}  w={weight:.2f}  weighted={weighted:.4f}  "
            f"hit={t.asteroids_hit}  acc={t.accuracy:.3f}  deaths={t.deaths}  mines={mines_used:.0f}"
        )
    total = total_weighted / max(total_weight, 1e-9)
    print(f'Weighted mean {label.lower()} score: {total:.4f}')
    return total


def main():
    json_path = Path(os.path.dirname(__file__), 'best_solution_cmaes.json')

    if not json_path.exists():
        print(f'No CMA-ES solution found at {json_path}.')
        print('Run cmaes_train_patched.py first.')
        return

    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    genome = [float(x) for x in data['genome']]
    print(f"Loaded CMA-ES iteration {data.get('iteration', '?')} — training score {data.get('train_score', 0.0):.4f}")
    print(f"Saved validation score: {data.get('validation_score', 0.0):.4f}")

    controller = GAFuzzyController(chromosome=genome)
    game = KesslerGame(settings=game_settings)

    train_total = run_set(game, controller, training_set, 'Training Set')
    val_total = run_set(game, controller, validation_set, 'Validation Set')

    print('\nSummary')
    print(f'Train weighted mean: {train_total:.4f}')
    print(f'Validation weighted mean: {val_total:.4f}')


if __name__ == '__main__':
    main()