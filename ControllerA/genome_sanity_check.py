import sys
sys.path.append('.')

import json
import os
import random
from pathlib import Path

from kesslergame import KesslerGame, GraphicsType
from HybridController2 import GAFuzzyController
from scenarios import validation_set

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
    "Stock": 1.10,
    "Donut Ring": 1.15,
    "Vertical Wall": 1.20,
    "Four Corner Assault": 1.25,
    "Cross (Rotating Look, CW)": 2.00,
    "Moving Maze (Rightward Tunnel)": 2.50,
}

game_settings = {
    'perf_tracker': False,
    'graphics_type': GraphicsType.NoGraphics,
    'realtime_multiplier': 0,
    'graphics_obj': None,
    'frequency': 30,
    'time_limit': 30,
    'competition_safe_mode': False,
}

def scenario_weight(name):
    return float(SCENARIO_WEIGHTS.get(name, 1.0))

def survival_fraction(team):
    for attr in ('time_alive', 'survival_time', 'time_survived', 'sim_time_alive', 'seconds_alive'):
        value = getattr(team, attr, None)
        if value is not None:
            try:
                return float(max(0.0, min(float(value) / 30.0, 1.0)))
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

def eval_genome(genome, label):
    controller = GAFuzzyController(chromosome=genome)
    game = KesslerGame(settings=game_settings)
    total_weighted = 0.0
    total_weight = 0.0
    details = []
    for scenario in validation_set:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        team = result.teams[0]
        raw = raw_score_team(team)
        weight = scenario_weight(scenario.name)
        weighted = raw * weight
        total_weighted += weighted
        total_weight += weight
        details.append((scenario.name, raw, weighted))
    mean = total_weighted / max(total_weight, 1e-9)
    print(f'{label:>12}: weighted val = {mean: .4f}')
    for name, raw, weighted in details:
        print(f'  {name}: raw={raw:.4f} weighted={weighted:.4f}')
    print()
    return mean

def load_json_genome(filename):
    path = Path(os.path.dirname(__file__), filename)
    if not path.exists():
        return None
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    genome = data.get('genome')
    if genome and len(genome) == 50:
        return [float(x) for x in genome]
    return None

def main():
    best_ga = load_json_genome('best_solution.json')
    best_cma = load_json_genome('best_solution_cmaes.json')
    neutral = [0.5] * 50
    rnd = [random.random() for _ in range(50)]

    if best_ga is not None:
        eval_genome(best_ga, 'best_ga')
    if best_cma is not None:
        eval_genome(best_cma, 'best_cma')
    eval_genome(neutral, 'neutral_0.5')
    eval_genome(rnd, 'random')

if __name__ == '__main__':
    main()
