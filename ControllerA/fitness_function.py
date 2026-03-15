# -*- coding: utf-8 -*-
# Fitness function for SuperiorFuzzyController
#
# Score per scenario matches training_script.py
# (hit + accuracy reward, death/mine penalties, clean-run bonus)
#
# Usage: drop this file next to example_fitness_function.py

import sys
sys.path.append('.')

from kesslergame import KesslerGame, GraphicsType
from HybridController2 import GAFuzzyController
from scenarios import training_set


def Fitness(individual, settings=None):
    controller = GAFuzzyController(chromosome=individual)

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
    }

    game_settings = {
        'perf_tracker':         False,
        'graphics_type':        GraphicsType.NoGraphics,
        'realtime_multiplier':  0,
        'graphics_obj':         None,
        'frequency':            30,
        'time_limit':           20,
        'competition_safe_mode': False,
    }
    game = KesslerGame(settings=game_settings)

    total_weighted = 0.0
    total_weight = 0.0
    for scenario in training_set:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        t = result.teams[0]
        mines_used = float(getattr(t, "mines_used", 0.0) or 0.0)
        score = (
            W_HIT * float(t.fraction_total_asteroids_hit)
            + W_ACC * float(t.accuracy)
            - W_DEATH * float(t.deaths)
            - W_MINES * mines_used
            + W_TIME * max(0.0, min(1.0, 0.0))  # survival_fraction placeholder
        )
        if int(t.deaths) == 0:
            score += W_SURVIVAL
        weight = float(SCENARIO_WEIGHTS.get(scenario.name, 1.0))
        total_weighted += score * weight
        total_weight += weight

    return total_weighted / max(total_weight, 1e-9),



if __name__ == "__main__":
    import random
    test_chrom = [random.random() for _ in range(50)]
    print("Test chromosome fitness:", Fitness(test_chrom))