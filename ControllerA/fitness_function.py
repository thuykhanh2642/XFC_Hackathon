# -*- coding: utf-8 -*-
# Fitness function for SuperiorFuzzyController
#
# Score per scenario = fraction_asteroids_hit + accuracy
# (same metric as the baseline so results are directly comparable)
#
# Usage: drop this file next to example_fitness_function.py

import sys
sys.path.append('.')

from kesslergame import KesslerGame, GraphicsType
from GAController import GAFuzzyController
from scenarios import training_set


def Fitness(individual, settings=None):
    controller = GAFuzzyController(chromosome=individual)
    total_score = 0.0

    game_settings = {
        'perf_tracker':         False,
        'graphics_type':        GraphicsType.NoGraphics,
        'realtime_multiplier':  0,
        'graphics_obj':         None,
        'frequency':            30,
        'time_limit':           30,
        'competition_safe_mode': False,
    }
    game = KesslerGame(settings=game_settings)

    for scenario in training_set:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        score = (result.teams[0].fraction_total_asteroids_hit
                 + result.teams[0].accuracy)
        total_score += score

    return total_score,


if __name__ == "__main__":
    import random
    test_chrom = [random.random() for _ in range(50)]
    print("Test chromosome fitness:", Fitness(test_chrom))