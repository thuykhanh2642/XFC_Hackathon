# Fitness function for GAFuzzyController
#
# Score = fraction_hit * 2.0 + accuracy - deaths * 0.5
#
# Compared to the baseline (fraction_hit + accuracy):
#   - Doubles the weight on actually clearing asteroids
#   - Explicitly penalizes deaths — a controller that survives longer
#     has more time to shoot, so survival is already incentivized,
#     but this makes it explicit
#   - Accuracy bonus still rewards not wasting ammo

import sys
sys.path.append('.')

from kesslergame import KesslerGame, GraphicsType
from GAController1 import GAFuzzyController
from scenarios import training_set


def Fitness(individual, settings=None):
    controller = GAFuzzyController(chromosome=individual)
    total_score = 0.0

    game_settings = {
        'perf_tracker':          False,
        'graphics_type':         GraphicsType.NoGraphics,
        'realtime_multiplier':   0,
        'graphics_obj':          None,
        'frequency':             30,
        'time_limit':            30,
        'competition_safe_mode': False,
    }
    game = KesslerGame(settings=game_settings)

    for scenario in training_set:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        t = result.teams[0]
        score = (t.fraction_total_asteroids_hit * 2.0
                 + t.accuracy
                 - t.deaths * 0.5)
        total_score += score

    return total_score,


if __name__ == "__main__":
    import random
    test_chrom = [random.random() for _ in range(50)]
    print("Test fitness:", Fitness(test_chrom))