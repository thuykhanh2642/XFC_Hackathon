import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from kesslergame import KesslerGame, GraphicsType
from MyAIController.scenarios import *
from MyAIController.example_controller_fuzzy import MyFuzzyController

game_settings = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.Tkinter,
    'realtime_multiplier': 1,
    'graphics_obj': None,
    'frequency': 60
}

scenario_list = [
    one_asteroid_still(),
    one_asteroid_slow_horizontal(),
    two_asteroids_still(),
    three_asteroids_still_row(),
    donut_ring(),
    four_corner(),
    stock_scenario(),
    vertical_wall_left(),
    donut_ring_closing(),
    asteroid_rain(),
    moving_maze_right(),
    crossing_lanes(),
    spiral_arms(),
    giants_with_kamikaze()
]

for scenario in scenario_list:
    game = KesslerGame(settings=game_settings)

    pre = time.perf_counter()

    try:
        score, perf_data = game.run(
            scenario=scenario,
            controllers=[MyFuzzyController()]
        )

        print("Scenario eval time:", time.perf_counter() - pre)
        print("Stop reason:", score.stop_reason)
        print("Asteroids hit:", [team.asteroids_hit for team in score.teams])
        print("Deaths:", [team.deaths for team in score.teams])
        print("Accuracy:", [team.accuracy for team in score.teams])
        print("Mean eval time:", [team.mean_eval_time for team in score.teams])

    except Exception as e:
        print("Scenario skipped or window closed.")
        print("Error:", e)