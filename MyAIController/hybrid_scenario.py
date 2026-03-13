# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.


"""HOW TO RUN
training: python kessler-game/neural_fuzzy/nf_train.py --task maneuver
        combat: python kessler-game/neural_fuzzy/nf_train.py --task combat
        optional args: --num_mfs 3 --lr 0.001 --batch_size 32 --val_frac 0.1 --epochs 100
inference: python kessler-game/neural_fuzzy/scenario_test.py
    
"""
import time
from kesslergame import Scenario, KesslerGame, GraphicsType
from hybrid_fuzzy import hybrid_controller
#from nf_controller import NFController
import scenarios as sc
#from human_controller import HumanController
#SCENARIO = sc.donut_ring()
#SCENARIO = sc.vertical_wall_left()
SCENARIO = sc.stock_scenario()
#SCENARIO = sc.spiral_arms()
#SCENARIO = sc.sniper_practice()
#SCENARIO = sc.crossing_lanes()
#SCENARIO = sc.asteroid_rain()
#SCENARIO = sc.giants_with_kamikaze()
#SCENARIO = sc.donut_ring_closing()
#SCENARIO = sc.four_corner()



game_settings = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.Tkinter,
    'realtime_multiplier': 1,
    'graphics_obj': None,
    'frequency': 30
}

game = KesslerGame(settings=game_settings)
pre = time.perf_counter()
score, perf_data = game.run(scenario=SCENARIO, controllers=[hybrid_controller()])
print('Scenario eval time:', time.perf_counter() - pre)
print(score.stop_reason)
print('Asteroids hit:', [team.asteroids_hit for team in score.teams])
print('Deaths:', [team.deaths for team in score.teams])
print('Accuracy:', [team.accuracy for team in score.teams])
print('Mean eval time:', [team.mean_eval_time for team in score.teams])
