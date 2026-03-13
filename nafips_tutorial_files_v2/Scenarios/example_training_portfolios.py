# -*- coding: utf-8 -*-
# Copyright © 2019-2020 Thales Avionics USA
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from kesslergame import Scenario

# Scenarios can be defined using the Scenario() class in fuzzy asteroids. Refer to
# examples below for creating your own or check out tas_scenarios.py for a large list
# of pre-made examples that you can be used. The "adv" prefix indicates a scenario has
# been set up for adversarial use (two competing ships)

adv_accuracy_test_1 = Scenario(
    name="adv_accuracy_test_1",
    asteroid_states=[{"position": (100, 700), "angle": 90, "speed": 165, "size": 1},
                     ],
    ship_states=[{"position": (900, 300), "team": 1}],
    seed=1, time_limit=7, ammo_limit_multiplier=5, stop_if_no_ammo=True
)

adv_accuracy_test_2 = Scenario(
    name="adv_accuracy_test_2",
    asteroid_states=[{"position": (100, 100), "angle": -90, "speed": 165, "size": 1},
                     ],
    ship_states=[{"position": (900, 300), "team": 1}],
    seed=1, time_limit=7, ammo_limit_multiplier=5, stop_if_no_ammo=True
)

adv_accuracy_test_3 = Scenario(
    name="adv_accuracy_test_3",
    asteroid_states=[{"position": (100, 700), "angle": 90, "speed": 165, "size": 1},
                     ],
    ship_states=[{"position": (900, 300), "team": 1}],
    seed=1, time_limit=7, ammo_limit_multiplier=5, stop_if_no_ammo=True
)

adv_accuracy_test_4 = Scenario(
    name="adv_accuracy_test_4",
    asteroid_states=[{"position": (100, 100), "angle": -90, "speed": 165, "size": 1},
                     ],
    ship_states=[{"position": (900, 300), "team": 1}],
    seed=1, time_limit=7, ammo_limit_multiplier=5, stop_if_no_ammo=True
)

adv_threat_test_1 = Scenario(
    name="adv_threat_test_1",
    asteroid_states=[{"position": (0, 300), "angle": 0.0, "speed": 40},
                     {"position": (700, 300), "angle": 0.0, "speed": 0},
                     ],
    ship_states=[{"position": (600, 400), "team": 1},
                 {"position": (600, 200), "team": 2}],
    seed=1, time_limit=60, ammo_limit_multiplier=1.2, stop_if_no_ammo=True
)

adv_threat_test_2 = Scenario(
    name="adv_threat_test_2",
    asteroid_states=[{"position": (800, 300), "angle": 180.0, "speed": 40},
                     {"position": (100, 300), "angle": 0.0, "speed": 0},
                     ],
    ship_states=[{"position": (200, 400), "team": 1},
                 {"position": (200, 200), "team": 2}],
    seed=1, time_limit=60, ammo_limit_multiplier=1.2, stop_if_no_ammo=True
)

adv_threat_test_3 = Scenario(
    name="adv_threat_test_3",
    asteroid_states=[{"position": (400, 0), "angle": 90.0, "speed": 40},
                     {"position": (400, 550), "angle": -90.0, "speed": 0},
                     ],
    ship_states=[{"position": (300, 450), "team": 1},
                 {"position": (500, 450), "team": 2}],
    seed=1, time_limit=60, ammo_limit_multiplier=1.2, stop_if_no_ammo=True
)

adv_threat_test_4 = Scenario(
    name="adv_threat_test_4",
    asteroid_states=[{"position": (400, 600), "angle": -90.0, "speed": 40},
                     {"position": (400, 50), "angle": -90.0, "speed": 0},
                     ],
    ship_states=[{"position": (300, 150), "team": 1},
                 {"position": (500, 500), "team": 2}],
    seed=1, time_limit=60, ammo_limit_multiplier=1.2, stop_if_no_ammo=True
)

# multi-asteroid but basic scenarios
adv_square_1 = Scenario(
    name="adv_square_1",
    asteroid_states=[{"position": (100, 100), "angle": -90.0, "speed": 0},
                     {"position": (100, 700), "angle": -90.0, "speed": 0},
                     {"position": (900, 100), "angle": -90.0, "speed": 0},
                     {"position": (900, 700), "angle": -90.0, "speed": 0},
                     ],
    ship_states=[{"position": (400, 400), "angle": 0, "lives": 3, "team": 1},
                 {"position": (600, 400), "angle": -180, "lives": 3, "team": 2},
                 ],
    seed=1, time_limit=60, ammo_limit_multiplier=1.2, stop_if_no_ammo=True
)

adv_square_2 = Scenario(
    name="adv_square_1",
    asteroid_states=[{"position": (350, 450), "angle": -90.0, "speed": 0},
                     {"position": (450, 550), "angle": -90.0, "speed": 0},
                     {"position": (450, 450), "angle": -90.0, "speed": 0},
                     {"position": (350, 550), "angle": -90.0, "speed": 0},
                     ],
    ship_states=[{"position": (100, 100), "angle": 0.0, "lives": 3, "team": 1},
                 {"position": (900, 100), "angle": -180, "lives": 3, "team": 2},
                 ],
    seed=1, time_limit=60, ammo_limit_multiplier=1.2, stop_if_no_ammo=True
)



# wall scenarios
adv_wall_bottom_easy = Scenario(
    name="adv_wall_bottom_easy",
    asteroid_states=[{"position": (100, 0), "angle": -90.0, "speed": 60},
                     {"position": (200, 0), "angle": -90.0, "speed": 60},
                     {"position": (300, 0), "angle": -90.0, "speed": 60},
                     {"position": (400, 0), "angle": -90.0, "speed": 60},
                     {"position": (500, 0), "angle": -90.0, "speed": 60},
                     {"position": (600, 0), "angle": -90.0, "speed": 60},
                     {"position": (700, 0), "angle": -90.0, "speed": 60},
                     {"position": (800, 0), "angle": -90.0, "speed": 60},
                     {"position": (900, 0), "angle": -90.0, "speed": 60},
                     ],
    ship_states=[{"position": (100, 400), "angle": 0, "lives": 3, "team": 1},
                 {"position": (700, 400), "angle": -180, "lives": 3, "team": 2},
                 ],
    seed=1, time_limit=120, ammo_limit_multiplier=1.2, stop_if_no_ammo=True
)


#-----------------------------------------------------------------------------------------------------------------------
# Portfolio definition - list of scenario objects
#-----------------------------------------------------------------------------------------------------------------------

training_portfolio = [adv_accuracy_test_1,
                      adv_accuracy_test_2,
                      adv_accuracy_test_3,
                      adv_accuracy_test_4,
                      adv_threat_test_1,
                      adv_threat_test_2,
                      adv_threat_test_3,
                      adv_threat_test_4,
                      adv_square_1,
                      adv_square_2,
                      adv_wall_bottom_easy]

test_portfolio = []