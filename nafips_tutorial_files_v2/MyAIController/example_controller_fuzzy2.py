# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from kesslergame import KesslerController
from .sa.util.helpers import trim_angle
from .sa.sa import SA
from typing import Dict, Tuple
import skfuzzy.control as ctrl
import skfuzzy as skf
import numpy as np
import math


class MyFuzzyController2(KesslerController):
    def __init__(self, chromosome=None):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        ...
        self.chromosome = chromosome
        self.sa = SA()
        self.aiming_fis = None
        self.aiming_fis_sim = None
        self.normalization_dist = None

        self.create_fuzzy_systems()

    def create_asteroid_threat_fis(self):
        # If we don't have a chromosome to get values from, use "default" values
        if not self.chromosome:
            # input 1 - distance to asteroid
            distance = ctrl.Antecedent(np.linspace(0.0, 1.0, 11), "distance")
            # input 2 - closure rate (speed towards to ship)
            closure_rate = ctrl.Antecedent(np.linspace(0.0, 1.0, 11), "closure_rate")
            # input 3 - relative bearing - angle between asteroids heading and to us
            relative_bearing = ctrl.Antecedent(np.linspace(-1.0, 1.0, 11), "relative_bearing")

            # output - perceived threat level of any given asteroid
            threat_level = ctrl.Consequent(np.linspace(-1.0, 1.0, 11), "threat_level")

            # creating 3 equally spaced membership functions for the inputs
            distance.automf(3, names=["close", "medium", "far"])
            closure_rate.automf(3, names=["low", "medium", "high"])
            # note we are ignoring negative/zero closure rate because that means moving away from us
            relative_bearing.automf(3, names=["zero", "small", "big"])

            # creating 3 triangular membership functions for the output
            threat_level["low"] = skf.trimf(threat_level.universe, [-1.0, -1.0, 0.0])
            threat_level["medium"] = skf.trimf(threat_level.universe, [-1.0, 0.0, 1.0])
            threat_level["high"] = skf.trimf(threat_level.universe, [0.0, 1.0, 1.0])

            # creating the rule base for the fuzzy system
            rule1 = ctrl.Rule(distance["close"] & closure_rate["low"] & relative_bearing["zero"], threat_level["medium"])
            rule2 = ctrl.Rule(distance["medium"] & closure_rate["low"] & relative_bearing["zero"], threat_level["medium"])
            rule3 = ctrl.Rule(distance["far"] & closure_rate["low"] & relative_bearing["zero"], threat_level["low"])
            rule4 = ctrl.Rule(distance["close"] & closure_rate["medium"] & relative_bearing["zero"], threat_level["high"])
            rule5 = ctrl.Rule(distance["medium"] & closure_rate["medium"] & relative_bearing["zero"], threat_level["medium"])
            rule6 = ctrl.Rule(distance["far"] & closure_rate["medium"] & relative_bearing["zero"], threat_level["low"])
            rule7 = ctrl.Rule(distance["close"] & closure_rate["high"] & relative_bearing["zero"], threat_level["high"])
            rule8 = ctrl.Rule(distance["medium"] & closure_rate["high"] & relative_bearing["zero"], threat_level["high"])
            rule9 = ctrl.Rule(distance["far"] & closure_rate["high"] & relative_bearing["zero"], threat_level["medium"])

            rule10 = ctrl.Rule(distance["close"] & closure_rate["low"] & relative_bearing["small"], threat_level["medium"])
            rule11 = ctrl.Rule(distance["medium"] & closure_rate["low"] & relative_bearing["small"], threat_level["low"])
            rule12 = ctrl.Rule(distance["far"] & closure_rate["low"] & relative_bearing["small"], threat_level["low"])
            rule13 = ctrl.Rule(distance["close"] & closure_rate["medium"] & relative_bearing["small"], threat_level["high"])
            rule14 = ctrl.Rule(distance["medium"] & closure_rate["medium"] & relative_bearing["small"], threat_level["medium"])
            rule15 = ctrl.Rule(distance["far"] & closure_rate["medium"] & relative_bearing["small"], threat_level["low"])
            rule16 = ctrl.Rule(distance["close"] & closure_rate["high"] & relative_bearing["small"], threat_level["high"])
            rule17 = ctrl.Rule(distance["medium"] & closure_rate["high"] & relative_bearing["small"], threat_level["medium"])
            rule18 = ctrl.Rule(distance["far"] & closure_rate["high"] & relative_bearing["small"], threat_level["medium"])

            rule19 = ctrl.Rule(distance["close"] & closure_rate["low"] & relative_bearing["big"], threat_level["low"])
            rule20 = ctrl.Rule(distance["medium"] & closure_rate["low"] & relative_bearing["big"], threat_level["low"])
            rule21 = ctrl.Rule(distance["far"] & closure_rate["low"] & relative_bearing["big"], threat_level["low"])
            rule22 = ctrl.Rule(distance["close"] & closure_rate["medium"] & relative_bearing["big"], threat_level["medium"])
            rule23 = ctrl.Rule(distance["medium"] & closure_rate["medium"] & relative_bearing["big"], threat_level["low"])
            rule24 = ctrl.Rule(distance["far"] & closure_rate["medium"] & relative_bearing["big"], threat_level["low"])
            rule25 = ctrl.Rule(distance["close"] & closure_rate["high"] & relative_bearing["big"], threat_level["medium"])
            rule26 = ctrl.Rule(distance["medium"] & closure_rate["high"] & relative_bearing["big"], threat_level["low"])
            rule27 = ctrl.Rule(distance["far"] & closure_rate["high"] & relative_bearing["big"], threat_level["low"])

            rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
                     rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18,
                     rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27]
            # creating a FIS controller from the rules + membership functions
            self.threat_level_fis = ctrl.ControlSystem(rules)
            # creating a controller sim to evaluate the FIS
            self.threat_level_fis_sim = ctrl.ControlSystemSimulation(self.threat_level_fis)
        else:
            # create FIS using GA chromosome
            # input 1 - distance to asteroid
            distance = ctrl.Antecedent(np.linspace(0.0, 1.0, 11), "distance")
            # input 2 - closure rate (speed towards to ship)
            closure_rate = ctrl.Antecedent(np.linspace(0.0, 1.0, 11), "closure_rate")
            # input 3 - relative bearing - angle between asteroids heading and to us
            relative_bearing = ctrl.Antecedent(np.linspace(0.0, 1.0, 11), "relative_bearing")

            # output - perceived threat level of any given asteroid
            threat_level = ctrl.Consequent(np.linspace(0.0, 1.0, 11), "threat_level")

            # Create membership functions from chromosome - Note that we're constraining the triangular membership
            # functions to have Ruspini partitioning
            # create distance membership functions from chromosome
            distance["close"] = skf.trimf(distance.universe, [-1.0, -1.0, self.chromosome[0]])
            distance["medium"] = skf.trimf(distance.universe, [-1.0, self.chromosome[0], 1.0])
            distance["far"] = skf.trimf(distance.universe, [self.chromosome[0], 1.0, 1.0])
            # create closure rate membership functions from chromosome
            closure_rate["low"] = skf.trimf(closure_rate.universe, [-1.0, -1.0, self.chromosome[1]])
            closure_rate["medium"] = skf.trimf(closure_rate.universe, [-1.0, self.chromosome[1], 1.0])
            closure_rate["high"] = skf.trimf(closure_rate.universe, [self.chromosome[1], 1.0, 1.0])
            # create relative bearing membership functions from chromosome
            relative_bearing["zero"] = skf.trimf(relative_bearing.universe, [-1.0, -1.0, self.chromosome[2]])
            relative_bearing["small"] = skf.trimf(relative_bearing.universe, [-1.0, self.chromosome[2], 1.0])
            relative_bearing["big"] = skf.trimf(relative_bearing.universe, [self.chromosome[2], 1.0, 1.0])

            # creating 3 triangular membership functions for the output
            threat_level["low"] = skf.trimf(threat_level.universe, [-1.0, -1.0, self.chromosome[3]*2-1])
            threat_level["medium"] = skf.trimf(threat_level.universe, [-1.0, self.chromosome[3]*2-1, 1.0])
            threat_level["high"] = skf.trimf(threat_level.universe, [self.chromosome[3]*2-1, 1.0, 1.0])

            input1_mfs = [distance["close"], distance["medium"], distance["far"]]
            input2_mfs = [closure_rate["low"], closure_rate["medium"], closure_rate["high"]]
            input3_mfs = [relative_bearing["zero"], relative_bearing["small"], relative_bearing["big"]]

            # create list of output membership functions to index into to create rule antecedents
            output_mfs = [threat_level["low"], threat_level["medium"], threat_level["high"]]

            # bin the values associated with rules - this is done so we can use the floats in the chromosome DNA
            # associated with the output membership functions in order to index into our predefined output membership
            # function set - i.e. the "output_mfs" list
            bins = np.array([0.0, 0.33333, 0.66666, 1.0])
            num_mfs1 = len(input1_mfs)
            num_mfs2 = len(input2_mfs)
            num_mfs3 = len(input3_mfs)
            num_rules = num_mfs1*num_mfs2*num_mfs3
            # grabbing the corresponding DNA values that determine the output mfs from the chromosome
            rules_raw = self.chromosome[3:3+num_rules]
            # binning the values to convert the floats to integer values to be used as indices
            ind = np.digitize(rules_raw, bins, right=True)-1
            ind = [int(min(max(idx, 0), 2)) for idx in ind]
            print(rules_raw)
            print(ind)

            count = 0
            # mapping the DNA indices to output_mfs
            try:
                rule_consequents_linear = [output_mfs[idx] for idx in ind]
            except:
                print(ind)
                print(rules_raw)
            # constructing the rules by combining our antecedents (conjunction of input mfs) with the corresponding
            # consequents (output mfs)
            rules = []
            # for ii in range(num_mfs1):
            #     for jj in range(num_mfs2):
            #         for kk in range(num_mfs3):
            #             rules.append(ctrl.Rule(input1_mfs[ii] & input2_mfs[jj] & input3_mfs[kk], rule_consequents_linear[count]))
            for kk in range(num_mfs3):
                for jj in range(num_mfs2):
                    for ii in range(num_mfs1):
                        rules.append(ctrl.Rule(input1_mfs[ii] & input2_mfs[jj] & input3_mfs[kk], rule_consequents_linear[count]))


            # creating a FIS controller from the rules + membership functions
            self.threat_level_fis = ctrl.ControlSystem(rules)
            # creating a controller sim to evaluate the FIS
            self.threat_level_fis_sim = ctrl.ControlSystemSimulation(self.threat_level_fis)

    def create_aiming_firing_fis(self):
        # Note: This is Scott Dick's work and is licensed under the Apache License Version 2.0
        # self.targeting_control is the targeting rulebase, which is static in this controller.
        # Declare variables
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1 * math.pi / 30, math.pi / 30, 0.1),
                                      'theta_delta')  # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')  # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')

        # Declare fuzzy sets for bullet_time (how long it takes for the bullet to reach the intercept point)
        bullet_time['S'] = skf.trimf(bullet_time.universe, [0, 0, 0.05])
        bullet_time['M'] = skf.trimf(bullet_time.universe, [0, 0.05, 0.1])
        bullet_time['L'] = skf.smf(bullet_time.universe, 0.0, 0.1)

        # Declare fuzzy sets for theta_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        theta_delta['NL'] = skf.zmf(theta_delta.universe, -1 * math.pi / 30, -2 * math.pi / 90)
        theta_delta['NM'] = skf.trimf(theta_delta.universe, [-1 * math.pi / 30, -2 * math.pi / 90, -1 * math.pi / 90])
        theta_delta['NS'] = skf.trimf(theta_delta.universe, [-2 * math.pi / 90, -1 * math.pi / 90, math.pi / 90])
        # theta_delta['Z'] = skf.trimf(theta_delta.universe, [-1*math.pi/90,0,math.pi/90])
        theta_delta['PS'] = skf.trimf(theta_delta.universe, [-1 * math.pi / 90, math.pi / 90, 2 * math.pi / 90])
        theta_delta['PM'] = skf.trimf(theta_delta.universe, [math.pi / 90, 2 * math.pi / 90, math.pi / 30])
        theta_delta['PL'] = skf.smf(theta_delta.universe, 2 * math.pi / 90, math.pi / 30)

        # Declare fuzzy sets for the ship_turn consequent; this will be returned as turn_rate.
        # Hard-coded for a game step of 1/30 seconds
        ship_turn['NL'] = skf.trimf(ship_turn.universe, [-180, -180, -120])
        ship_turn['NM'] = skf.trimf(ship_turn.universe, [-180, -120, -60])
        ship_turn['NS'] = skf.trimf(ship_turn.universe, [-120, -60, 60])
        # ship_turn['Z'] = skf.trimf(ship_turn.universe, [-60,0,60])
        ship_turn['PS'] = skf.trimf(ship_turn.universe, [-60, 60, 120])
        ship_turn['PM'] = skf.trimf(ship_turn.universe, [60, 120, 180])
        ship_turn['PL'] = skf.trimf(ship_turn.universe, [120, 180, 180])

        # Declare singleton fuzzy sets for the ship_fire consequent; -1 -> don't fire, +1 -> fire; this will be
        # thresholded
        #   and returned as the boolean 'fire'
        ship_fire['N'] = skf.trimf(ship_fire.universe, [-1, -1, 0.0])
        ship_fire['Y'] = skf.trimf(ship_fire.universe, [0.0, 1, 1])

        # Declare each fuzzy rule
        rule1 = ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule2 = ctrl.Rule(bullet_time['L'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule3 = ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule4 = ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule5 = ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule6 = ctrl.Rule(bullet_time['L'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule7 = ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule8 = ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N']))
        rule9 = ctrl.Rule(bullet_time['M'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['N']))
        rule10 = ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule11 = ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule12 = ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule13 = ctrl.Rule(bullet_time['M'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['N']))
        rule14 = ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N']))
        rule15 = ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y']))
        rule16 = ctrl.Rule(bullet_time['S'] & theta_delta['NM'], (ship_turn['NM'], ship_fire['Y']))
        rule17 = ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y']))
        # rule18 = ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y']))
        rule19 = ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y']))
        rule20 = ctrl.Rule(bullet_time['S'] & theta_delta['PM'], (ship_turn['PM'], ship_fire['Y']))
        rule21 = ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y']))

        # DEBUG
        # bullet_time.view()
        # theta_delta.view()
        # ship_turn.view()
        # ship_fire.view()

        # Declare the fuzzy controller, add the rules
        # This is an instance variable, and thus available for other methods in the same object. See notes.
        # self.targeting_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])

        self.aiming_control = ctrl.ControlSystem()
        self.aiming_control.addrule(rule1)
        self.aiming_control.addrule(rule2)
        self.aiming_control.addrule(rule3)
        # self.aiming_control.addrule(rule4)
        self.aiming_control.addrule(rule5)
        self.aiming_control.addrule(rule6)
        self.aiming_control.addrule(rule7)
        self.aiming_control.addrule(rule8)
        self.aiming_control.addrule(rule9)
        self.aiming_control.addrule(rule10)
        # self.aiming_control.addrule(rule11)
        self.aiming_control.addrule(rule12)
        self.aiming_control.addrule(rule13)
        self.aiming_control.addrule(rule14)
        self.aiming_control.addrule(rule15)
        self.aiming_control.addrule(rule16)
        self.aiming_control.addrule(rule17)
        # self.aiming_control.addrule(rule18)
        self.aiming_control.addrule(rule19)
        self.aiming_control.addrule(rule20)
        self.aiming_control.addrule(rule21)


    def create_fuzzy_systems(self):
        self.create_aiming_firing_fis()
        self.create_asteroid_threat_fis()

    def get_asteroid_distances(self, ship_state: Dict, game_state: Dict):
        # get distances to all asteroids in the game from ownship
        asteroid_dists = [np.sqrt((ship_state["position"][0] - asteroid["position"][0])**2 + (ship_state["position"][1] - asteroid["position"][1])**2) for asteroid in game_state["asteroids"]]
        return asteroid_dists

    def find_nearest_asteroid(self, ship_state: Dict, game_state: Dict):
        # create list of distances from the ship to each asteroid
        asteroid_dists = self.get_asteroid_distances(ship_state=ship_state, game_state=game_state)
        # get minimum distance
        ast_dist = min(asteroid_dists)
        # get index in list of nearest asteroid
        ast_idx = asteroid_dists.index(ast_dist)

        return ast_idx, ast_dist

    def compute_threat_level(self, ast_distance, closure_rate, ast_bearing):
        self.threat_level_fis_sim.input["distance"] = ast_distance
        self.threat_level_fis_sim.input["closure_rate"] = closure_rate
        self.threat_level_fis_sim.input["relative_bearing"] = ast_bearing
        self.threat_level_fis_sim.compute()
        return self.threat_level_fis_sim.output["threat_level"]

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller to determine what control actions to take

        Arguments:
            ship_state (dict): contains state information for your own ship
            game_state (dict): contains state information for all objects in the game

        Returns:
            float: thrust control value
            float: turn-rate control value
            bool: fire control value. Shoots if true
            bool: mine deployment control value. Lays mine if true
        """

        # update the situational awareness with current information
        self.sa.update(ship_state, game_state)

        # get the nearest 5 asteroid
        nearest_n_asteroid = self.sa.ownship.nearest_n(5)

        # calculate threat levels for all asteroids
        threat_levels = []
        for asteroid_idx in range(len(nearest_n_asteroid)):
            ast_distance = nearest_n_asteroid[asteroid_idx].distance_wrap
            closure_rate = nearest_n_asteroid[asteroid_idx].ship_closure_rate_wrap
            rel_bearing = nearest_n_asteroid[asteroid_idx].bearing - self.sa.ownship.heading

            threat_levels.append(self.compute_threat_level(ast_distance, closure_rate, rel_bearing))

        max_threat_idx = max(enumerate(threat_levels), key=lambda x: x[1])[0]

        # Set the target asteroid to the highest "threat" asteroid rather than just the nearest asteroid
        target_asteroid = nearest_n_asteroid[max_threat_idx]
        # target_asteroid = self.sa.ownship.nearest_n(1)[0]

        rel_bearing = trim_angle(self.sa.ownship.heading - target_asteroid.bearing)

        # if desired aim angle is outside of +- 1 deg, turn in that direction with max turn rate, otherwise don't turn
        turn_rate = 0
        if rel_bearing < -0.5:
            turn_rate = ship_state["turn_rate_range"][1]
        elif rel_bearing > 0.5:
            turn_rate = ship_state["turn_rate_range"][0]

        # set firing to always be true (fires as often as possible), all other values to 0
        thrust = 0
        fire = True
        drop_mine = False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        """
        Simple property used for naming controllers such that it can be displayed in the graphics engine

        Returns:
            str: name of this controller
        """
        return "Fuzzy Test1"

    # @property
    # def custom_sprite_path(self) -> str:
    #     return "Neo.png"
