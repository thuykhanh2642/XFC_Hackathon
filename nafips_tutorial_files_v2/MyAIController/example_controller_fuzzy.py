# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from kesslergame import KesslerController
from typing import Dict, Tuple
# from MyAIController.ai.sa.sa import SA
from .sa.sa import SA
from .sa.util.helpers import trim_angle
import skfuzzy.control as ctrl
import skfuzzy as skf
import numpy as np


class MyFuzzyController(KesslerController):
    def __init__(self, chromosome=None):
        """
        Any variables or initialization desired for the controller can be set up here
        """
        ...
        self.chromosome = chromosome # this leaves chromosome object as None - will trigger default behavior
        # or you can define your chromosome values (i.e. paste from your best found solution json) so your controller is
        # the same as the one found through your training.
        # self.chromosome = [
        #     0.2631645225918112,
        #     0.9352976507138837,
        #     0.9194881615834452,
        #     0.865825520047121,
        #     0.9537717256003297,
        #     0.3332987471941833,
        #     0.15500677821329611,
        #     0.6103279406781605,
        #     0.700110463994626,
        #     0.28645661550432655,
        #     0.7817669414563869,
        #     0.3242896160611243,
        #     0.2193792619771311,
        #     0.9187346087146049,
        #     0.7565990950849348,
        #     0.378982875637626,
        #     0.20469566850073984,
        #     0.4221919062502967,
        #     0.09472152646554266,
        #     0.22444174766372227,
        #     0.43972091554025095,
        #     0.3024938255921993,
        #     0.03951149080348482,
        #     0.07014695710862251,
        #     0.5085111197870378,
        #     0.7217226903052553,
        #     0.8478160198517868,
        #     0.9622056814666377,
        #     0.9292672901679256,
        #     0.8763434435737334,
        #     0.8715824359186305,
        #     0.622211987772582,
        #     0.09103610636756931,
        #     0.42932650446832776,
        #     0.11151119794790532,
        #     0.6449315965771553,
        #     0.39484371578702315,
        #     0.5792033115832056,
        #     0.31492571381606194,
        #     0.6833570548284794,
        #     0.8625524374751714,
        #     0.8127256586152101,
        #     0.2550469991459845,
        #     0.19763655967886684,
        #     0.8740087388142191,
        #     0.9902384288505531,
        #     0.8321982213704795,
        #     0.6787979639950257,
        #     0.3945538617015545,
        #     0.8955970115205919
        #   ]
        self.aiming_fis = None
        self.aiming_fis_sim = None
        self.normalization_dist = None
        self.sa = SA()

        # I put this in a separate function for cleanliness in the init procedure, but this just calls the functions
        # that create your FIS functions
        self.create_fuzzy_systems()

    def create_aiming_fis(self):
        # If we don't have a chromosome to get values from, use "default" values
        if not self.chromosome:
            # input 1 - distance to asteroid
            distance = ctrl.Antecedent(np.linspace(0.0, 1.0, 11), "distance")
            # input 2 - angle to asteroid (relative to ship heading)
            angle = ctrl.Antecedent(np.linspace(-1.0, 1.0, 11), "angle")

            # output - desired relative angle to match to aim ship at asteroid
            aiming_angle = ctrl.Consequent(np.linspace(-1.0, 1.0, 11), "aiming_angle")

            # creating 3 equally spaced membership functions for the inputs
            distance.automf(3, names=["close", "medium", "far"])
            angle.automf(3, names=["negative", "zero", "positive"])

            # creating 3 triangular membership functions for the output
            aiming_angle["negative"] = skf.trimf(aiming_angle.universe, [-1.0, -1.0, 0.0])
            aiming_angle["zero"] = skf.trimf(aiming_angle.universe, [-1.0, 0.0, 1.0])
            aiming_angle["positive"] = skf.trimf(aiming_angle.universe, [0.0, 1.0, 1.0])

            # creating the rule base for the fuzzy system
            rule1 = ctrl.Rule(distance["close"] & angle["negative"], aiming_angle["negative"])
            rule2 = ctrl.Rule(distance["medium"] & angle["negative"], aiming_angle["negative"])
            rule3 = ctrl.Rule(distance["far"] & angle["negative"], aiming_angle["negative"])
            rule4 = ctrl.Rule(distance["close"] & angle["zero"], aiming_angle["negative"])
            rule5 = ctrl.Rule(distance["medium"] & angle["zero"], aiming_angle["positive"])
            rule6 = ctrl.Rule(distance["far"] & angle["zero"], aiming_angle["positive"])
            rule7 = ctrl.Rule(distance["close"] & angle["positive"], aiming_angle["positive"])
            rule8 = ctrl.Rule(distance["medium"] & angle["positive"], aiming_angle["positive"])
            rule9 = ctrl.Rule(distance["far"] & angle["positive"], aiming_angle["positive"])

            rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]
            # creating a FIS controller from the rules + membership functions
            self.aiming_fis = ctrl.ControlSystem(rules)
            # creating a controller sim to evaluate the FIS
            self.aiming_fis_sim = ctrl.ControlSystemSimulation(self.aiming_fis)
        else:
            # create FIS using GA chromosome
            # input 1 - distance to asteroid
            distance = ctrl.Antecedent(np.linspace(0.0, 1.0, 11), "distance")
            # input 2 - angle to asteroid (relative to ship heading)
            angle = ctrl.Antecedent(np.linspace(-1.0, 1.0, 11), "angle")

            # output - desired relative angle to match to aim ship at asteroid
            aiming_angle = ctrl.Consequent(np.linspace(-1.0, 1.0, 11), "aiming_angle")

            # Create membership functions from chromosome - Note that we're constraining the triangular membership
            # functions to have Ruspini partitioning
            # create distance membership functions from chromosome
            distance["close"] = skf.trimf(distance.universe, [-1.0, -1.0, self.chromosome[0]])
            distance["medium"] = skf.trimf(distance.universe, [-1.0, self.chromosome[0], 1.0])
            distance["far"] = skf.trimf(distance.universe, [self.chromosome[0], 1.0, 1.0])
            # create angle membership functions from chromosome
            angle["negative"] = skf.trimf(angle.universe, [-1.0, -1.0, self.chromosome[1]*2-1])
            angle["zero"] = skf.trimf(angle.universe, [-1.0, self.chromosome[1]*2-1, 1.0])
            angle["positive"] = skf.trimf(angle.universe, [self.chromosome[1]*2-1, 1.0, 1.0])

            # creating 3 triangular membership functions for the output
            aiming_angle["negative"] = skf.trimf(aiming_angle.universe, [-1.0, -1.0, self.chromosome[2]*2-1])
            aiming_angle["zero"] = skf.trimf(aiming_angle.universe, [-1.0, self.chromosome[2]*2-1, 1.0])
            aiming_angle["positive"] = skf.trimf(aiming_angle.universe, [self.chromosome[2]*2-1, 1.0, 1.0])

            input1_mfs = [distance["close"], distance["medium"], distance["far"]]
            input2_mfs = [angle["negative"], angle["zero"], angle["positive"]]

            # create list of output membership functions to index into to create rule antecedents
            output_mfs = [aiming_angle["negative"], aiming_angle["zero"], aiming_angle["positive"]]

            # bin the values associated with rules - this is done so we can use the floats in the chromosome DNA
            # associated with the output membership functions in order to index into our predefined output membership
            # function set - i.e. the "output_mfs" list
            bins = np.array([0.0, 0.33333, 0.66666, 1.0])
            num_mfs1 = len(input1_mfs)
            num_mfs2 = len(input2_mfs)
            num_rules = num_mfs1*num_mfs2
            # grabbing the corresponding DNA values that determine the output mfs from the chromosome
            rules_raw = self.chromosome[3:3+num_rules]
            # binning the values to convert the floats to integer values to be used as indices - a somewhat hacky way
            # using direct integer encodings would be nicer and probably perform better - opportunity for improvement
            ind = np.digitize(rules_raw, bins, right=True)-1
            ind = [int(min(max(idx, 0), 2)) for idx in ind]


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

            for jj in range(num_mfs2):
                for ii in range(num_mfs1):
                    rules.append(ctrl.Rule(input1_mfs[ii] & input2_mfs[jj], rule_consequents_linear[count]))

            # creating a FIS controller from the rules + membership functions
            self.aiming_fis = ctrl.ControlSystem(rules)
            # creating a controller sim to evaluate the FIS
            self.aiming_fis_sim = ctrl.ControlSystemSimulation(self.aiming_fis)

    def create_fuzzy_systems(self):
        self.create_aiming_fis()

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

        # get the nearest asteroid
        nearest_asteroid = self.sa.ownship.nearest_n(1)[0]

        # grab the bearing to the asteroid
        relative_angle = trim_angle(nearest_asteroid.bearing - self.sa.ownship.heading)
        norm_relative_angle = self.sa.norm_angle(relative_angle)

        distance = nearest_asteroid.distance
        norm_ast_distance = self.sa.norm_distance(distance)

        # feed asteroid dist and angle to the FIS
        self.aiming_fis_sim.input["angle"] = norm_relative_angle
        self.aiming_fis_sim.input["distance"] = norm_ast_distance
        # compute fis output
        self.aiming_fis_sim.compute()
        # map normalized output to angle range [-180, 180], note that the output of the fis is determined by the membership functions and they go from -1 to 1
        desired_aim_angle = self.aiming_fis_sim.output["aiming_angle"]*180.0
        aim_angle_difference = trim_angle(desired_aim_angle - relative_angle)

        # this converts the desired aiming angle to a control action to be fed to the ship in terms of turn rate
        # set turn rate to 0
        turn_rate = 0
        # if desired aim angle is outside of +- 1 deg, turn in that direction with max turn rate, otherwise don't turn
        if aim_angle_difference < -0.5:
            turn_rate = ship_state["turn_rate_range"][1]
        elif aim_angle_difference > 0.5:
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
