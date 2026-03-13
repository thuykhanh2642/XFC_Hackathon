# -*- coding: utf-8 -*-
# Copyright © 2019-2020 Thales Avionics USA
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import numpy as np

from .saship import SAShip, OwnShip


class SA:
    """
    Situational Awareness contains state information for ownship, hostile ship(s), and the environment
    """

    def __init__(self):

        # Initiate SA properties that will be available to the AI
        self.time = 0.0
        self.ownship = OwnShip()
        self.metrics = SAMetrics()
        self.size_ratio_threshold = 2 # Used when determining whether or not we should shoot at a large asteroid

        self.map_diagonal_length = 0        

    def update(self, spaceship, observation):

        # Update our ship
        self.ownship.update(observation, spaceship)

        self.redships = []
        for ship in observation['ships']:
            if ship['id'] != spaceship['id']:
                red = SAShip()
                red.update(observation, ship)
                self.redships.append(red)

        # Update metrics class
        self.metrics.update(observation, self)

        if self.map_diagonal_length == 0:
            self.map_diagonal_length = np.sqrt(observation["map_size"][0]**2 + observation["map_size"][1]**2)

        # Update sim time
        self.time = observation['time']

    ##############################################################################################
    # What follows are a series of helper functions used for normalizing values to be fis inputs #
    ##############################################################################################

    def norm_angle(self, angle):
        # normal range is (-180, 180)
        return angle / 180

    def norm_distance(self, distance):
        # normal range is (0, inf) based on fuzzy-asteroids map size
        return distance / (self.map_diagonal_length / 2)

    def norm_speed_ast(self, speed):
        # normal range is (0, inf) based on momentum effects and scenario
        max = 240 # Use 240 since that is typically the fastest asteroids
        return speed / max

    def norm_speed_ship(self, speed):
        # normal range is (-240, 240) capped by fuzzy-asteroids
        return speed / 240

    def norm_tti(self, tti):
        # normal range is (0, inf) based on speed and distance
        max = 10 # Use 10 since that is typically the longer TTI for asteroids
        return tti / max

    def norm_size(self, size):
        # normal range is (1, 4) capped by fuzzy-asteroids
        max = 4
        return size / (max/2) - 1

    def norm_ast_num(self, num):
        # normal range is (0, inf) based on scenario
        # allow training to set normalization threshold for consistency across scenarios
        max = 100 # Use 100 since that is a typical large number of asteroids
        return num / max

    ##############################################################################################


class SAMetrics:
    """
    SAMetrics stores some general observations about the environment as a whole
    """
    def __init__(self):
        # Initiate properties that will be available to the AI
        self.avg_asteroid_size = 0
        self.stddev_asteroid_size = 0
        self.avg_asteroid_speed = 0
        self.stddev_asteroid_speed = 0

    def update(self, observation, sa):
        if len(observation['asteroids']) != 0:
            self.avg_asteroid_size = np.mean([asteroid.size for asteroid in sa.ownship.asteroids])
            self.stddev_asteroid_size = np.std([asteroid.size for asteroid in sa.ownship.asteroids])
            self.avg_asteroid_speed = np.mean([asteroid.speed for asteroid in sa.ownship.asteroids])
            self.stddev_asteroid_speed = np.std([asteroid.speed for asteroid in sa.ownship.asteroids])
