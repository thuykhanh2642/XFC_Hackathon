# -*- coding: utf-8 -*-
# Copyright © 2019-2020 Thales Avionics USA
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

from .saasteroids import SAAsteroid
from .sabullets import SABullet
from .util.helpers import trim_angle


class SAShip:
    def __init__(self):
        # Initiate the ship with properties that will be available to the AI
        self._time = -1
        self.position = [0.0, 0.0]
        self.lives = 0.0
        self.speed = 0.0
        self.heading = 0.0
        self.turnrate = 0.0
        self.acceleration = 0.0
        self.asteroids = []  # Stores asteroid information relative to this ship
        self.bullets = []  # Stores bullet information relative to this ship

        # Initiate history properties
        self._last_time = -1
        self.last_position = 0
        self.last_heading = 0
        self.last_speed = 0

    def update(self, observation, ship_dict):
        # Populate ship data histories for some parameters
        self._last_time = self._time
        self.last_position = self.position
        self.last_heading = self.heading
        self.last_speed = self.speed

        # Update current ship data
        self._time = observation['time']
        self.position = ship_dict['position']
        self.radius = ship_dict['radius']
        self.mass = ship_dict['mass']
        self.lives = ship_dict['lives_remaining']
        self.speed = ship_dict['speed']
        self.velocity = ship_dict['velocity']
        self.heading = trim_angle(ship_dict['heading']-90)
        self.turnrate = self._infer_turn_rate()
        self.acceleration = self._infer_acceleration()
        self.asteroids = [SAAsteroid(asteroid, self, observation['map_size']) for asteroid in observation['asteroids']]  # Pass self as ship for relative source
        self.bullets = [SABullet(bullet, self) for bullet in observation['bullets']] # Pass self as ship for relative source

    def _infer_turn_rate(self):
        try:
            return (self.heading - self.last_heading) / (self._time - self._last_time)
        except(ZeroDivisionError):
            return self.turnrate

    def _infer_acceleration(self):
        try:
            return (self.speed - self.last_speed) / ( self._time - self._last_time)
        except(ZeroDivisionError):
            return self.turnrate

    ######################################################################################
    # What follows are a series of helper functions used for getting pertinent asteroids #
    ######################################################################################

    def within_radius(self, radius: int):
        """
        Helper function to get the dictionaries for all asteroids within radius of owning ship
        """
        return [asteroid for asteroid in self.asteroids if asteroid.distance < radius]

    def within_radius_wrap(self, radius: int):
        """
        Helper function to get the dictionaries for all asteroids within radius_wrap of owning ship
        """
        return [asteroid for asteroid in self.asteroids if asteroid.distance_wrap < radius]

    def nearest_n(self, n_asteroids: int):
        """
        Helper function to get the dictionaries for N nearest asteroids to owning ship
        """
        return sorted(self.asteroids, key=lambda x: x.distance)[:n_asteroids]

    def nearest_n_wrap(self, n_asteroids: int):
        """
        Helper function to get the dictionaries for N nearest_wrap asteroids to owning ship accommodating wrapping
        """
        return sorted(self.asteroids, key=lambda x: x.distance_wrap)[:n_asteroids]

    def soonest_impact_n(self, n_asteroids: int):
        """
        Helper function to get the soonest impacting N asteroids to the owning ship
        """
        soonest = []
        impacters = [asteroid for asteroid in self.asteroids if asteroid.tti is not None]
        if len(impacters) == 0:
            return soonest
        return sorted(impacters, key=lambda x: x.tti)[:n_asteroids]

    def impact_less_than(self, time: int):
        """
        Helper function to get all asteroids with an impact time less than time
        """
        impacters = [asteroid for asteroid in self.asteroids if asteroid.tti is not None]
        return [asteroid for asteroid in impacters if asteroid.tti < time]

    ######################################################################################




class OwnShip(SAShip):
    """
    Stores additional information about our ship in addition to normal ship data
    """

    def __init__(self):
        super().__init__()

        # Initiate parameters we only know about our ship
        self.last_throttle_cmd = 0.0
        self.last_turn_cmd = 0.0
        self.bullets_remaining = 0
        self.bullet_ast_eq_ratio = 0 # ratio of (bullets_remaining) / (bullets needed to clear field)
        self._target_asteroid = None

    def update(self, observation, ourship):
        super().update(observation, ourship)

        # Update parameters we only know about our ship
        self.bullets_remaining = ourship['bullets_remaining']
        self.bullet_ast_eq_ratio = self._update_bullet_ratio

    def _update_bullet_ratio(self):
        # Calculates how many perfect shots are needed to clear the field at the given state
        shots_needed = 0
        for asteroid in self.asteroids.list:
            if asteroid['size'] == 1:
                shots_needed += 1
            if asteroid['size'] == 2:
                shots_needed += 4
            if asteroid['size'] == 3:
                shots_needed += 13
            if asteroid['size'] == 4:
                shots_needed += 40
        return (self.bullets_remaining - shots_needed) / shots_needed

    # target_asteroid is a property that can be passed asteroid dict relative to any ship and always stores relative
    # to OwnShip by matching characteristics to OwnShip asteroid list
    @property
    def target_asteroid(self):
        return self._target_asteroid

    @target_asteroid.setter
    def target_asteroid(self, in_ast):
        for own_ast in self.asteroids:
            if in_ast.position == own_ast.position:
                self._target_asteroid = own_ast
                break
