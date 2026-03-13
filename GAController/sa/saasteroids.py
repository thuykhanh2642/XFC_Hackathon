# -*- coding: utf-8 -*-
# Copyright © 2019-2020 Thales Avionics USA
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import numpy as np


class SAAsteroid:
    """
    Asteroids stores information about all asteroids in the environment that will be available to the AI
    """
    def __init__(self, asteroid_dict, ship, map_size):
        #   position:  The position of the asteroid in the window reference frame
        #   velocity:  The x and y components of velocity
        #   distance:  The distance from the ship owning this Asteroids object to the asteroid
        #   bearing:  The bearing from the ship owning this Asteroids object to the asteroid. (-180, 180), zero is north
        #   position_wrap:  The position of the asteroid warped across the map edge to the closest position to the ship
        #   distance_wrap:  The shortest distance to the asteroid from the ship owning this accounting for map edge wrap
        #   bearing_wrap:  The bearing from the ship owning this to asteroid accounting for map edge wrap
        #   speed:  The speed of the asteroid
        #   direction:  The angle of the asteroid's velocity vector. (-180, 180), zero is north
        #   size:  The size of the asteroid (1 through 4)
        #   tti:  The time-to-impact of the asteroid to collide with the owning ship. Is None if not on collision course

        self.position = asteroid_dict['position']
        self.velocity = asteroid_dict['velocity']
        self.size = asteroid_dict['size']
        self.mass = asteroid_dict['mass']
        self.radius = asteroid_dict['radius']

        self._ship = ship
        self._map_size = map_size

        self._distance = 0
        self._bearing = 0
        self._position_wrap = 0
        self._distance_wrap = 0
        self._bearing_wrap = 0
        self._speed = 0
        self._heading = 0
        self._tti = 0
        self._ship_relative_velocity = 0
        self._ship_relative_velocity_wrap = 0
        self._ship_closure_rate = 0
        self._ship_closure_rate_wrap = 0

    # Attribute getters to reduce unnecessary calculations for asteroids statistics

    @property
    def tti(self):
        if self._tti != 0:
            return self._tti
        else:
            # Set up needed values to make math cleaner
            xs = self._ship.position[0]
            ys = self._ship.position[1]
            xa = self.position_wrap[0]
            ya = self.position_wrap[1]
            vax = self.velocity[0]
            vay = self.velocity[1]
            vsx = self._ship.velocity[0]
            vsy = self._ship.velocity[1]
            so = self.radius + self._ship.radius

            # Add slight offset to 0 velocity differences to handle div by zero warnings
            if(vsx-vax) == 0:
                vsx += 1e-10
            if(vsy-vay) == 0:
                vsy += 1e-10

            tixs = [-(xs - xa + so) / (vsx - vax),
                    -(xs - xa - so) / (vsx - vax)]
            tix_high = max(tixs)
            tix_low = min(tixs)
            tiys = [-(ys - ya + so) / (vsy - vay),
                    -(ys - ya - so) / (vsy - vay)]
            tiy_high = max(tiys)
            tiy_low = min(tiys)

            if tix_high > tiy_low and tiy_high > tix_low:
                tti = min([min(tix_high, tiy_high), max(tix_low, tiy_low)])
                if tti < 0:
                    self._tti = None
                else:
                    self._tti = tti
            else:
                self._tti = None

            return self._tti

    @property
    def distance(self):
        if self._distance:
            return self._distance
        else:
            # Calculate, save, and return
            x_dist = self.position[0] - self._ship.position[0]
            y_dist = self.position[1] - self._ship.position[1]
            self._distance = np.sqrt((x_dist ** 2) + (y_dist ** 2))
            return self._distance

    @property
    def bearing(self):
        if self._bearing:
            return self._bearing
        else:
            # Calculate, save, and return
            x_dist = self.position[0] - self._ship.position[0]
            y_dist = self.position[1] - self._ship.position[1]
            self._bearing = np.degrees(np.arctan2(-x_dist, y_dist))
            return self._bearing

    @property
    def position_wrap(self):
        if self._position_wrap:
            return self._position_wrap
        else:
            # Calculate, save, and return
            new_ast_pos = []
            for dim in range(len(self.position)):
                delta_dim = self.position[dim] - self._ship.position[dim]
                if abs(delta_dim) > self._map_size[dim] / 2:
                    new_ast_pos.append(self.position[dim] - np.sign(delta_dim) * self._map_size[dim])
                else:
                    new_ast_pos.append(self.position[dim])
            self._position_wrap = new_ast_pos
            return self._position_wrap

    @property
    def distance_wrap(self):
        if self._distance_wrap:
            return self._distance_wrap
        else:
            # Calculate, save, and return
            x_dist = self.position_wrap[0] - self._ship.position[0]
            y_dist = self.position_wrap[1] - self._ship.position[1]
            self._distance_wrap = np.sqrt((x_dist ** 2) + (y_dist ** 2))
            return self._distance_wrap

    @property
    def bearing_wrap(self):
        if self._bearing_wrap:
            return self._bearing_wrap
        else:
            # Calculate, save, and return
            x_dist = self.position_wrap[0] - self._ship.position[0]
            y_dist = self.position_wrap[1] - self._ship.position[1]
            self._bearing_wrap = np.degrees(np.arctan2(-x_dist, y_dist))
            return self._bearing_wrap

    @property
    def speed(self):
        if self._speed:
            return self._speed
        else:
            # Calculate, save, and return
            self._speed = np.sqrt((self.velocity[0] ** 2) + (self.velocity[1] ** 2))
            return self._speed

    @property
    def heading(self):
        if self._heading:
            return self._heading
        else:
            # Calculate, save, and return
            self._heading = -np.degrees(np.arctan2(-self.velocity[0], self.velocity[1]))
            return self._heading

    @property
    def ship_relative_velocity(self):
        if self._ship_relative_velocity:
            return self._ship_relative_velocity
        else:
            # Calculate, save, and return
            self._ship_relative_velocity = [
                self.velocity[0] * np.sin(np.radians(self.bearing)) + self.velocity[1] * np.cos(np.radians(self.bearing)),
                self.velocity[0] * np.cos(np.radians(self.bearing)) + self.velocity[1] * np.sin(np.radians(self.bearing))
            ]
            return self._ship_relative_velocity

    @property
    def ship_relative_velocity_wrap(self):
        if self._ship_relative_velocity_wrap:
            return self._ship_relative_velocity_wrap
        else:
            # Calculate, save, and return
            self._ship_relative_velocity_wrap = [
                self.velocity[0] * np.sin(np.radians(self.bearing_wrap)) + self.velocity[1] * np.cos(np.radians(self.bearing_wrap)),
                self.velocity[0] * np.cos(np.radians(self.bearing_wrap)) + self.velocity[1] * np.sin(np.radians(self.bearing_wrap))
            ]
            return self._ship_relative_velocity_wrap

    @property
    def ship_closure_rate(self):
        if self._ship_closure_rate:
            return self._ship_closure_rate
        else:
            # dot product the asteroid velocity onto the position difference... 
            x_dist = self.position[0] - self._ship.position[0]
            y_dist = self.position[1] - self._ship.position[1]

            x_rel_vel = self.ship_relative_velocity[0]
            y_rel_vel = self.ship_relative_velocity[1]

            # Calculate, save, and return
            self._ship_closure_rate = -1 * (x_dist * x_rel_vel + y_dist * y_rel_vel)
            return self._ship_closure_rate
        
    @property
    def ship_closure_rate_wrap(self):
        if self._ship_closure_rate_wrap:
            return self._ship_closure_rate_wrap
        else:
            # dot product the asteroid velocity onto the position difference... 
            x_dist = self.position_wrap[0] - self._ship.position[0]
            y_dist = self.position_wrap[1] - self._ship.position[1]

            x_rel_vel = self.ship_relative_velocity_wrap[0]
            y_rel_vel = self.ship_relative_velocity_wrap[1]

            # Calculate, save, and return
            self._ship_closure_rate_wrap = -1 * (x_dist * x_rel_vel + y_dist * y_rel_vel)
            return self._ship_closure_rate_wrap