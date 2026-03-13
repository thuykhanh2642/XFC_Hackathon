# -*- coding: utf-8 -*-
# Copyright © 2019-2020 Thales Avionics USA
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.


class SABullet:
    """
    Bullets stores information about the bullets within the environment
    """
    def __init__(self, bullet_dict, ship):
        # SABullet is currently not implemented

        self.position = bullet_dict['position']
        self.heading = bullet_dict['heading']

        self._ship = ship

        self._distance = 0
        self._bearing = 0
