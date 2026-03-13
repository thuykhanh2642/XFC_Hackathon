# -*- coding: utf-8 -*-
# Copyright © 2019-2020 Thales Avionics USA
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

def trim_angle(angle):
    """
    Helper function that trims an angle to be between -180 and +180
    """

    if angle > 180.0:
        angle -= 360.0
    elif angle < -180:
        angle += 360.0

    return angle


def turn_rate_calc(sa):
    """
    From an sa which already has desired heading set, determine what the turn rate shall be that will reach that goal
    """

    # Assumed from fuzzy-asteroids:
    # Frequency = 30 HZ
    # max_turn_rate = 180 deg/s
    # 180 / 30 = 6
    max_turn_rate_timestep = 6
    desired_diff = trim_angle(sa.desired_heading - sa.ownship.heading)

    if desired_diff > max_turn_rate_timestep:
        turn_rate = 180.0
    elif desired_diff < -max_turn_rate_timestep:
        turn_rate = -180.0
    else:
        turn_rate = desired_diff * 30.0

    return turn_rate
