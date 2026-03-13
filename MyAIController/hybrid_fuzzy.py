#Author: Kyle Nguyen
#Date: September 2025
#Description: Hybrid fuzzy logic controller

import math
import os
from kesslergame.controller import KesslerController
from util import wrap180, triag, intercept_point
from data_log import Logger, FEATURES

SHIP_RADIUS = 20.0  # from API docs

def wrap_delta(d, size):
    """Shortest signed delta on a wrapping axis."""
    d = d % size
    if d > size / 2:
        d -= size
    return d

def toro_dx_dy(sx, sy, ax, ay, map_size):
    """Shortest dx, dy from ship to asteroid on toroidal map."""
    w, h = map_size
    return wrap_delta(ax - sx, w), wrap_delta(ay - sy, h)

def toro_dist(sx, sy, ax, ay, map_size):
    dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
    return math.hypot(dx, dy)




# threat priority calculation for targeting
"""A higher score means a more threatening asteroid,
calculate relative speed, factor in size and distance. """
def calculate_threat_priority(asteroid, ship_pos, ship_vel):
    ax, ay = asteroid.position
    dx, dy = ax - ship_pos[0], ay - ship_pos[1]
    #hypot gives sqrt(dx*dx + dy*dy)
    distance = math.hypot(dx, dy)#hypot is very very nice
    
    avx, avy = getattr(asteroid, "velocity", (0.0, 0.0))
    closing_speed = ((avx - ship_vel[0]) * dx + (avy - ship_vel[1]) * dy) / max(distance, 1)
    
    size = getattr(asteroid, "size", 2)
    
    """(1000 / distance) → closer asteroids = higher priority.
(max(closing_speed, 0) / 50) → if the asteroid is rushing toward you, add danger. If moving away, ignore it (max(...,0)).
(5 - size) -> smaller asteroids add to priority (maybe theyare harder to hit or dodge).

"""


    priority = (1000.0 / distance) + max(closing_speed, 0) / 50.0 + (5 - size)#give priority to smaller asteroids
    return priority

def find_closest_threat(asteroids, ship_pos, map_size):
    closest_dist = float('inf')
    closest_asteroid = None
    
    for asteroid in asteroids:
        ax, ay = asteroid.position
        center_dist = toro_dist(ship_pos[0], ship_pos[1], ax, ay, map_size)
        # Collision gap = center distance minus BOTH radii
        gap = center_dist - getattr(asteroid, "radius", 0.0) - SHIP_RADIUS
        if gap < closest_dist:
            closest_dist = gap
            closest_asteroid = asteroid
    
    return closest_asteroid, max(closest_dist, 1.0)


def rear_clearance(ship_pos, heading_deg, asteroids, map_size, check_range=200.0, safety=40.0):
    hx = math.cos(math.radians(heading_deg + 180))
    hy = math.sin(math.radians(heading_deg + 180))
    sx, sy = ship_pos
    for a in asteroids:
        ax, ay = a.position
        dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
        
        proj = dx*hx + dy*hy
        if 0 < proj < check_range:
            perp = abs(dx*(-hy) + dy*hx)
            if perp < safety + getattr(a, "radius", 0.0):
                return False

    return True



class hybrid_controller(KesslerController):
    def __init__(self):
        self.debug_counter = 0
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        self.maneuver_logger = Logger(os.path.join(data_dir, "maneuver.csv"), FEATURES, ["thrust", "turn_rate"])
        self.combat_logger   = Logger(os.path.join(data_dir, "combat.csv"), FEATURES, ["fire", "drop_mine"])
        self.enable_logging = True  # set False when using as DAgger expert

    def context(self, ship_state, game_state):
        sx, sy = ship_state.position
        heading = ship_state.heading
        asteroids = getattr(game_state, "asteroids", [])
        map_size = getattr(game_state, "map_size", (1000, 800))
        if not asteroids:
            return {
                "dist": 1000.0,
                "ttc": 100.0,
                "heading_err": 0.0,
                "approach_speed": 0.0,
                "ammo": getattr(ship_state, "ammo", 0),
                "mines": getattr(ship_state, "mines", 0),
                "threat_density": 0.0,
                "threat_angle": 0.0
            }

        closest, dist = find_closest_threat(asteroids, (sx, sy), map_size)
        ax, ay = closest.position
        dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
        avx, avy = getattr(closest, "velocity", (0.0, 0.0))
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))
        rel_vx, rel_vy = avx - svx, avy - svy
        raw_dist = math.hypot(dx, dy)
        approach_speed = (rel_vx * dx + rel_vy * dy) / max(raw_dist, 1)

        ttc = dist / max(abs(approach_speed), 1e-6)
        heading_err = wrap180(math.degrees(math.atan2(dy, dx)) - heading)
        density = len(asteroids) / 10.0

        return {
            "dist": dist,
            "ttc": ttc,
            "heading_err": heading_err,
            "approach_speed": approach_speed,
            "ammo": getattr(ship_state, "ammo", 0),
            "mines": getattr(ship_state, "mines", 0),
            "threat_density": density,
            "threat_angle": math.degrees(math.atan2(dy, dx))
        }

    def actions(self, ship_state, game_state):
        self.debug_counter += 1
        """for b in game_state.bullets:
            print("Bullet speed:", math.hypot(b.vx, b.vy))""" #checking bullet speed
        ctx = self.context(ship_state, game_state)

        asteroids = getattr(game_state, "asteroids", [])
        if not asteroids:
            return 0.0, 0.0, False, False 

        sx, sy = ship_state.position
        heading = ship_state.heading
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))
        map_size = getattr(game_state, "map_size", (1000, 800))

        closest_asteroid, closest_distance = find_closest_threat(asteroids, (sx, sy), map_size)
        if closest_asteroid is None:
            return 0.0, 0.0, False, False # no threats, safe

        ax, ay = closest_asteroid.position
        dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
        avx, avy = getattr(closest_asteroid, "velocity", (0.0, 0.0))

        rel_vel_x, rel_vel_y = avx - svx, avy - svy
        center_dist = math.hypot(dx, dy)  # center-to-center (wrapping-aware)
        approaching_speed = (rel_vel_x * dx + rel_vel_y * dy) / max(center_dist, 1)

        # Membership functions use CENTER distance (what they were tuned for)
        very_close = triag(center_dist, 0, 80, 160)
        close = triag(center_dist, 120, 200, 300)
        medium = triag(center_dist, 250, 400, 600)
        far = triag(center_dist, 500, 700, 1000)

        fast_approach = triag(approaching_speed, 50, 150, 300)
        slow_approach = triag(approaching_speed, 10, 50, 100)
        moving_away = triag(approaching_speed, -200, -50, 10)


        #Danger is high if very close, or close and approaching fast
        danger_level = max(very_close, min(close, max(fast_approach, slow_approach)))
        best_asteroid = max(asteroids, key=lambda a: calculate_threat_priority(a, (sx,sy), (svx,svy)))

        if self.debug_counter % 30 == 0:
            print(f"gap={closest_distance:.0f}, center={center_dist:.0f}, approach={approaching_speed:.0f}, danger={danger_level:.2f}")

        if closest_distance < 120 and approaching_speed > 30:
            #panic mode
            #dx, dy: vector from ship to asteroid
            #moving sideways depends on the amount of asteroids
            
            perp1 = (-dy, dx)# perpendicular vectors left and right
            perp2 = (dy, -dx)
            
            #all asteroid positions — use wrapping for directions
            vectors_to_asteroids = [toro_dx_dy(sx, sy, a.position[0], a.position[1], map_size) for a in asteroids]

            dot_products_perp1 = [(vx * perp1[0] + vy * perp1[1]) for (vx, vy) in vectors_to_asteroids]
            dot_products_perp2 = [(vx * perp2[0] + vy * perp2[1]) for (vx, vy) in vectors_to_asteroids]


            #counting how many asteroids are on each side
            score1 = sum(dot_products_perp1)
            score2 = sum(dot_products_perp2)
            perp = perp1 if score1 > score2 else perp2

            dodge_angle = math.degrees(math.atan2(perp[1], perp[0]))
            dodge_err = wrap180(dodge_angle - heading)
            turn_rate = max(-180.0, min(180.0, dodge_err * 4.0))
            thrust = 150.0

            if self.debug_counter % 30 == 0:
                print("MODE: Panic Mode")

        elif danger_level > 0.3:
            if rear_clearance((sx, sy), heading, asteroids, map_size):# clear behind, back off
                approach_angle = math.degrees(math.atan2(dy, dx)) #atan2 gives angle from x-axis to point (x,y)
                aim_err = wrap180(approach_angle - heading)
                
                #If aim_err = 10°, then aim_err * 3.0 = 30° turn.
                #If aim_err = 50°, then aim_err * 3.0 = 150° turn.
                #max of -180-180 turn rate
                turn_rate = max(-180.0, min(180.0, aim_err * 3.0)) 
                
                thrust = -120.0
                if self.debug_counter % 30 == 0:
                    print("MODE: Backoff (clear rear)")
            else: # blocked behind, dodge sideways
                perp1 = (-dy, dx)
                perp2 = (dy, -dx)
                vectors_to_asteroids = [toro_dx_dy(sx, sy, a.position[0], a.position[1], map_size) for a in asteroids]
                #Score how many asteroids are on each side
                """NOTE: Not very good, very far asteroids still count, should only count within a certain range"""
                score1 = sum(vx * perp1[0] + vy * perp1[1] for (vx, vy) in vectors_to_asteroids)
                score2 = sum(vx * perp2[0] + vy * perp2[1] for (vx, vy) in vectors_to_asteroids)
                perp = perp1 if score1 > score2 else perp2
                dodge_angle = math.degrees(math.atan2(perp[1], perp[0]))
                dodge_err = wrap180(dodge_angle - heading)
                turn_rate = max(-180.0, min(180.0, dodge_err * 3.0))
                thrust = 120.0  # push sideways
                if self.debug_counter % 30 == 0:
                    print("MODE: BLOCKED REAR -> SIDE-STEP")

        elif medium > 0.2:
            # pew pew time
            thrust = 80.0
            if best_asteroid:
                
                #bullet_speed = 800.0
                ix, iy = intercept_point((sx, sy), (svx, svy),best_asteroid.position, getattr(best_asteroid, "velocity", (0.0, 0.0)))    
                dx_i, dy_i = toro_dx_dy(sx, sy, ix, iy, map_size)

                desired_heading = math.degrees(math.atan2(dy_i, dx_i))
                heading_err = wrap180(desired_heading - heading)
                turn_rate = max(-180.0, min(180.0, heading_err * 3.0))
            else:
                turn_rate = 0.0
            if self.debug_counter % 30 == 0:
                print("MODE: ENGAGEMENT (pew pew)")

        else:
            #cruising, 
            thrust = 120.0
            approach_angle = math.degrees(math.atan2(dy, dx)) # angle to the closest asteroid
            approach_err = wrap180(approach_angle - heading) # how far off is it from our heading
            turn_rate = max(-180.0, min(180.0, approach_err * 2.0)) #clamping with gain factor of 2.0, the farther off, the harder we turn
            if self.debug_counter % 30 == 0:
                print("MODE: FAR APPROACH (cruisin')")

        if closest_distance > 100:
            ix, iy = intercept_point((sx, sy), (svx, svy),
                                    best_asteroid.position, best_asteroid.velocity)
            dx_i, dy_i = toro_dx_dy(sx, sy, ix, iy, map_size)
            desired_heading = math.degrees(math.atan2(dy_i, dx_i))
            heading_err = wrap180(desired_heading - heading)
            target_distance = math.hypot(dx_i, dy_i)

            #rv check
            bx, by = best_asteroid.position
            bvx, bvy = getattr(best_asteroid, "velocity", (0.0, 0.0))
            rel_vx, rel_vy = bvx - svx, bvy - svy
            rel_dx, rel_dy = toro_dx_dy(sx, sy, bx, by, map_size)
            dist_now = math.hypot(rel_dx, rel_dy) or 1.0
            closing_speed = (rel_vx * rel_dx + rel_vy * rel_dy) / dist_now

            fire =(
                abs(heading_err) < 20 and
                target_distance < 700 and
                closing_speed > 0)
        else:
            fire = False


        asteroid_size = getattr(closest_asteroid, "size", 2)
        drop_mine = False  # disabled — mines kill us more than asteroids

        # clamp into ship’s limits so it doesn’t freak out
        if hasattr(ship_state, "thrust_range"):
            lo, hi = ship_state.thrust_range
            thrust = max(lo, min(hi, thrust))
        if hasattr(ship_state, "turn_rate_range"):
            lo, hi = ship_state.turn_rate_range
            turn_rate = max(lo, min(hi, turn_rate))


        # Log data
        if self.enable_logging:
            try:
                thrust_c = max(-1.0, min(1.0, float(thrust) / 150.0))  # normalize –150 to 150  –1 to 1
                turn_rate_c = max(-1.0, min(1.0, float(turn_rate) / 180.0))  # normalize –180 to 180  –1 to 1  
    
                fire_c      = 1.0 if fire else 0.0
                mine_c      = 1.0 if drop_mine else 0.0
    
                self.maneuver_logger.log(ctx, (thrust_c, turn_rate_c))
                self.combat_logger.log(ctx, (fire_c, mine_c))
            except Exception as e:
                if self.debug_counter % 120 == 0:
                    print(f"[Logger warning] {e}")

        return float(thrust), float(turn_rate), fire, drop_mine
    @property
    def name(self) -> str:
        """
        Simple property used for naming controllers such that it can be displayed in the graphics engine

        Returns:
            str: name of this controller
        """
        return "HybridFuzzyController"