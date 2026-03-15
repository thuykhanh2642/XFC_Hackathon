#Author: Kyle Nguyen
#Date: September 2025
#Description: Hybrid fuzzy logic controller — improved v2

import math
from kesslergame.controller import KesslerController
from util import wrap180, triag

SHIP_RADIUS = 20.0
BULLET_SPEED = 800.0
TARGET_LOCK_FRAMES = 10  # ~0.33s at 30Hz


def wrap_delta(d, size):
    d = d % size
    if d > size / 2:
        d -= size
    return d


def toro_dx_dy(sx, sy, ax, ay, map_size):
    w, h = map_size
    return wrap_delta(ax - sx, w), wrap_delta(ay - sy, h)


def toro_dist(sx, sy, ax, ay, map_size):
    dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
    return math.hypot(dx, dy)


def toro_intercept(ship_pos, ast_pos, ast_vel, map_size):
    """Wrap-aware intercept: compute dx/dy on torus, then solve quadratic."""
    sx, sy = ship_pos
    dx, dy = toro_dx_dy(sx, sy, ast_pos[0], ast_pos[1], map_size)
    vx, vy = ast_vel

    a = vx**2 + vy**2 - BULLET_SPEED**2
    b = 2.0 * (dx * vx + dy * vy)
    c = dx**2 + dy**2

    if abs(a) < 1e-6:
        if abs(b) < 1e-6:
            return None
        t = -c / b
        if t <= 0:
            return None
        return sx + dx + vx * t, sy + dy + vy * t, t

    disc = b**2 - 4.0 * a * c
    if disc < 0:
        return None

    sq = math.sqrt(disc)
    t_best = None
    for t_cand in ((-b + sq) / (2.0 * a), (-b - sq) / (2.0 * a)):
        if t_cand > 1e-4 and (t_best is None or t_cand < t_best):
            t_best = t_cand

    if t_best is None:
        return None
    return sx + dx + vx * t_best, sy + dy + vy * t_best, t_best


def compute_tti(ship_pos, ship_vel, ast_pos, ast_vel, ast_radius, map_size):
    """Time-to-impact on toroidal map. Returns float or None if no collision."""
    sx, sy = ship_pos
    dx, dy = toro_dx_dy(sx, sy, ast_pos[0], ast_pos[1], map_size)
    vax, vay = ast_vel
    vsx, vsy = ship_vel
    so = ast_radius + SHIP_RADIUS

    dvx = vsx - vax
    dvy = vsy - vay
    if dvx == 0: dvx = 1e-10
    if dvy == 0: dvy = 1e-10

    tixs = [-(dx + so) / dvx, -(dx - so) / dvx]
    tiys = [-(dy + so) / dvy, -(dy - so) / dvy]
    tix_lo, tix_hi = min(tixs), max(tixs)
    tiy_lo, tiy_hi = min(tiys), max(tiys)

    if tix_hi > tiy_lo and tiy_hi > tix_lo:
        tti = min(min(tix_hi, tiy_hi), max(tix_lo, tiy_lo))
        return tti if tti > 0 else None
    return None


def find_impacters(asteroids, ship_pos, ship_vel, map_size, max_n=5):
    """Return list of (asteroid, tti) sorted by soonest impact."""
    hits = []
    for a in asteroids:
        r = getattr(a, "radius", 0.0)
        vel = getattr(a, "velocity", (0.0, 0.0))
        tti = compute_tti(ship_pos, ship_vel, a.position, vel, r, map_size)
        if tti is not None:
            hits.append((a, tti))
    hits.sort(key=lambda x: x[1])
    return hits[:max_n]


def calculate_threat_priority(asteroid, ship_pos, ship_vel, map_size):
    ax, ay = asteroid.position
    dx, dy = toro_dx_dy(ship_pos[0], ship_pos[1], ax, ay, map_size)
    distance = math.hypot(dx, dy)

    avx, avy = getattr(asteroid, "velocity", (0.0, 0.0))
    closing_speed = ((avx - ship_vel[0]) * dx + (avy - ship_vel[1]) * dy) / max(distance, 1)

    size = getattr(asteroid, "size", 2)
    priority = (1000.0 / max(distance, 1.0)) + max(closing_speed, 0) / 50.0 + (5 - size)
    return priority


def find_closest_threat(asteroids, ship_pos, map_size):
    closest_dist = float('inf')
    closest_asteroid = None

    for asteroid in asteroids:
        ax, ay = asteroid.position
        center_dist = toro_dist(ship_pos[0], ship_pos[1], ax, ay, map_size)
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
        proj = dx * hx + dy * hy
        if 0 < proj < check_range:
            perp = abs(dx * (-hy) + dy * hx)
            if perp < safety + getattr(a, "radius", 0.0):
                return False
    return True


def _weighted_dodge_direction(sx, sy, dx, dy, asteroids, map_size):
    """Pick perpendicular dodge direction, weighting nearby asteroids more."""
    perp1 = (-dy, dx)
    perp2 = (dy, -dx)
    score1, score2 = 0.0, 0.0
    for a in asteroids:
        adx, ady = toro_dx_dy(sx, sy, a.position[0], a.position[1], map_size)
        a_dist = math.hypot(adx, ady)
        if a_dist < 400:
            w = 1.0 / max(a_dist, 10.0)
            score1 += (adx * perp1[0] + ady * perp1[1]) * w
            score2 += (adx * perp2[0] + ady * perp2[1]) * w
    return perp1 if score1 > score2 else perp2


class hybrid_controller(KesslerController):
    def __init__(self):
        self.debug_counter = 0
        self._locked_target_pos = None
        self._lock_timer = 0

    def _pick_fire_target(self, asteroids, ship_pos, ship_vel, map_size):
        """Pick best target with target lock to prevent jitter."""
        if self._lock_timer > 0 and self._locked_target_pos is not None:
            lx, ly = self._locked_target_pos
            for a in asteroids:
                adx, ady = toro_dx_dy(lx, ly, a.position[0], a.position[1], map_size)
                if math.hypot(adx, ady) < 80:
                    self._lock_timer -= 1
                    self._locked_target_pos = a.position
                    return a
            self._locked_target_pos = None
            self._lock_timer = 0

        best = max(asteroids, key=lambda a: calculate_threat_priority(a, ship_pos, ship_vel, map_size))
        self._locked_target_pos = best.position
        self._lock_timer = TARGET_LOCK_FRAMES
        return best

    def actions(self, ship_state, game_state):
        self.debug_counter += 1

        asteroids = getattr(game_state, "asteroids", [])
        if not asteroids:
            return 0.0, 0.0, False, False

        sx, sy = ship_state.position
        heading = ship_state.heading
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))
        map_size = getattr(game_state, "map_size", (1000, 800))

        # --- THREAT ASSESSMENT ---
        closest_asteroid, closest_distance = find_closest_threat(asteroids, (sx, sy), map_size)
        if closest_asteroid is None:
            return 0.0, 0.0, False, False

        ax, ay = closest_asteroid.position
        dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
        avx, avy = getattr(closest_asteroid, "velocity", (0.0, 0.0))
        rel_vel_x, rel_vel_y = avx - svx, avy - svy
        center_dist = math.hypot(dx, dy)
        approaching_speed = (rel_vel_x * dx + rel_vel_y * dy) / max(center_dist, 1)

        very_close = triag(center_dist, 0, 80, 160)
        close = triag(center_dist, 120, 200, 300)
        medium = triag(center_dist, 250, 400, 600)
        fast_approach = triag(approaching_speed, 50, 150, 300)
        slow_approach = triag(approaching_speed, 10, 50, 100)
        danger_level = max(very_close, min(close, max(fast_approach, slow_approach)))

        # Real TTI-based threat detection
        impacters = find_impacters(asteroids, (sx, sy), (svx, svy), map_size)
        soonest_tti = impacters[0][1] if impacters else None
        dodge_target_ast = impacters[0][0] if impacters else closest_asteroid
        n_imminent = sum(1 for _, t in impacters if t < 1.5)

        # TTI overrides fuzzy danger — if something will hit us in < 1.5s, panic
        if soonest_tti is not None and soonest_tti < 1.5:
            danger_level = max(danger_level, 0.8)
        if soonest_tti is not None and soonest_tti < 0.8:
            danger_level = 1.0

        # --- FIRING (always evaluated, independent of movement) ---
        fire_target = self._pick_fire_target(asteroids, (sx, sy), (svx, svy), map_size)
        intercept = toro_intercept((sx, sy), fire_target.position,
                                   getattr(fire_target, "velocity", (0.0, 0.0)), map_size)

        if intercept is not None:
            ix, iy, bullet_t = intercept
            fire_dx, fire_dy = toro_dx_dy(sx, sy, ix, iy, map_size)
            desired_heading = math.degrees(math.atan2(fire_dy, fire_dx))
            fire_heading_err = wrap180(desired_heading - heading)
            fire_dist = math.hypot(fire_dx, fire_dy)
        else:
            fire_dx, fire_dy = toro_dx_dy(sx, sy, fire_target.position[0],
                                           fire_target.position[1], map_size)
            desired_heading = math.degrees(math.atan2(fire_dy, fire_dx))
            fire_heading_err = wrap180(desired_heading - heading)
            fire_dist = math.hypot(fire_dx, fire_dy)
            bullet_t = fire_dist / BULLET_SPEED

        # Tight fire angle — no closing_speed requirement (fires at stationary targets too)
        fire = (
            abs(fire_heading_err) < 6.0
            and fire_dist < 800
            and bullet_t < 1.0
        )

        # Wider tolerance for stationary/slow targets (intercept is exact)
        target_speed = math.hypot(*getattr(fire_target, "velocity", (0.0, 0.0)))
        if target_speed < 30.0 and intercept is not None:
            fire = fire or (abs(fire_heading_err) < 8.0)

        # --- MOVEMENT MODES ---
        # Use the actual impacter for dodge direction, not just closest by distance
        if impacters:
            dodge_ast = dodge_target_ast
            dodge_ax, dodge_ay = dodge_ast.position
            dodge_dx, dodge_dy = toro_dx_dy(sx, sy, dodge_ax, dodge_ay, map_size)
        else:
            dodge_dx, dodge_dy = dx, dy

        if n_imminent >= 2 and soonest_tti is not None and soonest_tti < 1.2:
            # FLEE — multiple threats, thrust away from threat centroid
            cx_t = sum(toro_dx_dy(sx, sy, a.position[0], a.position[1], map_size)[0]
                       for a, _ in impacters[:n_imminent]) / n_imminent
            cy_t = sum(toro_dx_dy(sx, sy, a.position[0], a.position[1], map_size)[1]
                       for a, _ in impacters[:n_imminent]) / n_imminent
            flee_angle = math.degrees(math.atan2(-cy_t, -cx_t))  # away from centroid
            flee_err = wrap180(flee_angle - heading)
            turn_rate = max(-180.0, min(180.0, flee_err * 4.0))
            thrust = 250.0

        elif (soonest_tti is not None and soonest_tti < 0.8) or \
             (closest_distance < 120 and approaching_speed > 30):
            # PANIC — dodge perpendicular to impacter, pick clearer side
            perp = _weighted_dodge_direction(sx, sy, dodge_dx, dodge_dy, asteroids, map_size)
            dodge_angle = math.degrees(math.atan2(perp[1], perp[0]))
            dodge_err = wrap180(dodge_angle - heading)
            turn_rate = max(-180.0, min(180.0, dodge_err * 4.0))
            thrust = 200.0

        elif danger_level > 0.3:
            if rear_clearance((sx, sy), heading, asteroids, map_size):
                # BACKOFF — face threat then reverse
                approach_angle = math.degrees(math.atan2(dodge_dy, dodge_dx))
                aim_err = wrap180(approach_angle - heading)
                turn_rate = max(-180.0, min(180.0, aim_err * 3.0))
                thrust = -150.0
            else:
                # SIDESTEP — rear blocked
                perp = _weighted_dodge_direction(sx, sy, dodge_dx, dodge_dy, asteroids, map_size)
                dodge_angle = math.degrees(math.atan2(perp[1], perp[0]))
                dodge_err = wrap180(dodge_angle - heading)
                turn_rate = max(-180.0, min(180.0, dodge_err * 3.0))
                thrust = 150.0

        elif medium > 0.2:
            # ENGAGEMENT — aim at target
            turn_rate = max(-180.0, min(180.0, fire_heading_err * 3.0))
            thrust = 0.0 if abs(fire_heading_err) < 10.0 else 60.0

        else:
            # CRUISE — approach target
            turn_rate = max(-180.0, min(180.0, fire_heading_err * 2.0))
            thrust = 100.0

        # In engagement/cruise modes, settle turn when nearly aimed and add damping
        in_dodge_mode = (
            (n_imminent >= 2 and soonest_tti is not None and soonest_tti < 1.2)
            or (soonest_tti is not None and soonest_tti < 0.8)
            or (closest_distance < 120 and approaching_speed > 30)
            or danger_level > 0.3
        )

        if not in_dodge_mode:
            if abs(fire_heading_err) < 3.0:
                turn_rate = 0.0
            elif abs(fire_heading_err) > 0.1:
                max_rate = abs(fire_heading_err) * 0.70 * 30.0
                turn_rate = max(-max_rate, min(max_rate, turn_rate))

        drop_mine = False

        if hasattr(ship_state, "thrust_range"):
            lo, hi = ship_state.thrust_range
            thrust = max(lo, min(hi, thrust))
        if hasattr(ship_state, "turn_rate_range"):
            lo, hi = ship_state.turn_rate_range
            turn_rate = max(lo, min(hi, turn_rate))

        return float(thrust), float(turn_rate), fire, drop_mine

    @property
    def name(self) -> str:
        return "HybridFuzzyController V2"