import math
from collections import deque
from kesslergame.controller import KesslerController
from util import *

# the ship is this big (pixels? units? idk, it's 20)
SHIP_RADIUS = 20.0
# bullets go really fast
BULLET_SPEED = 800.0

# shoot every frame, no chill
FIRE_COOLDOWN_FRAMES = 0
# one good frame is enough, we're not that picky
AIM_STABLE_REQUIRED = 1
# stick with a target for a while before getting distracted
GUN_LOCK_FRAMES = 12

MAX_BULLET_TIME = 2.0
# smallest allowed aim window before we just give up
MIN_ANGLE_LIMIT = 1.8
# if we're spinning faster than this we probably shouldn't shoot
STABLE_TURN_RATE = 42.0
# make the hitbox slightly bigger so we actually hit things
AIM_SAFETY = 1.05

# how good does a shot need to be? (0 = terrible, 1 = perfect)
FIRE_Q_STANDARD  = 0.35
FIRE_Q_PRESSURE  = 0.22
FIRE_Q_ENGAGE    = 0.45
FIRE_Q_GOOD_SHOT = 0.55

# run away parameters
EVASION_RANGE = 350.0
EVASION_FORCE_GAIN = 2.0
VELOCITY_FORCE_GAIN = 0.5
MAX_EVASION_SPEED = 220.0
MIN_EVASION_DIST = 180.0
ESCAPE_SMOOTHING = 0.22
TURN_FIRST_THRESH = 24.0
DEBUG_PRINT_EVERY = 6

# cleanup mode when theres one tiny rock left -> be more aggressive
CLEANUP_DANGER_MAX = 0.22
CLEANUP_HOLD_DIST = 520.0
CLEANUP_FIRE_DIST = 900.0
CLEANUP_EXTRA_ANGLE = 3.0


# quadratic formula for where will the bullet meet the rock
# returns (hit_x, hit_y, time) or None if impossible
def _solve_intercept(dx, dy, vx, vy):
    a = vx**2 + vy**2 - BULLET_SPEED**2
    b = 2.0 * (dx * vx + dy * vy)
    c = dx**2 + dy**2

    if abs(a) < 1e-6:
        if abs(b) < 1e-6:
            return None
        t = -c / b
        if t <= 0:
            return None
        return dx + vx * t, dy + vy * t, t

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
    return dx + vx * t_best, dy + vy * t_best, t_best


# accounting for the map wrapping around
def direct_intercept(ship_pos, ship_vel, ast_pos, ast_vel, map_size):
    sx, sy = ship_pos
    svx, svy = ship_vel
    avx, avy = ast_vel
    rvx = avx - svx
    rvy = avy - svy
    w, h = map_size

    best = None
    best_t = float('inf')
    for ox in (0, w, -w):
        for oy in (0, h, -h):
            dx = (ast_pos[0] + ox) - sx
            dy = (ast_pos[1] + oy) - sy
            if abs(dx) > w * 0.75 or abs(dy) > h * 0.75:
                continue
            result = _solve_intercept(dx, dy, rvx, rvy)
            if result is not None and result[2] < best_t:
                best_t = result[2]
                best = result
    return best


# one axis of the AABB collision interval check
# returns (lo, hi) time interval or None if they never overlap on this axis
def _axis_interval(offset, half_size, rel_vel):
    if abs(rel_vel) > 1e-9:
        t1 = -(offset + half_size) / rel_vel
        t2 = -(offset - half_size) / rel_vel
        return (min(t1, t2), max(t1, t2))
    else:
        if abs(offset) >= half_size:
            return None
        return (-1e18, 1e18)


#time to collision
# checks all 9 virtual copies because the map wraps and rocks sneak up from behind
def compute_tti(ship_pos, ship_vel, ast_pos, ast_vel, ast_radius, map_size):
    sx, sy = ship_pos
    vsx, vsy = ship_vel
    vax, vay = ast_vel
    so = ast_radius + SHIP_RADIUS
    dvx = vsx - vax
    dvy = vsy - vay
    w, h = map_size

    best_tti = None
    for ox in (0, w, -w):
        for oy in (0, h, -h):
            dx = (ast_pos[0] + ox) - sx
            dy = (ast_pos[1] + oy) - sy

            # skip images that are way too far away to matter
            if abs(dx) > w * 0.75 or abs(dy) > h * 0.75:
                continue

            xi = _axis_interval(dx, so, dvx)
            if xi is None:
                continue
            yi = _axis_interval(dy, so, dvy)
            if yi is None:
                continue

            enter = max(xi[0], yi[0])
            leave = min(xi[1], yi[1])
            if leave <= enter:
                continue

            tti = enter if enter > 0 else leave
            if tti > 0 and (best_tti is None or tti < best_tti):
                best_tti = tti

    return best_tti




# find the rocks that are actually going to hit us, sorted by soonest first
def find_impacters(asteroids, ship_pos, ship_vel, map_size, max_n=5):
    hits = []
    for a in asteroids:
        r = getattr(a, "radius", 0.0)
        vel = getattr(a, "velocity", (0.0, 0.0))
        tti = compute_tti(ship_pos, ship_vel, a.position, vel, r, map_size)
        if tti is not None:
            hits.append((a, tti))
    hits.sort(key=lambda x: x[1])
    return hits[:max_n]


# fuzzy membership functions, cuz why not (triangular for now)
def _falling(x, lo, hi):
    if x <= lo: return 1.0
    if x >= hi: return 0.0
    return (hi - x) / (hi - lo)

def _rising(x, lo, hi):
    if x <= lo: return 0.0
    if x >= hi: return 1.0
    return (x - lo) / (hi - lo)

def _trap(x, a, b, c, d):
    if x <= a or x >= d: return 0.0
    if b <= x <= c:      return 1.0
    if x < b:            return (x - a) / (b - a)
    return (d - x) / (d - c)


# how good is this shot? all four things need to be ok or the whole score tanks
def fuzzy_shot_quality(heading_err_abs, angle_limit, bullet_t, fire_dist, stable):
    aim  = _falling(heading_err_abs, angle_limit * 0.50, angle_limit * 1.10)
    time = _falling(bullet_t,  0.60, MAX_BULLET_TIME)
    dist = _falling(fire_dist, 350.0, 900.0)
    stab = 1.0 if stable else 0.70
    return aim * time * dist * stab


# don't go full speed if we're trying to turn at the same time
def fuzzy_thrust_scale(turn_err_abs):
    return max(0.15, _falling(turn_err_abs, 10.0, 35.0))


#returns three overlapping weights
# so there's no jarring snap when danger changes
def fuzzy_danger_blend(danger_level):
    engage_w     = _falling(danger_level, 0.30, 0.50)
    reposition_w = _trap(danger_level,    0.30, 0.42, 0.62, 0.75)
    evade_w      = _rising(danger_level,  0.58, 0.75)
    return engage_w, reposition_w, evade_w


# how urgent is this rock, roughly
def calculate_threat_priority(asteroid, ship_pos, ship_vel, map_size):
    ax, ay = asteroid.position
    dx, dy = toro_dx_dy(ship_pos[0], ship_pos[1], ax, ay, map_size)
    distance = math.hypot(dx, dy)
    avx, avy = getattr(asteroid, "velocity", (0.0, 0.0))
    closing_speed = ((avx - ship_vel[0]) * dx + (avy - ship_vel[1]) * dy) / max(distance, 1)
    size = getattr(asteroid, "size", 2)
    size_bonus = {4: 5.0, 3: 3.0, 2: 1.0, 1: -1.0}.get(size, 0.0)
    return (900.0 / max(distance, 1.0)) + max(closing_speed, 0.0) / 60.0 + size_bonus


# how many degrees off can we be and still hit the rock
# small rocks get a tiny bonus radius because otherwise we never hit them
def max_aim_error(asteroid, distance):
    radius = float(getattr(asteroid, "radius", 10.0))
    size   = int(getattr(asteroid, "size", 2))
    effective_radius = radius + {1: 5.0, 2: 3.0, 3: 1.5}.get(size, 0.0)
    if distance < effective_radius:
        return 180.0
    safe_ratio = min(effective_radius / distance, 1.0)
    return math.degrees(math.asin(safe_ratio)) * AIM_SAFETY


# score a potential gun target: big close rocks score high, tiny far ones score low
def gun_target_score(asteroid, ship_pos, ship_vel, heading_deg, map_size):
    intercept = direct_intercept(ship_pos, ship_vel,
                                  asteroid.position,
                                  getattr(asteroid, "velocity", (0.0, 0.0)),
                                  map_size)
    if intercept is None:
        return -1e9

    dx, dy, t = intercept
    aim_heading = math.degrees(math.atan2(dy, dx))
    aim_err = abs(wrap180(aim_heading - heading_deg))
    dist = math.hypot(dx, dy)
    size   = int(getattr(asteroid, "size", 2))
    radius = float(getattr(asteroid, "radius", 10.0))
    feasibility = (radius + 6.0) / max(dist, 1.0)
    size_bonus = {4: 10.0, 3: 6.0, 2: 2.5, 1: -2.0}.get(size, 0.0)
    return size_bonus + feasibility * 1400.0 - aim_err * 0.95 - t * 4.0 - dist * 0.003


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


def rear_clearance(ship_pos, heading_deg, asteroids, map_size,
                   check_range=200.0, safety=40.0):
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


def forward_clearance(ship_pos, heading_deg, asteroids, map_size,
                      check_range=220.0, safety=55.0):
    hx = math.cos(math.radians(heading_deg))
    hy = math.sin(math.radians(heading_deg))
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


# how fast are we moving toward a thing (negative = moving away, that's fine)
def projected_speed_toward(ship_vel, dir_x, dir_y):
    mag = math.hypot(dir_x, dir_y)
    if mag < 1e-6:
        return 0.0
    ux, uy = dir_x / mag, dir_y / mag
    return ship_vel[0] * ux + ship_vel[1] * uy


# every nearby rock pushes us away, fast rocks push harder
def potential_field_force(ship_pos, ship_vel, asteroids, map_size,
                           range_limit=EVASION_RANGE):
    fx, fy = 0.0, 0.0
    for a in asteroids:
        dx, dy = toro_dx_dy(ship_pos[0], ship_pos[1],
                             a.position[0], a.position[1], map_size)
        dist = math.hypot(dx, dy)
        if dist < range_limit:
            dist_factor = (range_limit - dist) / (dist + SHIP_RADIUS)
            avx, avy = getattr(a, "velocity", (0.0, 0.0))
            rel_vx = avx - ship_vel[0]
            rel_vy = avy - ship_vel[1]
            closing = (rel_vx * dx + rel_vy * dy) / max(dist, 1.0)
            vel_factor = max(0.0, closing) * VELOCITY_FORCE_GAIN
            weight = EVASION_FORCE_GAIN * dist_factor + vel_factor
            fx -= dx * weight
            fy -= dy * weight
    return fx, fy


class hybrid_controller(KesslerController):
    def __init__(self):
        self.debug_counter = 0
        self._gun_lock_pos = None
        self._gun_lock_timer = 0
        self._fire_cooldown = 0
        self._good_aim_frames = 0
        self._last_turn_rate = 0.0
        self._smoothed_escape_deg = None
        self._mine_escape_timer = 0
        
        
    # pick a rock to shoot at and stick with it for a few frames
    def _pick_gun_target(self, asteroids, ship_pos, ship_vel,
                          heading, map_size, closest_asteroid=None):
        if self._gun_lock_timer > 0 and self._gun_lock_pos is not None:
            lx, ly = self._gun_lock_pos
            for a in asteroids:
                adx, ady = toro_dx_dy(lx, ly, a.position[0], a.position[1], map_size)
                if math.hypot(adx, ady) < 80:
                    self._gun_lock_timer -= 1
                    self._gun_lock_pos = a.position
                    return a
            self._gun_lock_pos = None
            self._gun_lock_timer = 0

        best = None
        best_score = -1e9
        for a in asteroids:
            score = gun_target_score(a, ship_pos, ship_vel, heading, map_size)
            if score > best_score:
                best_score = score
                best = a
        if best is None and closest_asteroid is not None:
            best = closest_asteroid

        if best is not None:
            self._gun_lock_pos = best.position
            self._gun_lock_timer = GUN_LOCK_FRAMES
        return best

    def actions(self, ship_state, game_state):
        self.debug_counter += 1

        if self._mine_escape_timer > 0:
            self._mine_escape_timer -= 1
        if self._fire_cooldown > 0:
            self._fire_cooldown -= 1

        asteroids = getattr(game_state, "asteroids", [])
        if not asteroids:
            self._good_aim_frames = 0
            return 0.0, 0.0, False, False

        sx, sy   = ship_state.position
        heading  = ship_state.heading
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))
        map_size = getattr(game_state, "map_size", (1000, 800))

        # find the scariest rock
        closest_asteroid, closest_distance = find_closest_threat(
            asteroids, (sx, sy), map_size)
        if closest_asteroid is None:
            self._good_aim_frames = 0
            return 0.0, 0.0, False, False

        ax, ay = closest_asteroid.position
        dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
        avx, avy = getattr(closest_asteroid, "velocity", (0.0, 0.0))
        rel_vel_x, rel_vel_y = avx - svx, avy - svy
        center_dist = math.hypot(dx, dy)
        approaching_speed = (rel_vel_x * dx + rel_vel_y * dy) / max(center_dist, 1)

        # fuzzy danger level: 0 = chilling, 1 = screaming
        very_close    = triag(center_dist, 0, 80, 160)
        close         = triag(center_dist, 120, 200, 300)
        medium        = triag(center_dist, 250, 400, 600)
        fast_approach = triag(approaching_speed, 50, 150, 300)
        slow_approach = triag(approaching_speed, 10, 50, 100)
        danger_level  = max(very_close, min(close, max(fast_approach, slow_approach)))

        # find rocks on a collision course (including wrap-around sneakers from behind)
        impacters   = find_impacters(asteroids, (sx, sy), (svx, svy), map_size)
        soonest_tti = impacters[0][1] if impacters else None
        n_imminent  = sum(1 for _, t in impacters if t < 1.5)

        if soonest_tti is not None and soonest_tti < 1.5:
            danger_level = max(danger_level, 0.8)
        if soonest_tti is not None and soonest_tti < 0.8:
            danger_level = 1.0

        engage_w, reposition_w, evade_w = fuzzy_danger_blend(danger_level)

        # pick a target and figure out where to aim
        fire_target = self._pick_gun_target(
            asteroids, (sx, sy), (svx, svy), heading, map_size, closest_asteroid)
        target_vel  = getattr(fire_target, "velocity", (0.0, 0.0))
        target_size = getattr(fire_target, "size", 2)
        n_ast       = len(asteroids)

        # cleanup mode: one tiny rock left and we're not in danger
        cleanup_mode = (
            n_ast == 1
            and danger_level <= CLEANUP_DANGER_MAX
            and soonest_tti is None
            and target_size <= 1
        )

        intercept = direct_intercept(
            (sx, sy), (svx, svy), fire_target.position, target_vel, map_size)
        if intercept is not None:
            fire_dx, fire_dy, bullet_t = intercept
        else:
            fire_dx, fire_dy = toro_dx_dy(
                sx, sy, fire_target.position[0], fire_target.position[1], map_size)
            fire_dist_fallback = math.hypot(fire_dx, fire_dy)
            bullet_t = (fire_dist_fallback / BULLET_SPEED
                        if fire_dist_fallback > 1e-6
                        else MAX_BULLET_TIME + 1.0)

        desired_heading  = math.degrees(math.atan2(fire_dy, fire_dx))
        fire_heading_err = wrap180(desired_heading - heading)
        fire_dist        = math.hypot(fire_dx, fire_dy)

        angle_limit = max_aim_error(fire_target, fire_dist)
        angle_limit = max(angle_limit, MIN_ANGLE_LIMIT)
        if cleanup_mode and target_size > 1:
            angle_limit = max(angle_limit + CLEANUP_EXTRA_ANGLE, 5.0)

        turn_rate_mag       = abs(self._last_turn_rate)
        stable_turn_limit   = STABLE_TURN_RATE * (1.35 if cleanup_mode else 1.0)
        stable              = turn_rate_mag < stable_turn_limit
        aim_frames_required = 0 if cleanup_mode else AIM_STABLE_REQUIRED

        shot_q    = fuzzy_shot_quality(
            abs(fire_heading_err), angle_limit, bullet_t, fire_dist, stable)
        good_shot = shot_q >= FIRE_Q_GOOD_SHOT

        aim_build_limit = angle_limit * 1.2
        if (intercept is not None
                and bullet_t < MAX_BULLET_TIME
                and abs(fire_heading_err) < aim_build_limit
                and stable):
            self._good_aim_frames += 1
        else:
            self._good_aim_frames = 0

        # normal fire: lined up, stable, good quality
        fire = (
            intercept is not None
            and self._fire_cooldown == 0
            and self._good_aim_frames >= aim_frames_required
            and abs(fire_heading_err) < angle_limit
            and bullet_t < MAX_BULLET_TIME
            and fire_dist < (CLEANUP_FIRE_DIST if cleanup_mode else 950)
            and shot_q >= FIRE_Q_STANDARD
        )

        # panic fire: we're in danger and the shot is "good enough"
        if (not fire
                and intercept is not None
                and self._fire_cooldown == 0
                and danger_level > 0.35
                and bullet_t < min(MAX_BULLET_TIME, 1.35)
                and abs(fire_heading_err) < max(angle_limit * 1.35, angle_limit + 2.0)
                and fire_dist < 720
                and shot_q >= FIRE_Q_PRESSURE):
            fire = True

        # cleanup fire: one last rock, be precise (size=1 gets no extra slack)
        if cleanup_mode and intercept is not None and self._fire_cooldown == 0:
            cleanup_fire_limit = angle_limit if target_size <= 1 else max(angle_limit, 5.5)
            if (bullet_t < min(MAX_BULLET_TIME, 0.95)
                    and abs(fire_heading_err) < cleanup_fire_limit):
                fire = True

        # point blank: it's RIGHT THERE, just shoot
        if (danger_level > 0.7
                and fire_dist < 200
                and abs(fire_heading_err) < (angle_limit + 3.0)):
            fire = fire or (intercept is not None
                            and bullet_t < MAX_BULLET_TIME
                            and self._fire_cooldown == 0)

        # don't waste bullets on tiny far rocks if we can't aim properly
        small_target_guard = False
        if target_size <= 1 and fire_dist > 260 and abs(fire_heading_err) > 4.0 and not cleanup_mode:
            fire = False
            small_target_guard = True

        if fire:
            self._fire_cooldown = FIRE_COOLDOWN_FRAMES
            self._good_aim_frames = 0

        # potential field push: every rock pushes us away
        fx, fy = potential_field_force(
            (sx, sy), (svx, svy), asteroids, map_size)

        if abs(fx) < 1e-6 and abs(fy) < 1e-6:
            raw_escape_deg = desired_heading
        else:
            raw_escape_deg = math.degrees(math.atan2(fy, fx))

        # smooth the escape direction so we don't twitch like a maniac
        if self._smoothed_escape_deg is None:
            self._smoothed_escape_deg = raw_escape_deg
        else:
            diff = wrap180(raw_escape_deg - self._smoothed_escape_deg)
            self._smoothed_escape_deg = wrap180(
                self._smoothed_escape_deg + ESCAPE_SMOOTHING * diff)

        min_dist_to_any = min(
            (toro_dist(sx, sy, a.position[0], a.position[1], map_size)
             - getattr(a, "radius", 0) - SHIP_RADIUS)
            for a in asteroids
        )
        emergency = min_dist_to_any < MIN_EVASION_DIST

        threat_ahead        = abs(wrap180(math.degrees(math.atan2(dy, dx)) - heading)) < 75.0
        closing_into_threat = projected_speed_toward((svx, svy), dx, dy)
        rear_is_clear       = rear_clearance((sx, sy), heading, asteroids, map_size)
        front_is_clear      = forward_clearance((sx, sy), heading, asteroids, map_size)

        # movement modes, ordered scariest to most relaxed
        movement_mode = "normal"

        if cleanup_mode:
            movement_mode = "cleanup"
            target_angle  = desired_heading
            if fire_dist > CLEANUP_HOLD_DIST:
                base_thrust = 55.0 if front_is_clear else 0.0
            elif fire_dist > 180.0 and not good_shot:
                base_thrust = 28.0 if front_is_clear else 0.0
            else:
                base_thrust = 0.0

        elif n_imminent >= 2 and soonest_tti is not None and soonest_tti < 1.2:
            movement_mode = "multi_imminent"
            if rear_is_clear and threat_ahead:
                target_angle = desired_heading
                base_thrust  = -155.0 if good_shot else -180.0
            else:
                cx_t = sum(toro_dx_dy(sx, sy, a.position[0], a.position[1], map_size)[0]
                           for a, _ in impacters[:n_imminent]) / n_imminent
                cy_t = sum(toro_dx_dy(sx, sy, a.position[0], a.position[1], map_size)[1]
                           for a, _ in impacters[:n_imminent]) / n_imminent
                centroid_escape = math.degrees(math.atan2(-cy_t, -cx_t))
                target_angle = wrap180(0.72 * centroid_escape + 0.28 * desired_heading)
                base_thrust  = 120.0 if front_is_clear else 35.0

        elif emergency:
            movement_mode = "emergency"
            if rear_is_clear and (threat_ahead or closing_into_threat > 28.0):
                target_angle = desired_heading
                base_thrust  = -145.0 if good_shot else -170.0
            elif abs(fx) > 1e-6 or abs(fy) > 1e-6:
                target_angle = wrap180(0.78 * self._smoothed_escape_deg + 0.22 * desired_heading)
                base_thrust  = 95.0 if front_is_clear else 35.0
            else:
                target_angle = desired_heading
                base_thrust  = -120.0 if rear_is_clear else 0.0

        elif danger_level > 0.5:
            movement_mode = "high_danger"
            if rear_is_clear and threat_ahead and closing_into_threat > 12.0:
                target_angle = desired_heading
                base_thrust  = -115.0 if good_shot else -140.0
            elif abs(fx) > 1e-6 or abs(fy) > 1e-6:
                target_angle = wrap180(0.68 * self._smoothed_escape_deg + 0.32 * desired_heading)
                if front_is_clear and not good_shot:
                    base_thrust = 70.0
                elif good_shot:
                    base_thrust = -30.0 if rear_is_clear else 0.0
                else:
                    base_thrust = 20.0
            else:
                target_angle = desired_heading
                base_thrust  = -85.0 if rear_is_clear else 0.0

        elif danger_level > 0.3:
            movement_mode = "mid_danger"
            if rear_is_clear and threat_ahead and (closing_into_threat > 10.0 or center_dist < 245.0):
                target_angle = desired_heading
                base_thrust  = -75.0 if good_shot else -105.0
            elif abs(fx) > 1e-6 or abs(fy) > 1e-6:
                target_angle = wrap180(0.58 * self._smoothed_escape_deg + 0.42 * desired_heading)
                if good_shot:
                    base_thrust = 0.0 if front_is_clear else -35.0
                else:
                    base_thrust = 45.0 if front_is_clear else -55.0
            else:
                target_angle = desired_heading
                base_thrust  = 0.0 if good_shot else -50.0

        elif medium > 0.2:
            movement_mode = "medium"
            target_angle  = desired_heading
            base_thrust   = 30.0 if abs(fire_heading_err) < 15.0 else 55.0

        else:
            movement_mode = "low_danger"
            target_angle  = desired_heading
            base_thrust   = 75.0

        # how far off is heading from where we want to go
        err = wrap180(target_angle - heading)

        # dont go full throttle if were also trying to spin
        if base_thrust > 0:
            thrust = base_thrust * fuzzy_thrust_scale(abs(err))
        elif base_thrust < 0:
            thrust = base_thrust * max(0.85, _falling(abs(err), 20.0, 60.0))
        else:
            thrust = 0.0

        turn_rate = max(-180.0, min(180.0, err * 4.0))

        if hasattr(ship_state, "thrust_range"):
            lo, hi = ship_state.thrust_range
            thrust = max(lo, min(hi, thrust))
        if hasattr(ship_state, "turn_rate_range"):
            lo, hi = ship_state.turn_rate_range
            turn_rate = max(lo, min(hi, turn_rate))

        self._last_turn_rate = turn_rate

        debug_cleanup = (
            cleanup_mode
            or (n_ast <= 2 and abs(fire_heading_err) < max(angle_limit + 6.0, 10.0))
            or (n_ast == 1 and not fire and danger_level < 0.35)
        )
        if debug_cleanup and (self.debug_counter % DEBUG_PRINT_EVERY == 0):
            reasons = []
            if intercept is None:
                reasons.append("no_intercept")
            else:
                if bullet_t >= MAX_BULLET_TIME:
                    reasons.append(f"bullet_t={bullet_t:.2f}>=max")
                if abs(fire_heading_err) >= angle_limit:
                    reasons.append(f"err={abs(fire_heading_err):.2f}>lim={angle_limit:.2f}")
                if shot_q < FIRE_Q_STANDARD:
                    reasons.append(f"shot_q={shot_q:.2f}")
            if not stable:
                reasons.append(f"unstable_turn(last={turn_rate_mag:.1f})")
            if self._good_aim_frames < aim_frames_required:
                reasons.append(f"aim_frames={self._good_aim_frames}")
            if small_target_guard:
                reasons.append("small_target_guard")
            if cleanup_mode:
                reasons.append("cleanup_mode")
            if not reasons and fire:
                reasons.append("fire_ok")
            """print(
                f"[HFv3 {self.debug_counter:05d}] mode={movement_mode} n_ast={n_ast} "
                f"target_size={target_size} "
                f"target=({fire_target.position[0]:.1f},{fire_target.position[1]:.1f}) "
                f"des={desired_heading:.1f} err={abs(fire_heading_err):.1f} "
                f"lim={angle_limit:.1f} shot_q={shot_q:.2f} "
                f"bt={bullet_t:.2f} dist={fire_dist:.1f} stable={stable} "
                f"danger={danger_level:.2f} ew={engage_w:.2f} rw={reposition_w:.2f} "
                f"ew2={evade_w:.2f} aim_frames={self._good_aim_frames} fire={fire} "
                f"thrust={thrust:.1f} turn={turn_rate:.1f} "
                f"reason={'|'.join(reasons)}"
            )"""

        drop_mine = False

        if (
            danger_level > 0.6          # only when things are actually scary
            and not rear_is_clear       # something behind us
            and closing_into_threat < 0 # we're moving away from it
            and center_dist < 220       # it's pretty close
        ):
            drop_mine = True
            self._mine_escape_timer = 18
            if self._mine_escape_timer > 0 and thrust < 100.0:
                thrust = 100.0
            if self._mine_escape_timer > 0 and thrust < 0:
                thrust = 0.0
        return float(thrust), float(turn_rate), fire, drop_mine

    @property
    def name(self) -> str:
        return "HybridFuzzyController v3"