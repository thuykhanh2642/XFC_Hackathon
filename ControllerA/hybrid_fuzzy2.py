import math
from collections import deque
from kesslergame.controller import KesslerController
from util import wrap180, triag

# --- ship facts the universe agreed on ---
SHIP_RADIUS = 20.0
BULLET_SPEED = 800.0  # go fast little bullet

# --- shooting timing ---
FIRE_COOLDOWN_FRAMES = 0   # no mercy, fire every frame
AIM_STABLE_REQUIRED = 1    # one good frame is enough
GUN_LOCK_FRAMES = 14       # stick with a target for a bit

# --- aim geometry ---
MAX_BULLET_TIME = 2.0      # don't shoot at asteroids in another timezone
MIN_ANGLE_LIMIT = 1.8      # smallest allowed aim window
STABLE_TURN_RATE = 42.0    # spinning faster than this means we're not stable
AIM_SAFETY = 1.15          # fudge factor so we don't shave the hitbox

# --- fire quality gates (fuzzy score thresholds) ---
FIRE_Q_STANDARD  = 0.35    # good enough
FIRE_Q_ENGAGE    = 0.45    # picky mode
FIRE_Q_PRESSURE  = 0.25    # desperate mode
FIRE_Q_GOOD_SHOT = 0.55    # confident enough to stop moving

# --- evasion: run away, but elegantly ---
EVASION_RANGE = 380.0      # how far to care about incoming rocks
EVASION_FORCE_GAIN = 3.5   # how hard we push away
VELOCITY_FORCE_GAIN = 0.8
MAX_EVASION_SPEED = 220.0
MIN_EVASION_DIST = 160.0
ESCAPE_SMOOTHING = 0.50    # low-pass on escape heading: 0=instant, 1=comatose
TURN_FIRST_THRESH = 24.0   # back-compat name, not used directly anymore

# --- action search ---
TURN_GAIN = 7.0              # P-term gain for turning
TURN_SOFTEN_THRESH = 32.0    # start reducing thrust when error > this
SEARCH_DT = 0.16             # seconds per lookahead step
SEARCH_STEPS = 8             # steps to simulate (~1.3 s horizon)
SAFE_MARGIN = 70.0           # gap below this and we panic
STATIONARY_SPEED = 28.0      # slow enough to call "stopped"
COMMIT_FRAMES = 3            # frames before we're allowed to change our mind
COMMIT_MARGIN = 22.0         # how much better the new action must be to switch
TIE_MARGIN = 12.0            # tie-break window for mirrored actions
SYMMETRY_SIDE_BONUS = 7.0    # nudge toward our preferred turning side
STABLE_SHOT_HOLD_BONUS = 40.0 # big bonus for staying still on a good shot
TARGET_APPROACH_GAIN = 5.0   # reward for drifting toward the target

# --- PI controller for heading ---
INTEGRAL_GAIN   = 0.4   # how strongly the I-term nudges turn rate
INTEGRAL_DECAY  = 0.88  # per-frame leak: old errors fade, new ones dominate
INTEGRAL_UPDATE = 0.06  # how fast error feeds into the integrator
INTEGRAL_CLAMP  = 50.0  # cap so the I-term doesn't go full maniac

ACTION_LIBRARY = [
    ("hold", 0.0, 0.0),
    ("forward", 0.0, 75.0),
    ("brake", 0.0, -70.0),
    ("hard_brake", 0.0, -130.0),
    ("left_soft", -22.0, 35.0),
    ("right_soft", 22.0, 35.0),
    ("left", -48.0, 20.0),
    ("right", 48.0, 20.0),
    ("back_left", -38.0, -65.0),
    ("back_right", 38.0, -65.0),
]

# Wrap a delta into [-size/2, size/2] for toroidal maps
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

def direct_dx_dy(sx, sy, ax, ay):
    return ax - sx, ay - sy

# Solve the bullet-meets-asteroid quadratic. Returns (hit_dx, hit_dy, time) or None.
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

def direct_intercept(ship_pos, ship_vel, ast_pos, ast_vel):
    sx, sy = ship_pos
    svx, svy = ship_vel
    dx = ast_pos[0] - sx
    dy = ast_pos[1] - sy
    avx, avy = ast_vel
    rvx = avx - svx
    rvy = avy - svy
    return _solve_intercept(dx, dy, rvx, rvy)

# Fuzzy membership functions: trap, rising edge, falling edge
def _trap(x, a, b, c, d):
    """Trapezoidal MF: 0 outside [a,d], ramp up a→b, flat b→c, ramp down c→d."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (d - x) / (d - c)

def _rising(x, lo, hi):
    """Rising-edge MF: 0 at x<=lo, linear ramp, 1 at x>=hi."""
    if x <= lo: return 0.0
    if x >= hi: return 1.0
    return (x - lo) / (hi - lo)

def _falling(x, lo, hi):
    """Falling-edge MF: 1 at x<=lo, linear ramp, 0 at x>=hi."""
    return 1.0 - _rising(x, lo, hi)

# How good is this shot? Combines aim, flight time, distance, stability into [0,1].
# All four factors multiply, so being terrible at one kills the score entirely.
def fuzzy_shot_quality(heading_err_abs, angle_limit, bullet_t, fire_dist, stable):
    aim  = _falling(heading_err_abs, angle_limit * 0.50, angle_limit * 1.10)
    time = _falling(bullet_t,  0.60, MAX_BULLET_TIME)
    dist = _falling(fire_dist, 350.0, 900.0)
    stab = 1.0 if stable else 0.70
    return aim * time * dist * stab

# Smoothly reduce thrust when we're turning hard. Full power below 10°, mostly gone above 35°.
def fuzzy_thrust_scale(turn_err_abs):
    return max(0.15, _falling(turn_err_abs, 10.0, 35.0))

# How much are we in each mode? Overlapping bands mean no jarring snap transitions.
def fuzzy_danger_blend(danger_level):
    engage_w     = _falling(danger_level, 0.30, 0.50)
    reposition_w = _trap(danger_level,    0.30, 0.42, 0.62, 0.75)
    evade_w      = _rising(danger_level,  0.58, 0.75)
    return engage_w, reposition_w, evade_w

# Time-to-impact via AABB interval check. Returns seconds until collision, or None.
def compute_tti(ship_pos, ship_vel, ast_pos, ast_vel, ast_radius, map_size):
    sx, sy = ship_pos
    dx, dy = toro_dx_dy(sx, sy, ast_pos[0], ast_pos[1], map_size)
    vax, vay = ast_vel
    vsx, vsy = ship_vel
    so = ast_radius + SHIP_RADIUS

    dvx = vsx - vax
    dvy = vsy - vay

    if abs(dvx) > 1e-9:
        tixs = sorted([-(dx + so) / dvx, -(dx - so) / dvx])
        tix_lo, tix_hi = tixs[0], tixs[1]
    else:
        
        if abs(dx) >= so:
            return None
        tix_lo, tix_hi = -1e18, 1e18

    if abs(dvy) > 1e-9:
        tiys = sorted([-(dy + so) / dvy, -(dy - so) / dvy])
        tiy_lo, tiy_hi = tiys[0], tiys[1]
    else:
        if abs(dy) >= so:
            return None
        tiy_lo, tiy_hi = -1e18, 1e18

    enter = max(tix_lo, tiy_lo)
    leave = min(tix_hi, tiy_hi)
    if leave <= enter:
        return None
    tti = enter if enter > 0 else leave
    return tti if tti > 0 else None

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

def threat_time_metrics(ship_pos, ship_vel, ast_pos, ast_vel, ast_radius, map_size):
    sx, sy = ship_pos
    svx, svy = ship_vel
    dx, dy = toro_dx_dy(sx, sy, ast_pos[0], ast_pos[1], map_size)
    rvx = ast_vel[0] - svx
    rvy = ast_vel[1] - svy

    center_dist = math.hypot(dx, dy)
    surface_dist = max(0.0, center_dist - ast_radius - SHIP_RADIUS)
    if center_dist > 1e-6:
        closing_speed = (rvx * dx + rvy * dy) / center_dist
    else:
        closing_speed = 0.0

    rv2 = rvx * rvx + rvy * rvy
    if rv2 < 1e-8:
        t_closest = 0.0
        closest_center = center_dist
    else:
        t_closest = max(0.0, -((dx * rvx) + (dy * rvy)) / rv2)
        cdx = dx + rvx * t_closest
        cdy = dy + rvy * t_closest
        closest_center = math.hypot(cdx, cdy)
    closest_surface = max(0.0, closest_center - ast_radius - SHIP_RADIUS)

    tti = compute_tti(ship_pos, ship_vel, ast_pos, ast_vel, ast_radius, map_size)
    return dx, dy, center_dist, surface_dist, closing_speed, t_closest, closest_surface, tti

def threat_direction_bucket(rel_bearing_deg):
    a = abs(rel_bearing_deg)
    if a <= 50.0:
        return "front"
    if a >= 130.0:
        return "rear"
    return "side"

def score_threat_entry(entry):
    dist_term = 950.0 / max(entry["surface_dist"] + 35.0, 35.0)
    closing_term = max(entry["closing_speed"], 0.0) * 0.085

    if entry["tti"] is not None:
        tti_term = 260.0 / max(entry["tti"] + 0.18, 0.18)
    else:
        tti_term = 0.0

    if entry["t_closest"] <= 1.7:
        miss_term = 180.0 / max(entry["closest_surface"] + 28.0, 28.0)
    else:
        miss_term = 0.0

    size_term = {4: 22.0, 3: 13.0, 2: 6.0, 1: 0.0}.get(entry["size"], 0.0)
    direction_term = {
        "rear": 16.0 if entry["closing_speed"] > 0 else 7.0,
        "side": 10.0 if entry["closing_speed"] > 0 else 4.0,
        "front": 4.0,
    }[entry["direction"]]

    imminent_bonus = 0.0
    if entry["tti"] is not None and entry["tti"] < 1.2:
        imminent_bonus += 32.0
    if entry["closest_surface"] < 70.0:
        imminent_bonus += 24.0

    return dist_term + closing_term + tti_term + miss_term + size_term + direction_term + imminent_bonus

# Score every asteroid and sort worst-first. This is our threat radar.
def build_threat_table(asteroids, ship_pos, ship_vel, heading, map_size):
    threats = []
    for a in asteroids:
        ax, ay = a.position
        avx, avy = getattr(a, "velocity", (0.0, 0.0))
        radius = getattr(a, "radius", 0.0)
        size = getattr(a, "size", 2)
        dx, dy, center_dist, surface_dist, closing_speed, t_closest, closest_surface, tti = threat_time_metrics(
            ship_pos, ship_vel, (ax, ay), (avx, avy), radius, map_size
        )
        world_bearing = math.degrees(math.atan2(dy, dx))
        rel_bearing = wrap180(world_bearing - heading)
        direction = threat_direction_bucket(rel_bearing)
        entry = {
            "asteroid": a,
            "pos": (ax, ay),
            "vel": (avx, avy),
            "radius": radius,
            "size": size,
            "dx": dx,
            "dy": dy,
            "center_dist": center_dist,
            "surface_dist": surface_dist,
            "closing_speed": closing_speed,
            "t_closest": t_closest,
            "closest_surface": closest_surface,
            "tti": tti,
            "bearing": world_bearing,
            "rel_bearing": rel_bearing,
            "direction": direction,
        }
        entry["danger_score"] = score_threat_entry(entry)
        threats.append(entry)
    threats.sort(key=lambda e: e["danger_score"], reverse=True)
    return threats

# Distill the threat table into a single danger level and the worst offenders.
def summarize_threats(threats):
    if not threats:
        return None, [], 0.0, None, 0
    primary = threats[0]
    top_threats = threats[:3]
    max_score = primary["danger_score"]
    danger_level = min(1.0, max_score / 95.0)

    imminent = [t for t in threats if t["tti"] is not None and t["tti"] < 1.5]
    soonest_tti = min((t["tti"] for t in imminent), default=None)
    n_imminent = len(imminent)

    if soonest_tti is not None and soonest_tti < 1.2:
        danger_level = max(danger_level, 0.82)
    if soonest_tti is not None and soonest_tti < 0.7:
        danger_level = 1.0
    if primary["surface_dist"] < 85.0:
        danger_level = max(danger_level, 0.9)

    n_nearby = sum(1 for t in threats if t["surface_dist"] < 380.0)
    crowd_pressure = min(0.55, n_nearby * 0.028)
    danger_level = max(danger_level, crowd_pressure)

    return primary, top_threats, danger_level, soonest_tti, n_imminent

# Are we shooting, running, or somewhere in between?
def choose_mode(primary_threat, top_threats, danger_level, soonest_tti, n_imminent):
    if primary_threat is None:
        return "engage"

    if n_imminent >= 2 and soonest_tti is not None and soonest_tti < 1.15:
        return "evade"
    if primary_threat["surface_dist"] < 75.0:
        return "evade"
    if primary_threat["tti"] is not None and primary_threat["tti"] < 0.9:
        return "evade"

    if len(top_threats) == 1 and top_threats[0]["danger_score"] < 26.0:
        return "cleanup"

    rear_pressure = sum(
        1 for t in top_threats
        if t["direction"] == "rear" and t["closing_speed"] > 0 and (
            (t["tti"] is not None and t["tti"] < 1.6) or
            (t["t_closest"] < 1.2 and t["closest_surface"] < 90.0)
        )
    )
    if rear_pressure >= 1:
        return "reposition"

    if danger_level > 0.48 or n_imminent >= 1:
        return "reposition"

    return "engage"

# Something behind us is about to ruin our day.
def detect_rear_interrupt(top_threats):
    if not top_threats:
        return False, None
    best = None
    best_score = -1e9
    for t in top_threats:
        if t["direction"] not in ("rear", "side"):
            continue
        if t["closing_speed"] <= 0.0:
            continue
        score = 0.0
        if t["tti"] is not None:
            score += 260.0 / max(t["tti"] + 0.2, 0.2)
        if t["t_closest"] <= 1.6:
            score += 180.0 / max(t["closest_surface"] + 20.0, 20.0)
        score += max(t["closing_speed"], 0.0) * 0.15
        if t["direction"] == "rear":
            score += 22.0
        else:
            score += 12.0
        if score > best_score:
            best_score = score
            best = t

    if best is None:
        return False, None

    urgent = (
        (best["tti"] is not None and best["tti"] < 1.25) or
        (best["t_closest"] < 1.0 and best["closest_surface"] < 85.0)
    )
    return urgent, best

def calculate_threat_priority(asteroid, ship_pos, ship_vel, map_size):
    ax, ay = asteroid.position
    dx, dy = toro_dx_dy(ship_pos[0], ship_pos[1], ax, ay, map_size)
    distance = math.hypot(dx, dy)

    avx, avy = getattr(asteroid, "velocity", (0.0, 0.0))
    closing_speed = ((avx - ship_vel[0]) * dx + (avy - ship_vel[1]) * dy) / max(distance, 1)

    size = getattr(asteroid, "size", 2)
    size_bonus = {4: 5.0, 3: 3.0, 2: 1.0, 1: -1.0}.get(size, 0.0)
    return (900.0 / max(distance, 1.0)) + max(closing_speed, 0.0) / 60.0 + size_bonus

def max_aim_error(asteroid, distance):
    radius = float(getattr(asteroid, "radius", 10.0))
    size = int(getattr(asteroid, "size", 2))
    
    effective_radius = radius + {1: 5.0, 2: 3.0, 3: 1.5}.get(size, 0.0)
    if distance < effective_radius:
        return 180.0
    safe_ratio = min(effective_radius / distance, 1.0)
    error_rad = math.asin(safe_ratio)
    return math.degrees(error_rad) * AIM_SAFETY

def gun_target_score(asteroid, ship_pos, ship_vel, heading_deg):
    intercept = direct_intercept(ship_pos, ship_vel, asteroid.position, getattr(asteroid, "velocity", (0.0, 0.0)))
    if intercept is None:
        return -1e9

    dx, dy, t = intercept
    aim_heading = math.degrees(math.atan2(dy, dx))
    aim_err = abs(wrap180(aim_heading - heading_deg))
    dist = math.hypot(dx, dy)
    size = int(getattr(asteroid, "size", 2))
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

def forward_clearance(ship_pos, heading_deg, asteroids, map_size, check_range=220.0, safety=55.0):
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

def projected_speed_toward(ship_vel, dir_x, dir_y):
    mag = math.hypot(dir_x, dir_y)
    if mag < 1e-6:
        return 0.0
    ux, uy = dir_x / mag, dir_y / mag
    return ship_vel[0] * ux + ship_vel[1] * uy

# Sum repulsion vectors from nearby asteroids. Closing fast = extra push.
def potential_field_force(ship_pos, ship_vel, asteroids, map_size, range_limit=EVASION_RANGE):
    """
    Returns a (fx, fy) force vector pointing away from danger.
    Repulsion is stronger for close asteroids and for those moving toward the ship.
    """
    fx, fy = 0.0, 0.0
    for a in asteroids:
        dx, dy = toro_dx_dy(ship_pos[0], ship_pos[1], a.position[0], a.position[1], map_size)
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

def wrap_position(x, y, map_size):
    w, h = map_size
    return x % w, y % h

def lane_density_score(ship_pos, lane_heading_deg, asteroids, map_size, look_range=340.0, half_width=85.0):
    hx = math.cos(math.radians(lane_heading_deg))
    hy = math.sin(math.radians(lane_heading_deg))
    total = 0.0
    sx, sy = ship_pos
    for a in asteroids:
        ax, ay = a["pos"]
        dx, dy = toro_dx_dy(sx, sy, ax, ay, map_size)
        proj = dx * hx + dy * hy
        if 0.0 < proj < look_range:
            perp = abs(dx * (-hy) + dy * hx)
            lane_w = half_width + a["radius"]
            if perp < lane_w:
                closeness = 1.0 - (proj / look_range)
                cross_pen = 1.0 - min(perp / lane_w, 1.0)
                avx, avy = a["vel"]
                closing = max(0.0, -((avx) * hx + (avy) * hy))
                total += 1.5 * closeness + 1.1 * cross_pen + 0.006 * closing
    return total

def threat_table_snapshot(threats):
    snaps = []
    for t in threats:
        snaps.append({
            "pos": (float(t["pos"][0]), float(t["pos"][1])),
            "vel": (float(t["vel"][0]), float(t["vel"][1])),
            "radius": float(t["radius"]),
            "size": int(t["size"]),
            "danger": float(t["danger_score"]),
            "direction": t["direction"],
        })
    return snaps

def min_gap_to_asteroids(ship_pos, asteroids, map_size):
    best = float("inf")
    sx, sy = ship_pos
    for a in asteroids:
        d = toro_dist(sx, sy, a["pos"][0], a["pos"][1], map_size) - a["radius"] - SHIP_RADIUS
        if d < best:
            best = d
    return best if best != float("inf") else 9999.0

def blended_escape_heading(top_threats, desired_heading, rear_interrupt=False):
    if not top_threats:
        return desired_heading
    fx = fy = 0.0
    for t in top_threats[:3]:
        dist = max(t["center_dist"], 1.0)
        weight = (t["danger_score"] + max(t["closing_speed"], 0.0) * 0.35) / dist
        if rear_interrupt and t["direction"] in ("rear", "side"):
            weight *= 1.6
        fx -= t["dx"] * weight
        fy -= t["dy"] * weight
    if abs(fx) < 1e-6 and abs(fy) < 1e-6:
        return desired_heading
    esc = math.degrees(math.atan2(fy, fx))
    mix = 0.78 if rear_interrupt else 0.62
    return wrap180(mix * esc + (1.0 - mix) * desired_heading)

# Roll each candidate action forward in time and score how not-dead we are.
def simulate_action_score(
    ship_pos,
    ship_vel,
    ship_heading,
    thrust_cmd,
    target_heading,
    threat_snaps,
    map_size,
    desired_fire_heading,
    good_shot,
    mode,
    rear_interrupt,
    target_pos=None,          
    target_vel=None,          
):
    sx, sy = float(ship_pos[0]), float(ship_pos[1])
    svx, svy = float(ship_vel[0]), float(ship_vel[1])
    heading = float(ship_heading)
    score = 0.0
    horizon = [dict(a) for a in threat_snaps]

    if target_pos is not None and target_vel is not None:
        tx, ty = float(target_pos[0]), float(target_pos[1])
        tvx, tvy = float(target_vel[0]), float(target_vel[1])
        prev_target_dist = toro_dist(sx, sy, tx, ty, map_size)
    else:
        tx = ty = tvx = tvy = None
        prev_target_dist = None

    for step in range(SEARCH_STEPS):
        err = wrap180(target_heading - heading)
        turn_rate = max(-180.0, min(180.0, err * TURN_GAIN))

        step_thrust = thrust_cmd
        if abs(err) > TURN_SOFTEN_THRESH:
            if step_thrust > 0:
                step_thrust *= 0.42
            elif step_thrust < 0:
                step_thrust *= 0.92

        heading = wrap180(heading + turn_rate * SEARCH_DT)
        hx = math.cos(math.radians(heading))
        hy = math.sin(math.radians(heading))

        svx += hx * step_thrust * SEARCH_DT
        svy += hy * step_thrust * SEARCH_DT
        sx, sy = wrap_position(sx + svx * SEARCH_DT, sy + svy * SEARCH_DT, map_size)

        for a in horizon:
            ax, ay = a["pos"]
            avx, avy = a["vel"]
            a["pos"] = wrap_position(ax + avx * SEARCH_DT, ay + avy * SEARCH_DT, map_size)

        if target_pos is not None:
            tx, ty = wrap_position(tx + tvx * SEARCH_DT, ty + tvy * SEARCH_DT, map_size)

            retreating = (
                mode == "cleanup"
                and target_vel is not None
                and tx is not None
            )
            if retreating:
                tp_dx2 = tx - sx
                tp_dy2 = ty - sy
                tp_d2 = math.hypot(tp_dx2, tp_dy2)
                if tp_d2 > 1e-6:
                    closing2 = (tvx * tp_dx2 + tvy * tp_dy2) / tp_d2
                    retreating = closing2 < -1.0 and tp_d2 > 280.0
                else:
                    retreating = False

            if not good_shot or retreating:
                current_target_dist = toro_dist(sx, sy, tx, ty, map_size)
                reduction = prev_target_dist - current_target_dist
                score += reduction * TARGET_APPROACH_GAIN
                prev_target_dist = current_target_dist

        gap = min_gap_to_asteroids((sx, sy), horizon, map_size)
        # instant death: inside a rock
        if gap < 0.0:
            return -1e6 + gap * 1000.0

        score += min(gap, 220.0) * 1.15
        if gap < SAFE_MARGIN:
            score -= (SAFE_MARGIN - gap) * 50.0
        elif gap < 110.0:
            score -= (110.0 - gap) * 8.0

        worst_pen = 0.0
        rear_pen = 0.0
        for a in horizon[:3]:
            dx, dy = toro_dx_dy(sx, sy, a["pos"][0], a["pos"][1], map_size)
            dist = max(math.hypot(dx, dy), 1.0)
            surf = dist - a["radius"] - SHIP_RADIUS
            avx, avy = a["vel"]
            closing = ((avx - svx) * dx + (avy - svy) * dy) / dist
            local = max(closing, 0.0) * 1.0 + max(0.0, 95.0 - surf) * 1.6 + a.get("danger", 0.0) * 0.22
            worst_pen = max(worst_pen, local)
            bearing = wrap180(math.degrees(math.atan2(dy, dx)) - heading)
            if abs(bearing) > 105.0 and closing > 0.0:
                rear_pen += max(0.0, 125.0 - surf) * 1.7 + closing * 0.7

        score -= worst_pen
        score -= rear_pen * (1.45 if rear_interrupt else 1.0)

        fire_err = abs(wrap180(desired_fire_heading - heading))

        eng_w, rep_w, evd_w = fuzzy_danger_blend(
            0.0 if mode == "cleanup" else
            (1.0 if mode == "evade" else
             (0.8 if rear_interrupt else
              min(1.0, (worst_pen + rear_pen * 0.5) / 120.0)))
        )

        lane_mult_fuzzy = 15.0 * eng_w + 22.0 * rep_w + 36.0 * evd_w
        if mode == "cleanup":
            lane_mult_fuzzy = 8.0
        score -= lane_mult_fuzzy * lane_density_score((sx, sy), heading, horizon, map_size)

        if mode == "engage":
            if good_shot:
                score -= fire_err * fire_err * 0.5
            else:
                score -= fire_err * (0.8 * eng_w + 0.3 * rep_w + 0.05 * evd_w)
        elif mode == "cleanup":
            score -= fire_err * 2.5
        elif mode == "reposition":
            score -= fire_err * (0.25 if good_shot else 0.10)
        else:
            score -= fire_err * 0.05

        score -= 0.010 * (svx * svx + svy * svy)
        score -= step * 0.9

    final_gap = min_gap_to_asteroids((sx, sy), horizon, map_size)
    score += min(final_gap, 260.0) * 1.9
    return score

class hybrid_controller_v2(KesslerController):
    def __init__(self):
        self.debug_counter = 0
        self._gun_lock_pos = None
        self._gun_lock_timer = 0
        self._fire_cooldown = 0
        self._good_aim_frames = 0
        self._last_turn_rate = 0.0
        self._smoothed_escape_deg = None
        self._last_top_threats = []
        self._committed_action = "hold"
        self._commit_timer = 0
        self._preferred_side = 1
        self._debug_enabled = True
        self._last_action_debug = None
        self._integral_error = 0.0
        self._integral_gain = INTEGRAL_GAIN

    def _action_side(self, name):
        if "left" in name:
            return -1
        if "right" in name:
            return 1
        return 0

    def _is_mirrored_pair(self, a, b):
        pairs = {("left", "right"), ("left_soft", "right_soft"), ("back_left", "back_right")}
        return (a, b) in pairs or (b, a) in pairs

    def _prefer_committed_over_best(self, scored_actions, best_name, best_score):
        if self._commit_timer <= 0:
            return best_name, best_score
        for name, score, target_heading, thrust_cmd in scored_actions:
            if name == self._committed_action and score >= best_score - COMMIT_MARGIN:
                return name, score
        return best_name, best_score

    # Score all actions and pick the one that keeps us alive longest.
    def _choose_movement_action(
        self,
        ship_pos,
        ship_vel,
        heading,
        map_size,
        desired_heading,
        danger_level,
        good_shot,
        mode,
        top_threats,
        rear_interrupt,
        target_pos=None,          
        target_vel=None,          
    ):
        threat_snaps = threat_table_snapshot(top_threats if top_threats else [])
        escape_heading = blended_escape_heading(top_threats, desired_heading, rear_interrupt=rear_interrupt)

        if rear_interrupt:
            reference_heading = escape_heading
        elif mode == "evade":
            reference_heading = escape_heading
        elif mode == "reposition":
            reference_heading = wrap180(0.58 * escape_heading + 0.42 * desired_heading)
        else:
            reference_heading = desired_heading

        reverse_bias = danger_level * 18.0
        if rear_interrupt:
            reverse_bias += 28.0
        if mode == "evade":
            reverse_bias += 22.0

        ship_speed = math.hypot(ship_vel[0], ship_vel[1])
        near_stationary = ship_speed < STATIONARY_SPEED

        scored_actions = []
        for name, heading_offset, thrust_cmd in ACTION_LIBRARY:
            target_heading = wrap180(reference_heading + heading_offset)
            score = simulate_action_score(
                ship_pos, ship_vel, heading, thrust_cmd, target_heading, threat_snaps, map_size,
                desired_heading, good_shot, mode, rear_interrupt,
                target_pos=target_pos,          
                target_vel=target_vel,
            )

            if thrust_cmd < 0.0:
                score += reverse_bias
            if rear_interrupt and name in ("hold", "forward"):
                score -= 40.0
            if mode == "cleanup":
                if name == "hold":
                    score += 42.0
                elif thrust_cmd > 0:
                    score -= 25.0
            elif mode == "engage":
                if name == "hold" and good_shot:
                    score += 16.0
                if thrust_cmd < 0.0 and danger_level < 0.35:
                    score -= 12.0
                off_target = abs(wrap180(target_heading - desired_heading))
                score -= off_target * 0.9
                if off_target <= 5.0 and name in ("hold", "left_soft", "right_soft"):
                    score += 10.0
                
                n_nearby_snap = sum(1 for a in threat_snaps if
                    math.hypot(a["pos"][0] - ship_pos[0], a["pos"][1] - ship_pos[1]) < 400.0)
                if name in ("hold", "brake", "hard_brake") and n_nearby_snap >= 6:
                    score -= (n_nearby_snap - 5) * 14.0
            elif mode == "reposition":
                if name in ("left_soft", "right_soft"):
                    score += 12.0

            if good_shot and mode in ("engage", "cleanup") and not rear_interrupt:
                if name == "hold":
                    score += STABLE_SHOT_HOLD_BONUS
                elif abs(heading_offset) <= 22.0:
                    score += 7.0
                else:
                    score -= 5.0

            if good_shot and target_pos is not None and thrust_cmd > 0:
                
                score -= 30.0

            if (mode == "cleanup" and thrust_cmd > 0
                    and target_vel is not None and target_pos is not None):
                
                tp_dx, tp_dy = target_pos[0] - ship_pos[0], target_pos[1] - ship_pos[1]
                tp_dist = math.hypot(tp_dx, tp_dy)
                if tp_dist > 1e-6:
                    closing = (target_vel[0] * tp_dx + target_vel[1] * tp_dy) / tp_dist
                    if closing < -1.0 and tp_dist > 280.0:
                        
                        score += 30.0

            side = self._action_side(name)
            if near_stationary and side != 0:
                score += self._preferred_side * side * SYMMETRY_SIDE_BONUS

            if name == self._committed_action and self._commit_timer > 0:
                score += 10.0

            scored_actions.append((name, score, target_heading, thrust_cmd))

        scored_actions.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score, best_target, best_thrust = scored_actions[0]
        best_name, best_score = self._prefer_committed_over_best(scored_actions, best_name, best_score)

        for name, score, target_heading, thrust_cmd in scored_actions:
            if name == best_name:
                best_target = target_heading
                best_thrust = thrust_cmd
                break

        if len(scored_actions) >= 2:
            n2, s2, t2, th2 = scored_actions[1]
            if self._is_mirrored_pair(best_name, n2) and abs(best_score - s2) <= TIE_MARGIN:
                committed_side = self._action_side(self._committed_action)
                preferred_side = committed_side if committed_side != 0 else self._preferred_side
                if self._action_side(best_name) != preferred_side:
                    best_name, best_score, best_target, best_thrust = n2, s2, t2, th2

        chosen_side = self._action_side(best_name)
        if chosen_side != 0:
            self._preferred_side = chosen_side

        if best_name == self._committed_action:
            self._commit_timer = max(self._commit_timer - 1, 0) + 1
        else:
            self._committed_action = best_name
            self._commit_timer = COMMIT_FRAMES

        self._last_action_debug = {
            "reference_heading": reference_heading,
            "escape_heading": escape_heading,
            "mode": mode,
            "rear_interrupt": rear_interrupt,
            "good_shot": good_shot,
            "danger": danger_level,
            "chosen": (best_name, best_score, best_target, best_thrust),
            "top_scores": scored_actions[:4],
        }

        return best_name, best_target, best_thrust

    # Pick the best asteroid to shoot. Locks onto a target for a few frames to avoid flailing.
    def _pick_gun_target(self, asteroids, ship_pos, ship_vel, heading, map_size, closest_asteroid=None):
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
            score = gun_target_score(a, ship_pos, ship_vel, heading)
            if score > best_score:
                best_score = score
                best = a
        if best is None and closest_asteroid is not None:
            best = closest_asteroid

        if best is not None:
            if self._gun_lock_pos is None or toro_dist(
                best.position[0], best.position[1],
                self._gun_lock_pos[0], self._gun_lock_pos[1],
                map_size,
            ) > 120:
                
                self._integral_error = 0.0
            self._gun_lock_pos = best.position
            self._gun_lock_timer = GUN_LOCK_FRAMES
        return best

    # Main decision loop. Called every frame. Must not crash.
    def actions(self, ship_state, game_state):
        self.debug_counter += 1

        if self._fire_cooldown > 0:
            self._fire_cooldown -= 1
        if self._commit_timer > 0:
            self._commit_timer -= 1

        asteroids = getattr(game_state, "asteroids", [])
        if not asteroids:
            self._good_aim_frames = 0
            return 0.0, 0.0, False, False

        sx, sy = ship_state.position
        heading = ship_state.heading
        svx, svy = getattr(ship_state, "velocity", (0.0, 0.0))
        map_size = getattr(game_state, "map_size", (1000, 800))

        # Phase A: who wants us dead?
        threats = build_threat_table(asteroids, (sx, sy), (svx, svy), heading, map_size)
        primary_threat, top_threats, danger_level, soonest_tti, n_imminent = summarize_threats(threats)
        if primary_threat is None:
            self._good_aim_frames = 0
            return 0.0, 0.0, False, False

        mode = choose_mode(primary_threat, top_threats, danger_level, soonest_tti, n_imminent)
        rear_interrupt, rear_interrupt_threat = detect_rear_interrupt(top_threats)
        if rear_interrupt:
            danger_level = max(danger_level, 0.88)
            if mode == "engage":
                mode = "evade"
            elif mode == "cleanup":
                mode = "reposition"

        closest_asteroid = primary_threat["asteroid"]
        dx = primary_threat["dx"]
        dy = primary_threat["dy"]
        center_dist = primary_threat["center_dist"]
        medium = triag(center_dist, 250, 400, 600)

        impacters = [(t["asteroid"], t["tti"]) for t in threats if t["tti"] is not None]
        impacters.sort(key=lambda x: x[1])
        dodge_target_ast = impacters[0][0] if impacters else closest_asteroid

        self._last_top_threats = top_threats
        self._last_mode = mode
        self._last_rear_interrupt = rear_interrupt
        self._last_rear_interrupt_threat = rear_interrupt_threat

        # Phase B: pick a target and compute the intercept
        fire_target = self._pick_gun_target(asteroids, (sx, sy), (svx, svy), heading, map_size, closest_asteroid)
        target_vel = getattr(fire_target, "velocity", (0.0, 0.0))
        target_size = getattr(fire_target, "size", 2)

        intercept = direct_intercept((sx, sy), (svx, svy), fire_target.position, target_vel)
        if intercept is not None:
            fire_dx, fire_dy, bullet_t = intercept
        else:
            fire_dx, fire_dy = direct_dx_dy(sx, sy, fire_target.position[0], fire_target.position[1])
            fire_dist_fallback = math.hypot(fire_dx, fire_dy)
            bullet_t = fire_dist_fallback / BULLET_SPEED if fire_dist_fallback > 1e-6 else MAX_BULLET_TIME + 1.0

        desired_heading = math.degrees(math.atan2(fire_dy, fire_dx))
        fire_heading_err = wrap180(desired_heading - heading)
        
        self._integral_error *= INTEGRAL_DECAY
        self._integral_error = max(
            -INTEGRAL_CLAMP,
            min(INTEGRAL_CLAMP, self._integral_error + fire_heading_err * INTEGRAL_UPDATE),
        )

        fire_dist = math.hypot(fire_dx, fire_dy)

        angle_limit = max_aim_error(fire_target, fire_dist)
        angle_limit = max(angle_limit, MIN_ANGLE_LIMIT)

        turn_rate_mag = abs(self._last_turn_rate)
        stable = turn_rate_mag < STABLE_TURN_RATE

        cleanup_fire_limit = max(angle_limit + 1.8, 3.8)
        shot_good_limit = cleanup_fire_limit if mode == "cleanup" else max(angle_limit * 1.18, angle_limit + 1.0, 3.0)
        aim_build_limit = cleanup_fire_limit if mode == "cleanup" else max(angle_limit * 1.10, angle_limit + 0.8, 2.8)

        shot_q = fuzzy_shot_quality(
            abs(fire_heading_err), angle_limit, bullet_t, fire_dist, stable
        )
        
        good_shot = (intercept is not None and shot_q >= FIRE_Q_GOOD_SHOT)

        cleanup_shot_ready = (
            mode == "cleanup"
            and not rear_interrupt
            and intercept is not None
            and self._fire_cooldown == 0
            and stable
            and bullet_t < min(MAX_BULLET_TIME, 0.95)
            and abs(fire_heading_err) < cleanup_fire_limit
            and fire_dist < 700
        )

        if cleanup_shot_ready:
            self._good_aim_frames = AIM_STABLE_REQUIRED
        elif intercept is not None and bullet_t < MAX_BULLET_TIME and abs(fire_heading_err) < aim_build_limit and stable:
            self._good_aim_frames += 1
        else:
            self._good_aim_frames = 0

        fire = False
        fire_block_reasons = []

        can_fire_basic = (
            intercept is not None
            and self._fire_cooldown == 0
        )

        rear_blocked = (
            rear_interrupt
            and rear_interrupt_threat is not None
            and rear_interrupt_threat["direction"] == "rear"
        )
        if rear_blocked:
            fire_block_reasons.append("rear_interrupt")

        small_target_blocked = (
            mode != "cleanup"
            and target_size <= 1
            and fire_dist > 320
            and abs(fire_heading_err) > 5.5
        )
        if small_target_blocked:
            fire_block_reasons.append("small_target_guard")

        safe_retreat_blocked = False
        target_threat = next((t for t in threats if t["asteroid"] is fire_target), None)
        if target_threat is not None:
            combined_r = target_threat["radius"] + SHIP_RADIUS
            if (target_threat["closing_speed"] < -5.0
                    and target_threat["closest_surface"] > combined_r * 6.0
                    and target_threat["t_closest"] > 6.0
                    and len(asteroids) > 1):
                safe_retreat_blocked = True
                fire_block_reasons.append(f"safe_retreat(miss={target_threat['closest_surface']:.0f})")

        hard_blocked = rear_blocked or small_target_blocked or safe_retreat_blocked

        # Fire decision: hard blocks first, then priority-ordered enable paths
        if not hard_blocked and can_fire_basic:
            
            if cleanup_shot_ready:
                fire = True

            elif (danger_level > 0.7
                    and fire_dist < 200
                    and abs(fire_heading_err) < (angle_limit + 3.0)
                    and bullet_t < MAX_BULLET_TIME):
                fire = True

            elif (self._good_aim_frames >= AIM_STABLE_REQUIRED
                    and abs(fire_heading_err) < angle_limit
                    and shot_q >= FIRE_Q_STANDARD):
                fire = True

            elif (mode == "engage"
                    and not rear_interrupt
                    and stable
                    and bullet_t < min(MAX_BULLET_TIME, 0.95)
                    and abs(fire_heading_err) < shot_good_limit
                    and fire_dist < 620
                    and danger_level < 0.60
                    and shot_q >= FIRE_Q_ENGAGE):
                fire = True

            elif (danger_level > 0.35
                    and bullet_t < min(MAX_BULLET_TIME, 1.35)
                    and abs(fire_heading_err) < max(angle_limit * 1.35, angle_limit + 2.0)
                    and fire_dist < 720
                    and shot_q >= FIRE_Q_PRESSURE):
                fire = True

        if not fire and not hard_blocked:
            
            if intercept is None:
                fire_block_reasons.append("no_intercept")
            if self._fire_cooldown != 0:
                fire_block_reasons.append(f"cooldown={self._fire_cooldown}")
            if abs(fire_heading_err) >= angle_limit:
                fire_block_reasons.append(f"err={abs(fire_heading_err):.2f}>lim={angle_limit:.2f}")
            if bullet_t >= MAX_BULLET_TIME:
                fire_block_reasons.append(f"bt={bullet_t:.2f}>max={MAX_BULLET_TIME:.2f}")
            if not stable:
                fire_block_reasons.append(f"unstable_turn(last={turn_rate_mag:.1f})")
            if self._good_aim_frames < AIM_STABLE_REQUIRED:
                fire_block_reasons.append(f"aim_frames={self._good_aim_frames}")
            if shot_q < 0.35:
                fire_block_reasons.append(f"shot_q={shot_q:.2f}")
            if fire_dist >= 950:
                fire_block_reasons.append(f"dist={fire_dist:.1f}")

        if fire:
            self._fire_cooldown = FIRE_COOLDOWN_FRAMES
            self._good_aim_frames = 0
            
            if FIRE_COOLDOWN_FRAMES > 0:
                self._integral_error = 0.0

        # Phase C: smooth evasion via potential field
        fx, fy = potential_field_force((sx, sy), (svx, svy), asteroids, map_size)

        if abs(fx) < 1e-6 and abs(fy) < 1e-6:
            raw_escape_deg = desired_heading
        else:
            raw_escape_deg = math.degrees(math.atan2(fy, fx))

        if self._smoothed_escape_deg is None:
            self._smoothed_escape_deg = raw_escape_deg
        else:
            diff = wrap180(raw_escape_deg - self._smoothed_escape_deg)
            self._smoothed_escape_deg = wrap180(self._smoothed_escape_deg + ESCAPE_SMOOTHING * diff)

        min_dist_to_any = min(
            (toro_dist(sx, sy, a.position[0], a.position[1], map_size) - getattr(a, "radius", 0) - SHIP_RADIUS)
            for a in asteroids
        )
        emergency = min_dist_to_any < MIN_EVASION_DIST

        threat_ahead = abs(wrap180(math.degrees(math.atan2(dy, dx)) - heading)) < 75.0
        closing_into_threat = projected_speed_toward((svx, svy), dx, dy)
        rear_is_clear = rear_clearance((sx, sy), heading, asteroids, map_size)
        front_is_clear = forward_clearance((sx, sy), heading, asteroids, map_size)

        # Phase D: pick the best movement action
        _, target_angle, base_thrust = self._choose_movement_action(
            (sx, sy),
            (svx, svy),
            heading,
            map_size,
            desired_heading,
            danger_level,
            good_shot,
            mode,
            top_threats,
            rear_interrupt,
            target_pos=fire_target.position,
            target_vel=target_vel,
        )

        err = wrap180(target_angle - heading)

        if base_thrust > 0:
            thrust = base_thrust * fuzzy_thrust_scale(abs(err))
        elif base_thrust < 0:
            
            thrust = base_thrust * max(0.85, _falling(abs(err), 20.0, 60.0))
        else:
            thrust = 0.0

        turn_rate = max(-180.0, min(180.0, err * TURN_GAIN + self._integral_error * self._integral_gain))

        if hasattr(ship_state, "thrust_range"):
            lo, hi = ship_state.thrust_range
            thrust = max(lo, min(hi, thrust))
        if hasattr(ship_state, "turn_rate_range"):
            lo, hi = ship_state.turn_rate_range
            turn_rate = max(lo, min(hi, turn_rate))

        self._last_turn_rate = turn_rate

        if self._debug_enabled:
            debug_cleanup = (len(asteroids) <= 2)
            debug_ring_aim = (mode in ("engage", "cleanup") and danger_level < 0.20 and fire_target is not None and abs(fire_heading_err) > max(2.0, angle_limit * 0.8))
            if debug_cleanup or debug_ring_aim:
                tparts = []
                for t in top_threats[:3]:
                    tparts.append(
                        f"{t['direction'][0].upper()} ds={t['danger_score']:.1f} cd={t['center_dist']:.1f} cs={t['closing_speed']:.1f} tca={t['t_closest']:.2f} miss={t['closest_surface']:.1f}"
                    )
                reason = 'OK' if fire else ('|'.join(dict.fromkeys(fire_block_reasons)) if fire_block_reasons else 'blocked')
                print(
                    f"[P3DBG {self.debug_counter:05d}] mode={mode} n_ast={len(asteroids)} rear_interrupt={rear_interrupt} "
                    f"target_size={target_size} des={desired_heading:.1f} err={fire_heading_err:.2f} lim={angle_limit:.2f} "
                    f"bt={bullet_t:.2f} dist={fire_dist:.1f} stable={stable} danger={danger_level:.2f} fire={fire} reason={reason} "
                    f"threats=[{'; '.join(tparts)}]"
                )
                if self._last_action_debug is not None:
                    chosen = self._last_action_debug['chosen']
                    top_scores = ', '.join(f"{n}:{s:.1f}" for n, s, _, _ in self._last_action_debug['top_scores'])
                    print(
                        f"[P3DBG {self.debug_counter:05d}] move chosen={chosen[0]} score={chosen[1]:.1f} target={chosen[2]:.1f} thrust={chosen[3]:.1f} "
                        f"ref={self._last_action_debug['reference_heading']:.1f} esc={self._last_action_debug['escape_heading']:.1f} "
                        f"good_shot={self._last_action_debug['good_shot']} top=[{top_scores}]"
                    )

        drop_mine = False
        return float(thrust), float(turn_rate), fire, drop_mine

    @property
    def name(self) -> str:
        return "HybridFuzzyController PhaseC ThreatSearch"