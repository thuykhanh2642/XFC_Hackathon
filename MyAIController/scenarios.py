# ------------------------------------------------------------
# kessler-game/examples/scenarios.py
# Scenario definitions for the asteroid
# Each scenario returns with:
#  - map_size: (width, height)
#  - ship_states: list of ships
#  - asteroid_states OR num_asteroids: explicit asteroids vs random gen
#  - time_limit, ammo_limit_multiplier, stop_if_no_ammo
# ------------------------------------------------------------

import math
import random
from kesslergame import Scenario


def _mk_ship(team=1, pos=(400, 400), angle=0, mines=3):
    return {'position': pos, 'angle': angle, 'lives': 3, 'team': team, "mines_remaining": mines}

def _get_asteroid_list(scenario):

    try:
        fn = getattr(scenario, "asteroids", None)
        if callable(fn):
            lst = fn()
            if isinstance(lst, list):
                return lst
        lst = getattr(scenario, "asteroids", None)
        if isinstance(lst, list):
            return lst
    except Exception:
        pass
    return []

# ------------------------------------------------------------
# A general baseline: random asteroids
# ------------------------------------------------------------
def stock_scenario(map_size=(1000, 800)):
    random.seed(42)
    return Scenario(
        name="Stock Scenario",
        num_asteroids=15,
        ship_states=[_mk_ship(pos=(map_size[0] * 0.75, map_size[1] * 0.5), angle=180)],
        map_size=map_size,
        time_limit=60,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# ------------------------------------------------------------
# A vertical wall of big asteroids on the left, moving right
# ------------------------------------------------------------
def vertical_wall_left(map_size=(1000, 800), *,
                       count=12,
                       left_margin=10,
                       top_margin=40,
                       bottom_margin=40,
                       size_class=3,
                       wall_speed=150.0,
                       time_limit=60):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.5

    ship = {'position': (cx, cy), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    # Compute evenly spaced Y positions for the wall from top to bottom
    available_height = H - top_margin - bottom_margin
    spacing = available_height / max(1, count - 1)

    # All wall asteroids start at the same left X, different Ys
    x_pos = left_margin

    ast_states = []
    for i in range(count):
        y_pos = top_margin + i * spacing
        ast_states.append({
            'position': (x_pos, y_pos),
            'size': int(size_class),
            'angle': 0.0,
            'speed': float(wall_speed)
        })

    return Scenario(
        name="Vertical Wall Left (Big Moving Right)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# ------------------------------------------------------------
# Still work in progress, Spiral swarm: asteroids moves tangentially
# ------------------------------------------------------------
def spiral_arms(map_size=(1200, 900), *, arms=4, per_arm=10,
                 r_min_ratio=0.05, r_max_ratio=0.45,
                 base_speed=90.0, speed_step=8.0,
                 size_cycle=(3, 2, 2, 1), time_limit=75):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.5
    r_min = min(W, H) * r_min_ratio
    r_max = min(W, H) * r_max_ratio

    ship = {'position': (W * 0.75, H * 0.5), 'angle': 180, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    ast_states = []
    total = arms * per_arm
    for a in range(arms):
        arm_phase = (2.0 * math.pi / arms) * a
        for k in range(per_arm):
            t = k / max(1, (per_arm - 1))
            r = r_min + t * (r_max - r_min)
            theta = arm_phase + 4.0 * t * math.pi
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)

            # Tangential heading (theta + 90°)
            heading_deg = math.degrees(theta + math.pi / 2.0)
            spd = base_speed + speed_step * k
            size = int(size_cycle[(a * per_arm + k) % len(size_cycle)])

            ast_states.append({
                'position': (x, y),
                'size': size,
                'angle': float(heading_deg),
                'speed': float(spd),
            })


    return Scenario(
        name="Spiral Swarm",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )



# ------------------------------------------------------------
# Crossing lanes (horizontal + vertical “highways”)
# ------------------------------------------------------------
def crossing_lanes(map_size=(1200, 900), *,
                   rows=4, cols=5,
                   lane_margin=60, lane_speed=150.0, size_class=2, time_limit=70):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.5
    ship = {'position': (cx, cy), 'angle': 0, 'lives': 9, 'team': 1, 'mines_remaining': 3}

    ast_states = []

    # Horizontal lanes
    y_spacing = (H - 2 * lane_margin) / max(1, rows - 1)
    for r in range(rows):
        y = lane_margin + r * y_spacing

        # Alternate directions
        if r % 2 == 0:
            # left -> right
            x_positions = [lane_margin + i * ((W - 2 * lane_margin) / max(1, cols - 1)) for i in range(cols)]
            for x in x_positions:
                ast_states.append({'position': (x, y), 'size': size_class, 'angle': 0.0, 'speed': lane_speed})
        else:
            # right -> left
            x_positions = [W - lane_margin - i * ((W - 2 * lane_margin) / max(1, cols - 1)) for i in range(cols)]
            for x in x_positions:
                ast_states.append({'position': (x, y), 'size': size_class, 'angle': 180.0, 'speed': lane_speed})

    # Vertical lanes
    x_spacing = (W - 2 * lane_margin) / max(1, cols - 1)
    for c in range(cols):
        x = lane_margin + c * x_spacing
        if c % 2 == 0:
            # top -> down
            y_positions = [lane_margin + i * ((H - 2 * lane_margin) / max(1, rows - 1)) for i in range(rows)]
            for y in y_positions:
                ast_states.append({'position': (x, y), 'size': size_class, 'angle': 90.0, 'speed': lane_speed})
        else:
            # bottom -> up
            y_positions = [H - lane_margin - i * ((H - 2 * lane_margin) / max(1, rows - 1)) for i in range(rows)]
            for y in y_positions:
                ast_states.append({'position': (x, y), 'size': size_class, 'angle': 270.0, 'speed': lane_speed})

    return Scenario(
        name="Crossing Lanes",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# ------------------------------------------------------------
# Vertical rain clean rows fall from the top toward the ship
# ------------------------------------------------------------
def asteroid_rain(map_size=(1000, 800), *,
                  columns=10, waves=3, top_margin=20, spacing_ratio=0.8,
                  fall_speed=180.0, size_class=2, time_limit=60):
    
    W, H = map_size
    ship = {'position': (W * 0.5, H * 0.85), 'angle': 270, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    # Use only a fraction of the width so edge wrap doesn’t bunch columns too tightly
    usable_w = W * spacing_ratio
    left = (W - usable_w) * 0.5
    dx = usable_w / max(1, columns - 1)

    # Build columns for several waves
    ast_states = []
    for w in range(waves):
        y0 = top_margin - w * 70.0  # staggered starts
        for c in range(columns):
            x = left + c * dx
            ast_states.append({
                'position': (x, y0),
                'size': int(size_class),
                'angle': 270.0,
                'speed': float(fall_speed)
            })

    return Scenario(
        name="Asteroid Rain",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# ------------------------------------------------------------
# Big slow giants with packs of fast small kamikaze rocks
# ------------------------------------------------------------
def giants_with_kamikaze(map_size=(1200, 900), *,
                         giants=5, smalls_per_giant=6,
                         giant_speed=60.0, small_speed=220.0,
                         time_limit=75):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.5
    ship = {'position': (cx, cy), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    # Fixed RNG seed for reproducible sprays around each giant
    rng = random.Random(1337)
    ast_states = []

    # Giants: big class=3 moving left->right and right->left on alternating rows
    rows = max(1, giants)
    y_spacing = H / (rows + 1)
    for i in range(giants):
        y = y_spacing * (i + 1)
        angle = 0.0 if (i % 2 == 0) else 180.0
        x = W * (0.1 if angle == 0.0 else 0.9)
        ast_states.append({
            'position': (x, y),
            'size': 3,
            'angle': angle,
            'speed': float(giant_speed)
        })

        # Smalls: sprays pointing roughly toward ship
        for k in range(smalls_per_giant):
            # spawn around the giant with a little jitter
            sx = x + rng.uniform(-60, 60)
            sy = y + rng.uniform(-60, 60)
            heading = math.degrees(math.atan2(cy - sy, cx - sx)) + rng.uniform(-15, 15)
            ast_states.append({
                'position': (sx, sy),
                'size': 1,
                'angle': float(heading),
                'speed': float(small_speed)
            })

    return Scenario(
        name="Giants with Kamikaze",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# --------------------------------------
# Stationary aim range in a large arena 
# ---------------------------------------
def sniper_practice(map_size=(2000, 1400), *,
                    time_limit=120,
                    near_ring=(8, 0.25, 2),
                    mid_ring=(10, 0.40, 2),
                    far_ring=(12, 0.60, 1),
                    top_row_count=8):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.1  # ship near bottom center
    ship = {'position': (cx, cy), 'angle': 90, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    def ring_asteroids(count, radius_ratio, size):
        r = min(W, H) * radius_ratio
        return [{
            'position': (cx + r * math.cos(2 * math.pi * i / count),
                         cy + r * math.sin(2 * math.pi * i / count)),
            'size': int(size),
            'angle': 0.0,
            'speed': 0.0
        } for i in range(count)]

    ast_states = []
    ast_states += ring_asteroids(*near_ring)
    ast_states += ring_asteroids(*mid_ring)
    ast_states += ring_asteroids(*far_ring)

    # Long-range sniper line near top
    cols = max(2, int(top_row_count))
    left, right = W * 0.10, W * 0.90
    y_top = H * 0.85
    for i in range(cols):
        t = i / (cols - 1)
        x = left + t * (right - left)
        ast_states.append({
            'position': (x, y_top),
            'size': 1,
            'angle': 0.0,
            'speed': 0.0
        })

    return Scenario(
        name="Sniper Practice (Large Arena)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )


# ------------------------------------
# Donut shaped ring around the player
# ------------------------------------
def donut_ring(map_size=(1000, 800), *, count=24, radius_ratio=0.35, size_class=2, time_limit=60):

    # Takes the map size and splits it into Width and Height 
    W, H = map_size

    # This finds the center of the map: half of width(cx) and half of height(cy)
    cx, cy = W * 0.5, H * 0.5

    # This decides how far away from the center the donut ring will be
    # Using the smaller width or height to center
    r = min(W, H) * radius_ratio

    # The ships's position, angle, lives, mines, and belongs to team 1
    ship = {'position': (cx, cy), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    # Starts and empty list that will later hold all asteroid info
    ast_states = []

    # This loop will run once for each number of asteroid (count)
    for i in range(count):

        # Calculates the angle for each asteroid around the circle in a full 360 degree circle (2pi radains)
        theta = 2.0 * math.pi * (i / count)
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)

        # adds asteroid to the list, its postion, size, angle, and speed
        ast_states.append({
            'position': (x, y),
            'size': int(size_class),
            'angle': 0.0,
            'speed': 0.0,
        })

    # Returns scenario name, map size, passes the asteroid list, ship state, how long the round lasted, turns off ammo count
    return Scenario(
        name="Donut Ring",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# -----------------------------------------------------
# Closing donut: ring asteroids head toward the center
# -----------------------------------------------------
def donut_ring_closing(map_size=(1200, 900), *,
                       count=24,
                       start_radius_ratio=0.45,
                       size_class=3,
                       inward_speed=60.0,
                       time_limit=80):

    W, H = map_size
    cx, cy = W * 0.5, H * 0.5

    ship = {'position': (cx, cy), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    # Convert radius ratio to actual pixels
    r = min(W, H) * start_radius_ratio
    ast_states = []

    # Build the closing ring
    for i in range(count):
        theta = 2.0 * math.pi * (i / count)
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)

        # Heading that points directly toward the center from (x, y)
        heading_deg = math.degrees(math.atan2(cy - y, cx - x))

        ast_states.append({
            'position': (x, y),
            'size': int(size_class),
            'angle': float(heading_deg),
            'speed': float(inward_speed)
        })

    return Scenario(
        name="Donut Ring (Closing In, Large Asteroids)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )

# ----------------------------------------------------------------
# Rotating Cross is a 4 lines shaped as a cross rotating clockwise
# ----------------------------------------------------------------
def rotating_cross(map_size=(1400, 1000), *,
                            arm_density=26,
                            omega_deg_per_s=8.0, 
                            clockwise=True,
                            tip_speed_scale=0.08,
                            size_cycle=(3,2,2,1),
                            time_limit=55):
    W, H = map_size
    cx, cy = W * 0.5, H * 0.5

    # Player far left
    ship = {'position': (W * 0.10, cy), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    ast_states = []

    # Angular speed in radians/sec; sign controls direction
    omega = math.radians(omega_deg_per_s) * (-1.0 if clockwise else 1.0)

    # Compute maximum extents from center to each edge along cardinal directions
    r_right = W - cx     # center to right edge along +X
    r_left  = cx         # center to left edge  along -X
    r_up    = cy         # center to top edge   along -Y
    r_down  = H - cy     # center to bottom edge along +Y

    # Lines defined by base angle and max radius to edge in that direction
    arms = [
        (0.0,              r_right),  # right
        (math.pi,          r_left),   # left
        (math.pi / 2.0,    r_down),   # down (screen y+)
        (3.0 * math.pi/2., r_up),     # up   (screen y-)
    ]

    # Build each arm from center (r=0) to the specific edge radius
    for phi, r_max in arms:
        for i in range(arm_density + 1):
            t = i / max(1, arm_density)
            r = t * r_max

            # Position along the line
            x = cx + r * math.cos(phi)
            y = cy + r * math.sin(phi)

            # Tangent direction = line angle ± 90°
            heading = phi + (math.pi / 2.0) * (-1.0 if clockwise else 1.0)
            heading_deg = float(math.degrees(heading))

            # Tangential speed so all radis share the same angular rate
            v = abs(omega) * r

            # For the outermost tip at the edge, slow speed to keep it attached
            if i == arm_density:
                v *= float(tip_speed_scale)

            ast_states.append({
                'position': (x, y),
                'size': int(size_cycle[i % len(size_cycle)]),
                'angle': heading_deg,
                'speed': float(v)
            })

    return Scenario(
        name=f"Cross (Rotating Look, {'CW' if clockwise else 'CCW'})",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit, 
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )


def moving_maze_right(map_size=(1800, 1000), *,
                      rows=11,                 # number of horizontal bands of rocks
                      cols=22,                 # density along X
                      margin=90,               # empty buffer around the edges before placing rocks
                      speed=140.0,             # all asteroids move to the right
                      size_cycle=(2, 2, 3),    # repeating sizes to make walls feel chunky
                      waves=2.5,               # how many sinusoidal wiggles of the corridor across the map
                      amplitude_ratio=0.28,    # corridor vertical swing as a fraction of min(W,H)
                      corridor_width_ratio=0.18, # thickness of the safe path
                      time_limit=95):
    """
    'Asteroid Maze' with a snaking safe path. All rocks slide right together, so the corridor
    remains open (it's a moving tunnel). Start the ship inside the corridor near the left.

    Tweak:
    - rows/cols: overall density
    - speed: rightward drift of entire maze
    - waves, amplitude_ratio: shape of the snake path
    - corridor_width_ratio: difficulty (smaller = tighter)
    - size_cycle: mix rock sizes for visual walls
    """
    W, H = map_size
    cx, cy = W * 0.5, H * 0.5

    # Ship starts near left edge inside the corridor, pointing right
    ship = {'position': (W * 0.10, H * 0.50), 'angle': 0, 'lives': 3, 'team': 1, 'mines_remaining': 3}

    # Corridor geometry (sinusoidal centerline across X)
    A = min(W, H) * amplitude_ratio
    corridor_half = (min(W, H) * corridor_width_ratio) * 0.5

    def corridor_center_y(x):
        # sine that completes `waves` wiggles from left to right
        return H * 0.5 + A * math.sin(2.0 * math.pi * waves * (x / max(1.0, W)))

    # Build a grid of candidate asteroid positions, skip any that fall inside the corridor
    usable_w = W - 2 * margin
    usable_h = H - 2 * margin
    dx = usable_w / max(1, cols - 1)
    dy = usable_h / max(1, rows - 1)

    ast_states = []
    idx = 0
    for r in range(rows):
        y = margin + r * dy
        for c in range(cols):
            x = margin + c * dx

            # Keep a gap where the safe corridor runs
            y_c = corridor_center_y(x)
            if abs(y - y_c) <= corridor_half:
                continue  # leave space for the tunnel

            # Stagger every other row a bit for a tighter maze feel
            x_spawn = x + (dx * 0.35 if (r % 2 == 1) else 0.0)

            ast_states.append({
                'position': (x_spawn, y),
                'size': int(size_cycle[idx % len(size_cycle)]),
                'angle': 0.0,           # move to the right
                'speed': float(speed)
            })
            idx += 1

    return Scenario(
        name="Moving Maze (Rightward Tunnel)",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )
    
def four_corner(map_size=(1200, 900), *,
                cluster_size=10,
                corner_margin=80,
                size_class=2,
                speed=0.0,
                time_limit=70):


    W, H = map_size
    cx, cy = W * 0.5, H * 0.5

    ship = {
        'position': (cx, cy),
        'angle': 0,
        'lives': 3,
        'team': 1,
        'mines_remaining': 0
    }

    ast_states = []

    # Corner spawn positions
    corners = [
        (corner_margin, corner_margin),               # Top-left
        (W - corner_margin, corner_margin),           # Top-right
        (corner_margin, H - corner_margin),           # Bottom-left
        (W - corner_margin, H - corner_margin)        # Bottom-right
    ]

    # Generate asteroid clusters in each corner
    for (cxn, cyn) in corners:
        for i in range(cluster_size):
            # random spread inside each cluster
            x = cxn + random.uniform(-40, 40)
            y = cyn + random.uniform(-40, 40)

            # angle toward center
            heading = math.degrees(math.atan2(cy - y, cx - x))

            ast_states.append({
                'position': (x, y),
                'size': int(size_class),
                'angle': float(heading),
                'speed': float(speed)
            })

    return Scenario(
        name="Four Corner Assault",
        map_size=map_size,
        num_asteroids=0,
        asteroid_states=ast_states,
        ship_states=[ship],
        time_limit=time_limit,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False
    )


# Training set — used by fitness_function.py
# A diverse mix that forces the GA to generalise:
#   static targets, fast movers, closing rings, crossing traffic

training_set = [
    one_asteroid_still(),
    one_asteroid_slow_horizontal(),
    two_asteroids_still(),
    three_asteroids_still_row(),
    stock_scenario(),
    donut_ring(),
    donut_ring_closing(),
    vertical_wall_left(),
    asteroid_rain(),
    crossing_lanes(),
    giants_with_kamikaze(),
    spiral_arms(),
]

validation_set = [
    moving_maze_right(),
    four_corner(),
    rotating_cross(),
    sniper_practice(),
]
