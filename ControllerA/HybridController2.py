# GA Fuzzy Controller — Hybrid Architecture
#
# CHROMOSOME LAYOUT  (50 genes, each a float in [0.0, 1.0])
#
#  FIS 1 — Aiming  (angle_error x bullet_time -> turn_rate)
#   [0]  angle_error MF breakpoint   (maps [-1,  1] universe)
#   [1]  bullet_time MF breakpoint   (maps [ 0,  1] universe)
#   [2]  turn_rate   MF breakpoint   (maps [-1,  1] universe)
#   [3-11] 9 rule consequent labels  (3x3 table)
#
#  FIS 2 — Target Priority  (norm_tti x norm_size -> priority)
#   [12] norm_tti  MF breakpoint     (maps [0, 1] universe)
#   [13] norm_size MF breakpoint     (maps [0, 1] universe)
#   [14] priority  MF breakpoint     (maps [0, 1] universe)
#   [15-23] 9 rule consequent labels
#
#  Scalar parameters
#   [24] fire_threshold_deg   -> [0,  15] deg
#   [25] n_candidates         -> bins to {3, 5, 7, 10}
#   [26] evasion_tti_hard     -> [0.3, 3.0] s  (full dodge below this TTI)
#   [27] evasion_tti_soft     -> [1.0, 6.0] s  (partial dodge below this TTI)
#   [28] thrust_hard_frac     -> [0.6, 1.0] x MAX_THRUST
#   [29] thrust_soft_frac     -> [0.1, 0.5] x MAX_THRUST
#   [30] mine_tti_threshold   -> [0.0, 3.0] s  (0 = never deploy)
#   [31-49] padding

from kesslergame import KesslerController
from typing import Dict, Tuple
from sa.sa import SA
from sa.util.helpers import trim_angle
import skfuzzy.control as ctrl
import skfuzzy as skf
import numpy as np
import math

BULLET_SPEED = 800.0
MAX_THRUST   = 480.0
MAX_TURN     = 180.0


def _compute_intercept(ship_pos, ast_pos_wrap, ast_vel):
    """Quadratic intercept solve — returns (ix, iy, tof) or None."""
    dx = ast_pos_wrap[0] - ship_pos[0]
    dy = ast_pos_wrap[1] - ship_pos[1]
    vx, vy = ast_vel[0], ast_vel[1]

    a = vx**2 + vy**2 - BULLET_SPEED**2
    b = 2.0 * (dx * vx + dy * vy)
    c = dx**2 + dy**2

    if abs(a) < 1e-6:
        if abs(b) < 1e-6:
            return None
        t = -c / b
        if t <= 0:
            return None
        return ast_pos_wrap[0] + vx * t, ast_pos_wrap[1] + vy * t, t

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
    return (
        ast_pos_wrap[0] + vx * t_best,
        ast_pos_wrap[1] + vy * t_best,
        t_best,
    )


def _geometric_dodge_thrust(ship_heading_deg, threat_ast, thrust_magnitude):
    """
    Thrust perpendicular to the asteroid's velocity vector.
    Always geometrically correct — no training required.
    """
    vx = threat_ast.velocity[0]
    vy = threat_ast.velocity[1]
    speed = math.sqrt(vx**2 + vy**2)
    if speed < 1e-6:
        return thrust_magnitude

    perp1 = (-vy / speed, vx / speed)
    perp2 = (vy / speed, -vx / speed)

    h_rad = math.radians(ship_heading_deg)
    ship_dir = (math.sin(h_rad), -math.cos(h_rad))

    dot1 = ship_dir[0] * perp1[0] + ship_dir[1] * perp1[1]
    dot2 = ship_dir[0] * perp2[0] + ship_dir[1] * perp2[1]

    best_dot = dot1 if abs(dot1) >= abs(dot2) else dot2
    return math.copysign(thrust_magnitude, best_dot)


def _build_fis_2input(c, in1_name, in1_range, bp1_idx, in1_labels,
                      in2_name, in2_range, bp2_idx, in2_labels,
                      out_name, out_range, bpo_idx, out_labels,
                      rules_start):
    def _make_mfs(obj, universe, bp, labels):
        lo, hi = universe[0], universe[-1]
        bp = float(np.clip(bp, universe[1], universe[-2]))
        obj[labels[0]] = skf.trimf(universe, [lo, lo, bp])
        obj[labels[1]] = skf.trimf(universe, [lo, bp, hi])
        obj[labels[2]] = skf.trimf(universe, [bp, hi, hi])

    u1 = np.linspace(in1_range[0], in1_range[1], 11)
    u2 = np.linspace(in2_range[0], in2_range[1], 11)
    uo = np.linspace(out_range[0], out_range[1], 11)

    in1 = ctrl.Antecedent(u1, in1_name)
    in2 = ctrl.Antecedent(u2, in2_name)
    out = ctrl.Consequent(uo, out_name)

    def _bp(gene, lo, hi):
        return gene * (hi - lo) + lo

    _make_mfs(in1, u1, _bp(c[bp1_idx], in1_range[0], in1_range[1]), in1_labels)
    _make_mfs(in2, u2, _bp(c[bp2_idx], in2_range[0], in2_range[1]), in2_labels)
    _make_mfs(out, uo, _bp(c[bpo_idx], out_range[0], out_range[1]), out_labels)

    mfs1 = [in1[l] for l in in1_labels]
    mfs2 = [in2[l] for l in in2_labels]
    out_mfs = [out[l] for l in out_labels]

    bins = np.array([0.0, 0.33334, 0.66667, 1.0])
    raw = [c[rules_start + k] for k in range(9)]
    ind = [int(min(max(int(np.digitize(v, bins, right=True)) - 1, 0), 2)) for v in raw]
    consequents = [out_mfs[i] for i in ind]

    rules = []
    k = 0
    for j in range(3):
        for i in range(3):
            rules.append(ctrl.Rule(mfs1[i] & mfs2[j], consequents[k]))
            k += 1

    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


def _safe_compute(sim, inputs, output_key, fallback):
    try:
        for k, v in inputs.items():
            sim.input[k] = float(v)
        sim.compute()
        return float(sim.output[output_key])
    except Exception:
        return fallback


class GAFuzzyController(KesslerController):
    """
    Hybrid: GA-tuned fuzzy aiming + deterministic geometric evasion.

    The GA tunes WHEN to dodge and how aggressively (thresholds + magnitudes).
    The dodge direction is always geometrically correct — perpendicular to the
    incoming asteroid's velocity. No training needed for basic survival.
    """

    _DEFAULT_CHROM = [
        0.5, 0.5, 0.5,
        0.17, 0.50, 0.83,
        0.17, 0.50, 0.83,
        0.17, 0.50, 0.83,
        0.35, 0.50, 0.50,
        0.17, 0.50, 0.83,
        0.50, 0.83, 0.83,
        0.83, 0.83, 0.50,
        0.20,
        0.50,
        0.23,
        0.50,
        0.80,
        0.40,
        0.30,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ]

    def __init__(self, chromosome=None):
        self.chromosome = chromosome if chromosome is not None else list(self._DEFAULT_CHROM)
        self.sa = SA()
        self._build_fis_systems()

    def _build_fis_systems(self):
        c = self.chromosome
        self.aiming_sim = _build_fis_2input(
            c,
            "angle_error", (-1.0, 1.0), 0, ("neg", "zero", "pos"),
            "bullet_time", (0.0, 1.0), 1, ("short", "med", "long"),
            "turn_rate", (-1.0, 1.0), 2, ("neg", "zero", "pos"),
            rules_start=3,
        )
        self.priority_sim = _build_fis_2input(
            c,
            "norm_tti", (0.0, 1.0), 12, ("imminent", "soon", "distant"),
            "norm_size", (0.0, 1.0), 13, ("small", "medium", "large"),
            "priority", (0.0, 1.0), 14, ("ignore", "low", "high"),
            rules_start=15,
        )

    @property
    def _fire_threshold(self):
        return self.chromosome[24] * 15.0

    @property
    def _n_candidates(self):
        return [3, 5, 7, 10][int(min(self.chromosome[25] * 4.0, 3))]

    @property
    def _evasion_tti_hard(self):
        return 0.3 + self.chromosome[26] * 2.7

    @property
    def _evasion_tti_soft(self):
        return 1.0 + self.chromosome[27] * 5.0

    @property
    def _thrust_hard(self):
        return (0.6 + self.chromosome[28] * 0.4) * MAX_THRUST

    @property
    def _thrust_soft(self):
        return (0.1 + self.chromosome[29] * 0.4) * MAX_THRUST

    @property
    def _mine_tti_threshold(self):
        return self.chromosome[30] * 3.0

    def should_drop_mine(self, closest_tti, impacters, fire_ready):
        """More conservative mine use: only spend mines when danger is real."""
        if self._mine_tti_threshold <= 0 or closest_tti is None:
            return False

        if closest_tti > self._evasion_tti_hard * 0.7:
            return False

        if closest_tti < self._mine_tti_threshold * 0.5:
            return True

        imminent_count = sum(
            1
            for ast in impacters
            if getattr(ast, "tti", None) is not None and ast.tti < self._mine_tti_threshold
        )
        if imminent_count >= 2:
            return True

        if not fire_ready and closest_tti < self._mine_tti_threshold:
            return True

        return False

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        self.sa.update(ship_state, game_state)

        if not self.sa.ownship.asteroids:
            return 0.0, 0.0, False, False

        tti_hard = self._evasion_tti_hard
        tti_soft = self._evasion_tti_soft
        tti_cap = max(tti_soft, 6.0)

        impacters = self.sa.ownship.soonest_impact_n(3)
        dodge_threat = impacters[0] if impacters else None
        closest_tti = dodge_threat.tti if dodge_threat is not None else None

        n = min(self._n_candidates, len(self.sa.ownship.asteroids))
        candidates = self.sa.ownship.nearest_n_wrap(n)
        best_target = candidates[0]
        best_priority = -1.0

        for ast in candidates:
            raw_tti = ast.tti if ast.tti is not None else tti_cap
            norm_tti = float(np.clip(raw_tti / tti_cap, 0.0, 1.0))
            norm_size = float((ast.size - 1) / 3.0)
            p = _safe_compute(
                self.priority_sim,
                {"norm_tti": norm_tti, "norm_size": norm_size},
                "priority",
                fallback=0.5,
            )
            if p > best_priority:
                best_priority = p
                best_target = ast

        intercept = _compute_intercept(
            self.sa.ownship.position,
            best_target.position_wrap,
            best_target.velocity,
        )
        if intercept is not None:
            ix, iy, bullet_t = intercept
            dx = ix - self.sa.ownship.position[0]
            dy = iy - self.sa.ownship.position[1]
            aim_bearing = math.degrees(math.atan2(-dx, dy))
            norm_bullet_t = float(np.clip(bullet_t / 5.0, 0.0, 1.0))
        else:
            aim_bearing = best_target.bearing_wrap
            norm_bullet_t = 0.5

        angle_error = trim_angle(aim_bearing - self.sa.ownship.heading)
        norm_angle_error = float(np.clip(angle_error / 180.0, -1.0, 1.0))

        norm_turn = _safe_compute(
            self.aiming_sim,
            {"angle_error": norm_angle_error, "bullet_time": norm_bullet_t},
            "turn_rate",
            fallback=float(np.sign(norm_angle_error)) * 0.5,
        )
        turn_rate = float(np.clip(norm_turn * MAX_TURN, -MAX_TURN, MAX_TURN))

        thrust = 0.0
        if dodge_threat is not None and closest_tti is not None:
            if closest_tti < tti_hard:
                thrust = _geometric_dodge_thrust(
                    self.sa.ownship.heading, dodge_threat, self._thrust_hard)
            elif closest_tti < tti_soft:
                thrust = _geometric_dodge_thrust(
                    self.sa.ownship.heading, dodge_threat, self._thrust_soft)

        fire = (
            intercept is not None
            and abs(angle_error) < self._fire_threshold
            and norm_bullet_t < 0.85
        )

        drop_mine = self.should_drop_mine(
            closest_tti=closest_tti,
            impacters=impacters,
            fire_ready=fire,
        )

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "GA Fuzzy Controller"
