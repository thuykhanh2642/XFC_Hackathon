# Fuzzy GA Controller
#
# Key improvements over the baseline controllers:
#   1. LEAD AIMING   - analytically computes the intercept point so bullets
#                      meet the asteroid rather than chasing where it WAS.
#   2. THREAT-BASED TARGET SELECTION - FIS scores every nearby asteroid on
#                      TTI + size; ship targets the most dangerous, not nearest.
#   3. EVASION THRUST - a separate FIS decides when / how hard to thrust
#                      based on incoming threat TTI and approach geometry.
#   4. FULLY GA-TUNABLE - membership-function breakpoints, rule tables, fire
#                      threshold, evasion sensitivity, and thrust scale are
#                      all encoded in the 50-gene chromosome.
#
# 
# CHROMOSOME LAYOUT  (50 genes, each a float in [0.0, 1.0])
# 
#  FIS 1 — Aiming  (angle_error × bullet_time → turn_rate)
#   [0]  angle_error MF breakpoint   (maps [-1, 1] universe)
#   [1]  bullet_time MF breakpoint   (maps [ 0, 1] universe)
#   [2]  turn_rate   MF breakpoint   (maps [-1, 1] universe)
#   [3–11] 9 rule consequent labels  (3 × 3 rule table)
#
#  FIS 2 — Evasion Thrust  (min_tti × approach_angle → thrust)
#   [12] min_tti       MF breakpoint (maps [ 0, 1] universe)
#   [13] approach_angle MF breakpoint(maps [ 0, 1] universe)
#   [14] thrust         MF breakpoint(maps [-1, 1] universe)
#   [15–23] 9 rule consequent labels
#
#  FIS 3 — Target Priority  (norm_tti × norm_size → priority)
#   [24] norm_tti  MF breakpoint     (maps [ 0, 1] universe)
#   [25] norm_size MF breakpoint     (maps [ 0, 1] universe)
#   [26] priority  MF breakpoint     (maps [ 0, 1] universe)
#   [27–35] 9 rule consequent labels
#
#  Scalar parameters
#   [36] fire_threshold_deg    → linear map to [0°, 15°]
#   [37] evasion_tti_cap       → linear map to [2 s, 12 s]
#   [38] n_candidates          → bins to {3, 5, 7, 10}
#   [39] thrust_scale          → linear map to [0.2, 1.0]
#   [40] mine_tti_threshold    → linear map to [0.0, 3.0] s  (0 = never)
#   [41–49] padding / future use
# 

from kesslergame import KesslerController
from typing import Dict, Tuple
from sa.sa import SA
from sa.util.helpers import trim_angle
import skfuzzy.control as ctrl
import skfuzzy as skf
import numpy as np
import math

BULLET_SPEED = 800.0   # px/s  (approximate Kessler bullet speed)
MAX_THRUST   = 480.0   # px/s² (Kessler thrust limit)
MAX_TURN     = 180.0   # deg/s (Kessler turn-rate limit)


# 
#  Utility
# 

def _compute_intercept(ship_pos, ast_pos_wrap, ast_vel):
    """
    Solve for the intercept point where a bullet fired from *ship_pos* will
    collide with an asteroid at *ast_pos_wrap* moving with *ast_vel*.

    Approach: set up the quadratic
        ||(ast_pos + vel·t) - ship_pos||² = (BULLET_SPEED·t)²
    and take the smallest positive root.

    Returns (ix, iy, t) on success, or None if no valid solution exists.
    """
    dx = ast_pos_wrap[0] - ship_pos[0]
    dy = ast_pos_wrap[1] - ship_pos[1]
    vx, vy = ast_vel[0], ast_vel[1]

    a = vx**2 + vy**2 - BULLET_SPEED**2
    b = 2.0 * (dx * vx + dy * vy)
    c = dx**2 + dy**2

    if abs(a) < 1e-6:                       # asteroid speed ≈ bullet speed
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
    return (ast_pos_wrap[0] + vx * t_best,
            ast_pos_wrap[1] + vy * t_best,
            t_best)


def _build_fis_2input(
        c,                          # full chromosome (list of floats)
        in1_name, in1_range, bp1_idx,   in1_labels,
        in2_name, in2_range, bp2_idx,   in2_labels,
        out_name, out_range, bpo_idx,   out_labels,
        rules_start,
        default_rules=None):
    """
    Construct a 2-input, 1-output Mamdani FIS with Ruspini-partitioned
    triangular membership functions.

    Each input has 3 triangular MFs sharing one interior breakpoint.
    All breakpoints are decoded from the chromosome as a linear map from
    [0, 1] onto the respective universe.

    The 9 rule consequents (3×3 table) are encoded as floats that are
    binned into {0, 1, 2} to index the three output MF labels.

    Returns a ctrl.ControlSystemSimulation ready to .compute().
    """
    def _make_mfs(ant_or_con, universe, bp, labels):
        u_lo, u_hi = universe[0], universe[-1]
        bp_clamped = float(np.clip(bp, universe[1], universe[-2]))
        ant_or_con[labels[0]] = skf.trimf(universe, [u_lo, u_lo, bp_clamped])
        ant_or_con[labels[1]] = skf.trimf(universe, [u_lo, bp_clamped, u_hi])
        ant_or_con[labels[2]] = skf.trimf(universe, [bp_clamped, u_hi, u_hi])

    def _decode_bp(gene_val, lo, hi):
        return gene_val * (hi - lo) + lo

    # ── build antecedents ──────────────────────────────────────────────────
    u1 = np.linspace(in1_range[0], in1_range[1], 11)
    u2 = np.linspace(in2_range[0], in2_range[1], 11)
    uo = np.linspace(out_range[0],  out_range[1],  11)

    in1 = ctrl.Antecedent(u1, in1_name)
    in2 = ctrl.Antecedent(u2, in2_name)
    out = ctrl.Consequent(uo, out_name)

    bp1 = _decode_bp(c[bp1_idx], in1_range[0], in1_range[1])
    bp2 = _decode_bp(c[bp2_idx], in2_range[0], in2_range[1])
    bpo = _decode_bp(c[bpo_idx], out_range[0],  out_range[1])

    _make_mfs(in1, u1, bp1, in1_labels)
    _make_mfs(in2, u2, bp2, in2_labels)
    _make_mfs(out, uo, bpo, out_labels)

    mfs1 = [in1[l] for l in in1_labels]
    mfs2 = [in2[l] for l in in2_labels]
    out_mfs = [out[l] for l in out_labels]

    # ── rule consequents ──────────────────────────────────────────────────
    bins = np.array([0.0, 0.33334, 0.66667, 1.0])
    if default_rules is not None and c is None:
        raw = default_rules
    else:
        raw = [c[rules_start + k] for k in range(9)]

    ind = np.digitize(raw, bins, right=True) - 1
    ind = [int(min(max(i, 0), 2)) for i in ind]
    consequents = [out_mfs[i] for i in ind]

    # ── assemble rules (outer loop = input2, inner = input1) ─────────────
    rules = []
    k = 0
    for j in range(3):
        for i in range(3):
            rules.append(ctrl.Rule(mfs1[i] & mfs2[j], consequents[k]))
            k += 1

    fis = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(fis)


def _safe_compute(sim, inputs: dict, output_key: str, fallback: float) -> float:
    """Feed inputs into a FIS sim, compute, and return output; use fallback on error."""
    try:
        for k, v in inputs.items():
            sim.input[k] = float(v)
        sim.compute()
        return float(sim.output[output_key])
    except Exception:
        return fallback


# 
#  Controller
# 

class GAFuzzyController(KesslerController):
    """
    GA-trainable fuzzy controller with lead aiming, threat-based target
    selection, and evasion thrust.
    """

    # Default chromosome: sensible hand-coded starting point
    _DEFAULT_CHROM = [
        # FIS1 breakpoints + rules (aiming)
        0.5,  0.5,  0.5,                       # genes 0-2: MF breakpoints
        0.17, 0.50, 0.83,                       # rules: (neg,short)→neg,  (zero,short)→zero, (pos,short)→pos
        0.17, 0.50, 0.83,                       # rules: (neg,med)→neg,   (zero,med)→zero,  (pos,med)→pos
        0.17, 0.50, 0.83,                       # rules: (neg,long)→neg,  (zero,long)→zero, (pos,long)→pos
        # FIS2 breakpoints + rules (evasion)
        0.30, 0.50, 0.80,                       # genes 12-14: MF breakpoints (low TTI→thrust)
        0.83, 0.83, 0.50,                       # rules: (imm,head)→fwd,  (soon,head)→fwd,  (far,head)→none
        0.83, 0.50, 0.50,                       # rules: (imm,obl)→fwd,   (soon,obl)→none,  (far,obl)→none
        0.50, 0.17, 0.17,                       # rules: (imm,side)→none, (soon,side)→back, (far,side)→back
        # FIS3 breakpoints + rules (priority)
        0.35, 0.50, 0.50,                       # genes 24-26: MF breakpoints
        0.17, 0.50, 0.83,                       # rules: (imm,small)→low, (soon,small)→med, (far,small)→high
        0.50, 0.83, 0.83,                       # rules: (imm,med)→med,   (soon,med)→high,  (far,med)→high
        0.83, 0.83, 0.50,                       # rules: (imm,large)→high,(soon,large)→high,(far,large)→med
        # Scalar params
        0.20,   # gene 36 → fire threshold ≈ 3°
        0.50,   # gene 37 → evasion TTI cap ≈ 7 s
        0.50,   # gene 38 → n_candidates → 5
        0.80,   # gene 39 → thrust scale 0.84
        0.30,   # gene 40 → mine TTI threshold ≈ 0.9 s
        # padding
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ]

    def __init__(self, chromosome=None):
        self.chromosome = chromosome if chromosome is not None else self._DEFAULT_CHROM
        self.sa = SA()
        self._build_fis_systems()

    # ── FIS construction ─────────────────────────────────────────────────────

    def _build_fis_systems(self):
        c = self.chromosome

        # FIS 1 – Aiming
        self.aiming_sim = _build_fis_2input(
            c,
            "angle_error", (-1.0,  1.0), 0,  ("neg",      "zero",   "pos"),
            "bullet_time", ( 0.0,  1.0), 1,  ("short",    "med",    "long"),
            "turn_rate",   (-1.0,  1.0), 2,  ("neg",      "zero",   "pos"),
            rules_start=3,
        )

        # FIS 2 – Evasion Thrust
        self.thrust_sim = _build_fis_2input(
            c,
            "min_tti",       (0.0, 1.0), 12, ("imminent", "soon",  "far"),
            "approach_angle",(0.0, 1.0), 13, ("head_on",  "oblique","side"),
            "thrust",        (-1.0,1.0), 14, ("backward", "none",   "forward"),
            rules_start=15,
        )

        # FIS 3 – Target Priority
        self.priority_sim = _build_fis_2input(
            c,
            "norm_tti",  (0.0, 1.0), 24, ("imminent", "soon",   "distant"),
            "norm_size", (0.0, 1.0), 25, ("small",    "medium", "large"),
            "priority",  (0.0, 1.0), 26, ("ignore",   "low",    "high"),
            rules_start=27,
        )

    # ── Scalar parameter decoders ────────────────────────────────────────────

    @property
    def _fire_threshold(self) -> float:
        """Angle error (degrees) below which the ship fires."""
        return self.chromosome[36] * 15.0           # [0°, 15°]

    @property
    def _evasion_tti_cap(self) -> float:
        """TTI cap used when normalising min_tti for the thrust FIS."""
        return 2.0 + self.chromosome[37] * 10.0    # [2 s, 12 s]

    @property
    def _n_candidates(self) -> int:
        """Number of nearby asteroids evaluated for target priority."""
        return [3, 5, 7, 10][int(min(self.chromosome[38] * 4.0, 3))]

    @property
    def _thrust_scale(self) -> float:
        """Fraction of MAX_THRUST to use."""
        return 0.2 + self.chromosome[39] * 0.8     # [0.2, 1.0]

    @property
    def _mine_tti_threshold(self) -> float:
        """Deploy mine if best-threat TTI < this value (0 = never deploy)."""
        return self.chromosome[40] * 3.0            # [0 s, 3 s]

    # ── Target selection ─────────────────────────────────────────────────────

    def _priority(self, tti_normed: float, size_normed: float) -> float:
        return _safe_compute(
            self.priority_sim,
            {"norm_tti": np.clip(tti_normed, 0.0, 1.0),
             "norm_size": np.clip(size_normed, 0.0, 1.0)},
            "priority",
            fallback=0.5,
        )

    # ── Main loop ────────────────────────────────────────────────────────────

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        self.sa.update(ship_state, game_state)

        if not self.sa.ownship.asteroids:
            return 0.0, 0.0, False, False

        tti_cap = self._evasion_tti_cap

        # ── 1. Target selection ──────────────────────────────────────────
        n = min(self._n_candidates, len(self.sa.ownship.asteroids))
        candidates = self.sa.ownship.nearest_n_wrap(n)

        best_target   = candidates[0]
        best_priority = -1.0

        for ast in candidates:
            raw_tti    = ast.tti if ast.tti is not None else tti_cap
            norm_tti   = float(np.clip(raw_tti / tti_cap, 0.0, 1.0))
            norm_size  = float((ast.size - 1) / 3.0)           # size 1-4 → 0-1
            p = self._priority(norm_tti, norm_size)
            if p > best_priority:
                best_priority = p
                best_target   = ast

        # ── 2. Lead aiming (intercept calculation) ───────────────────────
        intercept = _compute_intercept(
            self.sa.ownship.position,
            best_target.position_wrap,
            best_target.velocity,
        )

        if intercept is not None:
            ix, iy, bullet_t = intercept
            dx = ix - self.sa.ownship.position[0]
            dy = iy - self.sa.ownship.position[1]
            aim_bearing   = math.degrees(math.atan2(-dx, dy))  # North=0 convention
            norm_bullet_t = float(np.clip(bullet_t / 5.0, 0.0, 1.0))
        else:
            aim_bearing   = best_target.bearing_wrap
            norm_bullet_t = 0.5
            bullet_t      = None

        angle_error      = trim_angle(aim_bearing - self.sa.ownship.heading)
        norm_angle_error = float(np.clip(angle_error / 180.0, -1.0, 1.0))

        # ── 3. Aiming FIS → turn rate ────────────────────────────────────
        norm_turn = _safe_compute(
            self.aiming_sim,
            {"angle_error": norm_angle_error, "bullet_time": norm_bullet_t},
            "turn_rate",
            fallback=np.sign(norm_angle_error) * 0.5,
        )
        turn_rate = float(np.clip(norm_turn * MAX_TURN, -MAX_TURN, MAX_TURN))

        # ── 4. Evasion thrust FIS ────────────────────────────────────────
        thrust = 0.0
        impacters = self.sa.ownship.soonest_impact_n(3)

        if impacters:
            threat    = impacters[0]
            raw_tti   = threat.tti if threat.tti is not None else tti_cap
            norm_tti  = float(np.clip(raw_tti / tti_cap, 0.0, 1.0))

            # approach_angle: how head-on is the asteroid?
            # 0 = coming straight at us,  1 = passing nearly perpendicular
            ast_heading   = threat.heading
            to_ship_angle = trim_angle(threat.bearing_wrap + 180.0)
            approach_diff = abs(trim_angle(ast_heading - to_ship_angle)) / 180.0
            approach_angle = float(np.clip(approach_diff, 0.0, 1.0))

            norm_thrust = _safe_compute(
                self.thrust_sim,
                {"min_tti": norm_tti, "approach_angle": approach_angle},
                "thrust",
                fallback=0.0,
            )
            # map [-1, 1] → [-MAX_THRUST, MAX_THRUST], scaled by GA parameter
            thrust = float(np.clip(
                norm_thrust * MAX_THRUST * self._thrust_scale,
                -MAX_THRUST, MAX_THRUST,
            ))

        # ── 5. Fire decision ─────────────────────────────────────────────
        fire = (abs(angle_error) < self._fire_threshold) and (intercept is not None)

        # ── 6. Mine deployment ───────────────────────────────────────────
        mine_thresh = self._mine_tti_threshold
        drop_mine   = False
        if mine_thresh > 0 and impacters:
            t_impact = impacters[0].tti
            if t_impact is not None and t_impact < mine_thresh:
                drop_mine = True

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "GAFuzzy Controller"