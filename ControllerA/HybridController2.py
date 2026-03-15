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
TARGET_LOCK_FRAMES = 12


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


def _safe_unit(x, y):
    n = math.sqrt(x * x + y * y)
    if n < 1e-8:
        return 0.0, 0.0
    return x / n, y / n


def _learned_dodge_thrust(ship_heading_deg, ship_pos, threat_ast, thrust_magnitude, chromosome):
    """
    Blend between:
      - perpendicular-to-velocity dodge
      - radial escape from asteroid
      - mild forward bias
    Then choose thrust sign based on ship heading.
    """
    vx, vy = threat_ast.velocity
    ax, ay = threat_ast.position_wrap

    vel_u = _safe_unit(vx, vy)
    if abs(vel_u[0]) < 1e-8 and abs(vel_u[1]) < 1e-8:
        return thrust_magnitude

    perp1 = (-vel_u[1], vel_u[0])
    perp2 = (vel_u[1], -vel_u[0])

    away_u = _safe_unit(ship_pos[0] - ax, ship_pos[1] - ay)

    h_rad = math.radians(ship_heading_deg)
    ship_fwd = (math.sin(h_rad), -math.cos(h_rad))

    # genes 32..34 control direction blend
    w_perp = 0.25 + chromosome[32] * 1.25
    w_away = chromosome[33] * 1.25
    w_fwd  = (chromosome[34] - 0.5) * 0.8

    def blended_score(perp):
        bx = w_perp * perp[0] + w_away * away_u[0] + w_fwd * ship_fwd[0]
        by = w_perp * perp[1] + w_away * away_u[1] + w_fwd * ship_fwd[1]
        bu = _safe_unit(bx, by)
        return ship_fwd[0] * bu[0] + ship_fwd[1] * bu[1]

    s1 = blended_score(perp1)
    s2 = blended_score(perp2)
    best = s1 if abs(s1) >= abs(s2) else s2
    return math.copysign(thrust_magnitude, best)


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
    """Hybrid fuzzy aiming + learned dodge gating, with target lock for stable aim."""

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
        self._target_lock = None
        self._target_lock_timer = 0

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
        return max(2.0, self.chromosome[24] * 15.0)

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

    @property
    def _aim_lock_angle_deg(self):
        return 1.0 + self.chromosome[31] * 11.0

    @property
    def _closing_gate_soft(self):
        return 0.05 + self.chromosome[35] * 0.80

    @property
    def _closing_gate_hard(self):
        return self.chromosome[36] * 0.60

    @property
    def _size_bias(self):
        return self.chromosome[37] * 0.50

    @property
    def _aim_lock_soft_suppression(self):
        return self.chromosome[38]

    def should_drop_mine(self, closest_tti, impacters, fire_ready):
        if self._mine_tti_threshold <= 0 or closest_tti is None:
            return False
        if closest_tti > self._evasion_tti_hard * 0.7:
            return False
        if closest_tti < self._mine_tti_threshold * 0.5:
            return True

        imminent_count = sum(
            1 for ast in impacters
            if getattr(ast, "tti", None) is not None and ast.tti < self._mine_tti_threshold
        )
        if imminent_count >= 2:
            return True
        if not fire_ready and closest_tti < self._mine_tti_threshold:
            return True
        return False

    def _pick_target(self, tti_cap):
        # Keep current target briefly to avoid wrap/intercept jitter.
        if self._target_lock_timer > 0 and self._target_lock is not None:
            # Validate the locked target still exists (match by position proximity).
            lock_pos = self._target_lock.position
            refreshed = None
            for ast in self.sa.ownship.asteroids:
                dx = ast.position[0] - lock_pos[0]
                dy = ast.position[1] - lock_pos[1]
                if dx * dx + dy * dy < 2500:  # within ~50px
                    refreshed = ast
                    break
            if refreshed is not None:
                self._target_lock = refreshed
                self._target_lock_timer -= 1
                return refreshed
            # Target destroyed — fall through to pick a new one.
            self._target_lock = None
            self._target_lock_timer = 0

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

        self._target_lock = best_target
        self._target_lock_timer = TARGET_LOCK_FRAMES
        return best_target

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        self.sa.update(ship_state, game_state)

        if not self.sa.ownship.asteroids:
            self._target_lock = None
            self._target_lock_timer = 0
            return 0.0, 0.0, False, False

        tti_hard = self._evasion_tti_hard
        tti_soft = self._evasion_tti_soft
        tti_cap = max(tti_soft, 6.0)

        impacters = self.sa.ownship.soonest_impact_n(3)
        dodge_threat = impacters[0] if impacters else None
        closest_tti = dodge_threat.tti if dodge_threat is not None else None

        best_target = self._pick_target(tti_cap)

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

        # --- FIX: sign sanity check ---
        # If the FIS says turn away from target when error is large, override
        # with a simple proportional controller (correct direction).
        abs_err = abs(angle_error)
        if abs_err > 8.0 and turn_rate * angle_error < 0:
            turn_rate = float(np.sign(angle_error)) * min(abs_err * 2.0, MAX_TURN)

        # --- FIX: proportional damping to prevent overshoot ---
        # Limit turn rate so the ship can't overshoot the target in one frame.
        # At 30 Hz, turning X deg/s moves X/30 deg per frame.
        # Cap turn rate so per-frame movement is at most 70% of remaining error.
        if abs_err > 0.1:
            max_rate_for_error = abs_err * 0.70 * 30.0   # 70% of error per frame × fps
            turn_rate = float(np.clip(turn_rate, -max_rate_for_error, max_rate_for_error))

        # Wider settle zone to avoid micro-oscillation.
        settle_zone = max(2.5, 0.45 * self._fire_threshold)
        if abs_err < settle_zone:
            turn_rate = 0.0

        aim_locked = (
            intercept is not None
            and abs(angle_error) < self._aim_lock_angle_deg
            and norm_bullet_t < 0.55
        )
        if aim_locked and abs(angle_error) < self._fire_threshold:
            turn_rate = 0.0

        thrust = 0.0
        if dodge_threat is not None and closest_tti is not None:
            sx, sy = self.sa.ownship.position
            ax, ay = dodge_threat.position_wrap
            vx, vy = dodge_threat.velocity

            rel_x = sx - ax
            rel_y = sy - ay
            rel_u = _safe_unit(rel_x, rel_y)
            vel_u = _safe_unit(vx, vy)

            # 1.0 means asteroid velocity points directly toward ship.
            closing_alignment = max(0.0, -(vel_u[0] * rel_u[0] + vel_u[1] * rel_u[1]))
            norm_size = float((dodge_threat.size - 1) / 3.0)

            hard_gate = self._closing_gate_hard - self._size_bias * norm_size
            soft_gate = self._closing_gate_soft - self._size_bias * norm_size

            urgent_soft_threat = closest_tti < max(0.65 * tti_hard, 0.45)
            easy_shot_freeze = (
                intercept is not None
                and norm_bullet_t < 0.45
                and abs(angle_error) < max(2.0, 0.6 * self._fire_threshold)
            )

            soft_allowed = (
                (not easy_shot_freeze) and
                ((not aim_locked) or urgent_soft_threat or (self._aim_lock_soft_suppression < 0.35))
            )
            should_hard_dodge = (
                (not easy_shot_freeze)
                and closest_tti < tti_hard
                and closing_alignment >= max(0.0, hard_gate)
            )
            should_soft_dodge = (
                closest_tti < tti_soft
                and closing_alignment >= max(0.0, soft_gate)
                and soft_allowed
            )

            if should_hard_dodge:
                thrust = _learned_dodge_thrust(
                    self.sa.ownship.heading,
                    self.sa.ownship.position,
                    dodge_threat,
                    self._thrust_hard,
                    self.chromosome,
                )
            elif should_soft_dodge:
                thrust = _learned_dodge_thrust(
                    self.sa.ownship.heading,
                    self.sa.ownship.position,
                    dodge_threat,
                    self._thrust_soft,
                    self.chromosome,
                )

        fire = bool(
            intercept is not None
            and abs(angle_error) < self._fire_threshold
            and norm_bullet_t < 0.85
        )

        # Loosen fire gate for stationary/slow targets — intercept is exact,
        # no reason to restrict bullet time.
        stationary_like_target = (
            best_target is not None
            and math.hypot(best_target.velocity[0], best_target.velocity[1]) < 20.0
        )
        if not fire and stationary_like_target and intercept is not None:
            if abs(angle_error) < self._fire_threshold:
                fire = True  # allow any range for stationary targets

        if stationary_like_target and intercept is not None and norm_bullet_t < 0.50:
            if abs(angle_error) < max(2.0, 0.5 * self._fire_threshold):
                turn_rate = 0.0
                thrust = 0.0

        if fire and abs(angle_error) < max(1.0, 0.35 * self._fire_threshold):
            turn_rate = 0.0

        # --- FLEE MODE ---
        # When overwhelmed (multiple imminent impacters), abandon aiming
        # and thrust away from the centroid of threats.
        flee_threshold = 0.3 + self.chromosome[39] * 1.2  # gene 39: flee TTI threshold
        flee_count = sum(
            1 for ast in impacters
            if ast.tti is not None and ast.tti < flee_threshold
        )
        if flee_count >= 2:
            sx, sy = self.sa.ownship.position
            cx_threat = sum(a.position_wrap[0] for a in impacters[:flee_count]) / flee_count
            cy_threat = sum(a.position_wrap[1] for a in impacters[:flee_count]) / flee_count
            away_x, away_y = _safe_unit(sx - cx_threat, sy - cy_threat)
            h_rad = math.radians(self.sa.ownship.heading)
            ship_fwd = (math.sin(h_rad), -math.cos(h_rad))
            dot = ship_fwd[0] * away_x + ship_fwd[1] * away_y
            thrust = math.copysign(self._thrust_hard, dot)
            # Don't zero turn_rate — keep aiming while fleeing

        drop_mine = bool(self.should_drop_mine(
            closest_tti=closest_tti,
            impacters=impacters,
            fire_ready=fire,
        ))

        return float(thrust), float(turn_rate), bool(fire), bool(drop_mine)

    @property
    def name(self) -> str:
        return "GA Fuzzy Controller"
