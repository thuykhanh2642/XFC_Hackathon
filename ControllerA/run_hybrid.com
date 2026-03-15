# run_hybrid_v2.py
# Test hybrid_fuzzy_v2 on all scenarios with graphics + scoring
#
# Usage:  python run_hybrid_v2.py
#   Add --no-graphics for headless mode
#   Add --scenario "Name" to run a single scenario

import sys
sys.path.append('.')

from kesslergame import KesslerGame, GraphicsType
from hybrid_fuzzy_v2 import hybrid_controller
from scenarios import (
    one_asteroid_still, one_asteroid_slow_horizontal,
    two_asteroids_still, three_asteroids_still_row,
    stock_scenario, donut_ring, donut_ring_closing,
    vertical_wall_left, asteroid_rain, crossing_lanes,
    giants_with_kamikaze, spiral_arms, four_corner,
    rotating_cross, moving_maze_right,
)

# ── Scoring (matches training_script.py) ──────────────────────────────
W_HIT      = 1.20
W_ACC      = 0.40
W_DEATH    = 0.45
W_SURVIVAL = 0.75
W_MINES    = 0.05
W_TIME     = 0.35

def score_team(team, time_limit):
    mines = float(getattr(team, "mines_used", 0.0) or 0.0)
    surv = 0.0
    for attr in ("time_alive", "survival_time", "time_survived", "sim_time_alive", "seconds_alive"):
        v = getattr(team, attr, None)
        if v is not None:
            surv = max(0.0, min(float(v) / time_limit, 1.0))
            break
    s = (W_HIT * float(team.fraction_total_asteroids_hit)
         + W_ACC * float(team.accuracy)
         - W_DEATH * float(team.deaths)
         - W_MINES * mines
         + W_TIME * surv)
    if int(team.deaths) == 0:
        s += W_SURVIVAL
    return s

# ── All scenarios ─────────────────────────────────────────────────────
ALL_SCENARIOS = [
    # Easy
    one_asteroid_still(),
    one_asteroid_slow_horizontal(),
    two_asteroids_still(),
    three_asteroids_still_row(),
    # Medium
    stock_scenario(),
    donut_ring(),
    donut_ring_closing(),
    vertical_wall_left(),
    asteroid_rain(),
    crossing_lanes(),
    giants_with_kamikaze(),
    spiral_arms(),
    four_corner(),
    # Hard / validation
    rotating_cross(),
    moving_maze_right(),
]

# ── Main ──────────────────────────────────────────────────────────────
def main():
    use_graphics = "--no-graphics" not in sys.argv
    single = None
    if "--scenario" in sys.argv:
        idx = sys.argv.index("--scenario")
        if idx + 1 < len(sys.argv):
            single = sys.argv[idx + 1]

    time_limit = 60

    game_settings = {
        'perf_tracker':         True,
        'graphics_type':        GraphicsType.Tkinter if use_graphics else GraphicsType.NoGraphics,
        'realtime_multiplier':  1 if use_graphics else 0,
        'graphics_obj':         None,
        'frequency':            30,
        'time_limit':           time_limit,
        'competition_safe_mode': False,
    }

    scenarios = ALL_SCENARIOS
    if single:
        scenarios = [s for s in scenarios if single.lower() in s.name.lower()]
        if not scenarios:
            print(f"No scenario matching '{single}'. Available:")
            for s in ALL_SCENARIOS:
                print(f"  - {s.name}")
            return

    controller = hybrid_controller()
    game = KesslerGame(settings=game_settings)

    results = []
    total_score = 0.0

    print(f"\n{'='*80}")
    print(f"  HYBRID FUZZY V2 — {'Graphics ON' if use_graphics else 'Headless'}")
    print(f"  Scenarios: {len(scenarios)}  |  Time limit: {time_limit}s")
    print(f"{'='*80}\n")

    for scenario in scenarios:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        team = result.teams[0]
        s = score_team(team, time_limit)
        mines = float(getattr(team, "mines_used", 0.0) or 0.0)

        results.append({
            'name': scenario.name,
            'score': s,
            'hit_frac': float(team.fraction_total_asteroids_hit),
            'asteroids_hit': team.asteroids_hit,
            'accuracy': float(team.accuracy),
            'deaths': int(team.deaths),
            'mines': mines,
        })
        total_score += s

        print(f"  {scenario.name:<45s}  score={s:>7.4f}  "
              f"hit={team.asteroids_hit:>3}  acc={team.accuracy:.3f}  "
              f"deaths={team.deaths}  mines={mines:.0f}")

    avg = total_score / len(results) if results else 0
    print(f"\n{'─'*80}")
    print(f"  TOTAL: {total_score:.4f}  |  AVERAGE: {avg:.4f}  |  Scenarios: {len(results)}")
    print(f"{'─'*80}")

    # Summary table
    print(f"\n  {'Scenario':<45s} {'Score':>7s} {'Hit%':>6s} {'Acc':>6s} {'Deaths':>6s}")
    print(f"  {'─'*45} {'─'*7} {'─'*6} {'─'*6} {'─'*6}")
    for r in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"  {r['name']:<45s} {r['score']:>7.4f} {r['hit_frac']*100:>5.1f}% {r['accuracy']:>6.3f} {r['deaths']:>6d}")


if __name__ == "__main__":
    main()