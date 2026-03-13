# compare_controllers.py
# Runs multiple controllers against the same scenarios and prints a
# side-by-side comparison table.
#
# Usage:  python compare_controllers.py

import sys, os, json, time
sys.path.append('.')

from pathlib import Path
from kesslergame import KesslerGame, GraphicsType
from scenarios import training_set

# ── Import whichever controllers you want to compare ─────────────────────────
from GAController import GAFuzzyController
from example_controller_fuzzy import MyFuzzyController   # baseline


# ─────────────────────────────────────────────────────────────────────────────
#  Config — add/remove entries here to compare different controllers
# ─────────────────────────────────────────────────────────────────────────────

def load_genome(path):
    p = Path(path)
    if not p.exists():
        print(f"  [warn] {path} not found — using default chromosome")
        return None
    with p.open() as f:
        return [float(x) for x in json.load(f)["genome"]]

CONTROLLERS = [
    ("Baseline (no genome)",  MyFuzzyController,   None),
    ("GA Controller (trained)", GAFuzzyController, load_genome("best_solution.json")),
]

GAME_SETTINGS = {
    'perf_tracker':          False,
    'graphics_type':         GraphicsType.NoGraphics,
    'realtime_multiplier':   0,
    'graphics_obj':          None,
    'frequency':             30,
    'time_limit':            60,
    'competition_safe_mode': False,
}

# ─────────────────────────────────────────────────────────────────────────────

def evaluate_controller(name, controller_cls, genome):
    """Run one controller across all scenarios, return per-scenario stats."""
    ctrl = controller_cls(chromosome=genome) if genome is not None else controller_cls()
    game = KesslerGame(settings=GAME_SETTINGS)

    results = []
    for scenario in training_set:
        t0 = time.perf_counter()
        result, _ = game.run(scenario=scenario, controllers=[ctrl])
        elapsed = time.perf_counter() - t0
        t = result.teams[0]
        results.append({
            "scenario":   scenario.name,
            "hit":        t.asteroids_hit,
            "frac":       t.fraction_total_asteroids_hit,
            "accuracy":   t.accuracy,
            "deaths":     t.deaths,
            "score":      t.fraction_total_asteroids_hit * 2.0 + t.accuracy - t.deaths * 0.5,
            "time":       elapsed,
        })
    return results


def print_comparison(all_results):
    names    = [r[0] for r in all_results]
    results  = [r[1] for r in all_results]
    scenarios = [r["scenario"] for r in results[0]]

    col_w = 22

    # ── Header ───────────────────────────────────────────────────────────────
    print()
    print("=" * (18 + col_w * len(names)))
    print("  CONTROLLER COMPARISON")
    print("=" * (18 + col_w * len(names)))
    header = f"  {'Scenario':<16}"
    for n in names:
        header += f"  {n[:col_w-2]:^{col_w}}"
    print(header)
    print(f"  {'':16}  " + "  ".join([f"{'frac':>5} {'acc':>5} {'die':>3} {'score':>6}"] * len(names)))
    print("-" * (18 + col_w * len(names)))

    # ── Per-scenario rows ─────────────────────────────────────────────────────
    totals = [[0.0, 0.0, 0, 0.0] for _ in names]   # frac, acc, deaths, score

    for i, scenario_name in enumerate(scenarios):
        row = f"  {scenario_name:<16}"
        for ci, res in enumerate(results):
            r = res[i]
            row += (f"  {r['frac']:5.2f} {r['accuracy']:5.2f} "
                    f"{r['deaths']:3d} {r['score']:6.3f}")
            totals[ci][0] += r['frac']
            totals[ci][1] += r['accuracy']
            totals[ci][2] += r['deaths']
            totals[ci][3] += r['score']
        print(row)

    # ── Totals ────────────────────────────────────────────────────────────────
    print("-" * (18 + col_w * len(names)))
    n_s = len(scenarios)
    row = f"  {'AVERAGE':<16}"
    for ci in range(len(names)):
        row += (f"  {totals[ci][0]/n_s:5.2f} {totals[ci][1]/n_s:5.2f} "
                f"{totals[ci][2]/n_s:5.1f} {totals[ci][3]/n_s:6.3f}")
    print(row)

    row = f"  {'TOTAL SCORE':<16}"
    for ci in range(len(names)):
        row += f"  {'':>5} {'':>5} {'':>3} {totals[ci][3]:6.3f}"
    print(row)
    print("=" * (18 + col_w * len(names)))

    # ── Winner ───────────────────────────────────────────────────────────────
    scores = [t[3] for t in totals]
    best   = scores.index(max(scores))
    print(f"\n  Winner: {names[best]}  (total score {scores[best]:.3f})")
    if len(names) == 2:
        delta = scores[best] - scores[1 - best]
        pct   = delta / max(abs(scores[1 - best]), 1e-6) * 100
        print(f"  Margin: +{delta:.3f}  ({pct:+.1f}%)")
    print()


def main():
    all_results = []
    for name, cls, genome in CONTROLLERS:
        print(f"Evaluating: {name} ...")
        t0  = time.perf_counter()
        res = evaluate_controller(name, cls, genome)
        elapsed = time.perf_counter() - t0
        total_score = sum(r["score"] for r in res)
        print(f"  Done in {elapsed:.1f}s — total score {total_score:.3f}")
        all_results.append((name, res))

    print_comparison(all_results)


if __name__ == "__main__":
    main()