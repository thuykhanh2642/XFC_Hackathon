import sys
sys.path.append('.')

import json
import os
from pathlib import Path
from kesslergame import KesslerGame, GraphicsType
from HybridController2 import GAFuzzyController
from scenarios import training_set, validation_set

W_HIT = 1.50
W_ACC = 0.35
W_DEATH = 0.75
W_SURVIVAL = 0.50
W_MINES = 0.05


def score_team(team):
    mines_used = float(getattr(team, "mines_used", 0.0) or 0.0)
    score = (
        W_HIT * float(team.fraction_total_asteroids_hit)
        + W_ACC * float(team.accuracy)
        - W_DEATH * float(team.deaths)
        - W_MINES * mines_used
    )
    if int(team.deaths) == 0:
        score += W_SURVIVAL
    return score


def run_set(game, controller, scenarios, label):
    total = 0.0
    print(f"\n=== {label} ===")
    for scenario in scenarios:
        result, _ = game.run(scenario=scenario, controllers=[controller])
        t = result.teams[0]
        score = score_team(t)
        total += score
        mines_used = float(getattr(t, 'mines_used', 0.0) or 0.0)
        print(
            f"{scenario.name}: score={score:.4f}  hit={t.asteroids_hit}  "
            f"acc={t.accuracy:.3f}  deaths={t.deaths}  mines={mines_used:.0f}"
        )
    print(f"Total {label.lower()} score: {total:.4f}")
    return total


def main():
    json_path = Path(os.path.dirname(__file__), "best_solution_cmaes.json")

    if not json_path.exists():
        print(f"No CMA-ES solution found at {json_path}.")
        print("Run cmaes_train.py first.")
        return

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    genome = [float(x) for x in data["genome"]]
    print(f"Loaded CMA-ES iteration {data.get('iteration', '?')} — training score {data.get('train_score', 0.0):.4f}")
    print(f"Saved validation score: {data.get('validation_score', 0.0):.4f}")

    controller = GAFuzzyController(chromosome=genome)

    game_settings = {
        'perf_tracker': True,
        'graphics_type': GraphicsType.Tkinter,
        'realtime_multiplier': 1,
        'graphics_obj': None,
        'frequency': 30,
        'time_limit': 60,
        'competition_safe_mode': False,
    }
    game = KesslerGame(settings=game_settings)

    train_total = run_set(game, controller, training_set, "Training Set")
    val_total = run_set(game, controller, validation_set, "Validation Set")

    print("\nSummary")
    print(f"Train total: {train_total:.4f}")
    print(f"Validation total: {val_total:.4f}")


if __name__ == "__main__":
    main()
