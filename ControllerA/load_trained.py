# Load and run the best trained GAFuzzyController
#type: ignore
# Run from project root:   python load_trained.py

import sys, os, json
sys.path.append('.')

from pathlib import Path
from kesslergame import KesslerGame, GraphicsType
from HybridController2 import GAFuzzyController
from scenarios import training_set


W_HIT = 1.50
W_ACC = 0.35
W_DEATH = 0.75
W_SURVIVAL = 0.50
W_MINES = 0.05


def score_team(team):
    """Match training/evaluation score to training_script.py."""
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


# Load best solution
json_path = Path(os.path.dirname(__file__), "best_solution.json")

if not json_path.exists():
    print(f"No trained solution found at {json_path}.")
    print("Run training_script.py first, or the controller will use its built-in defaults.")
    genome = None
else:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    genome = [float(x) for x in data["genome"]]
    print(f"Loaded generation {data['generation']} — training fitness {data['fitness'][0]:.4f}")
    if "validation_score" in data and data["validation_score"] is not None:
        print(f"Saved validation score: {float(data['validation_score']):.4f}")
    print(f"Genome: {genome}")

# Run with graphics
controller = GAFuzzyController(chromosome=genome)

game_settings = {
    'perf_tracker':         True,
    'graphics_type':        GraphicsType.Tkinter,
    'realtime_multiplier':  1,
    'graphics_obj':         None,
    'frequency':            30,
    'time_limit':           60,
    'competition_safe_mode': False,
}
game = KesslerGame(settings=game_settings)

total_score = 0.0
for scenario in training_set:
    result, _ = game.run(scenario=scenario, controllers=[controller])
    team = result.teams[0]
    mines_used = float(getattr(team, "mines_used", 0.0) or 0.0)
    score = score_team(team)
    total_score += score
    print(
        f"Scenario '{scenario.name}' — "
        f"score {score:.4f}  "
        f"hit {team.asteroids_hit} asteroids  "
        f"accuracy {team.accuracy:.3f}  "
        f"deaths {team.deaths}  "
        f"mines {mines_used:.0f}"
    )

print(
    f"\nAverage training-aligned score across {len(training_set)} scenarios: "
    f"{total_score / len(training_set):.4f}"
)
