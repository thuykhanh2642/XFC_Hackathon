# Load and run the best trained GAFuzzyController
#
# Run from project root:   python load_trained.py

import sys, os, json
sys.path.append('.')

from pathlib import Path
from kesslergame import KesslerGame, GraphicsType
from GAController import GAFuzzyController
from Scenarios.example_scenarios import training_set

#Load best solution
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
    print(f"Genome: {genome}")

#Run with graphics
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
    score = (result.teams[0].fraction_total_asteroids_hit
             + result.teams[0].accuracy)
    total_score += score
    print(
        f"Scenario '{scenario.name}' — "
        f"hit {result.teams[0].asteroids_hit} asteroids  "
        f"accuracy {result.teams[0].accuracy:.3f}  "
        f"deaths {result.teams[0].deaths}"
    )

print(f"\nAverage score across {len(training_set)} scenarios: "
      f"{total_score / len(training_set):.4f}")