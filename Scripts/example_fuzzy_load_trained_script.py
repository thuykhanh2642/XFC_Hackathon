# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import sys
import os

sys.path.append('.')

from kesslergame import KesslerController, KesslerGame, GraphicsType, TrainerEnvironment
from MyAIController.example_controller_fuzzy import MyFuzzyController
# from MyAIController.example_controller_fuzzy2 import MyFuzzyController2
from Scenarios.example_scenarios import training_set

import json
from pathlib import Path

# Path to the saved JSON file
# json_path = Path("solution_history/best_solution_gen_0001.json")
json_path = Path(os.path.dirname(__file__), "best_solution.json")

# Load JSON
with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Extract fields with appropriate Python types
generation = int(data["generation"])
fitness = [float(x) for x in data["fitness"]]   # usually a 1-element list in your script
genome = [float(x) for x in data["genome"]]     # plain Python list of floats
n_genes = int(data["n_genes"])
population_size = int(data["population_size"])
cxpb = float(data["cxpb"])
mutpb = float(data["mutpb"])

# Optional convenience variable if you want scalar fitness
best_fitness = fitness[0] if len(fitness) == 1 else fitness

# print best solution values - note this is fitness from your fitness function so depends on your training portfolio
print("Generation:", generation)
print("Best fitness:", best_fitness)
print("Genome length:", n_genes)
print("Genome:", genome)


from example_fitness_function import exampleFitness

# orginal fitness function from DEAP onemax example
# def evalOneMax(individual):
#     # print("test", individual[:2])
#     # print(type(individual))
#     return sum(individual),

# evaluate the genome/chromosome
controller = MyFuzzyController(chromosome=genome)
total_score = 0
game_settings = {'perf_tracker': False,
                 'graphics_type': GraphicsType.Tkinter,
                 'realtime_multiplier': 1,
                 'graphics_obj': None,
                 'frequency': 30,
                 'time_limit': 30,
                 'competition_safe_mode': False}
game = KesslerGame(settings=game_settings)  # Instantiate the game object
# game = TrainerEnvironment(settings=game_settings)  # Kessler default settings for training - has some defaults that override what we want though

for scenario in training_set:
    result, _ = game.run(scenario=scenario, controllers=[controller])
    # scores = [team.asteroids_hit for team in result.teams]
    # total_score += scores[0]
    score = result.teams[0].fraction_total_asteroids_hit + result.teams[0].accuracy
    total_score += score

print("Average score: ", total_score/len(training_set))
