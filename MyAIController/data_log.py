"""
For logging game data to CSV files.

Data structure:
Each row in the CSV file contains feature values followed by target action values.
Features: dist, ttc, heading_err, approach_speed, ammo, mines, threat_density, threat_angle
Targets: thrust, turn_rate

Example row:
{
    "dist": 100.0,
    "ttc": 5.0,
    "heading_err": 0.2,
    "approach_speed": -1.5,
    "ammo": 3,
    "mines": 1,
    "threat_density": 0.7,
    "threat_angle": 45.0,
    "thrust": 0.8,
    "turn_rate": -0.1
}
"""
import csv, os
from pathlib import Path


FEATURES = [
    "dist",
    "ttc",
    "heading_err",
    "approach_speed",
    "ammo",
    "mines",
    "threat_density",
    "threat_angle",
]

TARGET = ["thrust", "turn_rate"]

class Logger:
    
    def __init__(self, filepath, features, targets):
        self.filepath = filepath
        self.features = features
        self.targets = targets
        self.fieldnames = features + targets
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        file_exists = os.path.exists(filepath)

        self.file = open(self.filepath, mode='a', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not file_exists or os.path.getsize(filepath) == 0:
            self.writer.writeheader()
            
    def log(self, ctx, actions):
        row = {name: ctx.get(name, "") for name in self.features}
        for name, value in zip(self.targets, actions):
            row[name] = value
        self.writer.writerow(row)
        self.file.flush()
        
    def close(self):
        self.file.close()