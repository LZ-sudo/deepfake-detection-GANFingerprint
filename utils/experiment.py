# utils/experiment.py
import os
import json
# import time
from datetime import datetime

class ExperimentTracker:
    def __init__(self, experiment_name, base_dir="experiments"):
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.config_file = os.path.join(self.experiment_dir, "config.json")
        self.results_file = os.path.join(self.experiment_dir, "results.json")
        self.log_file = os.path.join(self.experiment_dir, "log.txt")
        
    def save_config(self, config):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=4)
            
    def log(self, message):
        """Log a message to the log file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(message)
        
    def save_results(self, results):
        """Save results to JSON file"""
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=4)