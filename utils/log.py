import os
import datetime
import json

class Logger:
    def __init__(self, log_dir, run_name, config_dict):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{run_name}.log")
        
        # Initialize file and write header/config
        with open(self.log_path, 'w') as f:
            f.write(f"=== TRAINING LOG: {run_name} ===\n")
            f.write(f"Date: {datetime.datetime.now()}\n")
            f.write("=== CONFIGURATION ===\n")
            # Pretty print config dictionary
            f.write(json.dumps(config_dict, indent=4, default=str)) 
            f.write("\n=====================\n\n")
            
    def log(self, message):
        print(message) # Print to console
        with open(self.log_path, 'a') as f:
            f.write(message + "\n") # Write to file