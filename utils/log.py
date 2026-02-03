import os
import datetime
import json


class Logger:
    def __init__(self, log_dir, run_name, config_dict, resume=False):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{run_name}.log")
        
        # Determine mode: 'a' for append if resuming, 'w' for fresh start
        mode = 'a' if resume and os.path.exists(self.log_path) else 'w'
        
        with open(self.log_path, mode) as f:
            if mode == 'a':
                f.write(f"\n\n{'='*30}\n")
                f.write(f"RESUMING TRAINING: {datetime.datetime.now()}\n")
                f.write(f"{'='*30}\n\n")
            else:
                f.write(f"=== TRAINING LOG: {run_name} ===\n")
                f.write(f"Date: {datetime.datetime.now()}\n")
                f.write("=== CONFIGURATION ===\n")
                f.write(json.dumps(config_dict, indent=4, default=str)) 
                f.write("\n=====================\n\n")
            
    def log(self, message):
        # Optional: format timestamp for each log line
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        print(log_message) # Print to console
        with open(self.log_path, 'a') as f:
            f.write(log_message + "\n") # Write to file