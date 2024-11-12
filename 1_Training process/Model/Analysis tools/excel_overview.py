import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set the path to the directory containing the runs
results_dir = r"Production\ray_results\train_model_2024-11-02_13-27-01"

# Initialize an empty list to store results
data = []

# Loop through each subdirectory (each run) and collect results
for run_dir in os.listdir(results_dir):
    run_path = os.path.join(results_dir, run_dir)
    results_file = os.path.join(run_path, 'result.json')
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            try:
                
                results = json.load(f)
                # Extract metrics
                metrics = {
                    'accuracy': results.get('accuracy'),
                    'precision': results.get('precision'),
                    'recall': results.get('recall'),
                    'f1_score': results.get('f1_score')
                }
                
                # Extract configuration
                config = results.get('config', {})
                
                # Convert hidden_sizes to a string for categorization
                hidden_sizes = str(config.get('hidden_sizes', []))
                
                # Combine metrics and config
                combined = {**metrics, **config, 'hidden_sizes': hidden_sizes, 'run_path':run_path}
                data.append(combined)
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file: {results_file} - {e}")

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)
df.to_excel("Production/Model analysis/performance_analysis_nov4.xlsx")