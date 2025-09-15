import os
import json
import re 

def calculate_exact_match_scores_modified(base_path):
    # Dictionary to hold results
    results = {}

    # Regular expression to match files of interest
    pattern = re.compile(r"I-scienceqa_.*\.json")

    # Walk through the directory to find all relevant files
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if pattern.match(file):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    logs = data['logs']
                    distract_type_scores = {}

                    for log in logs:
                        distract_type = log['doc']['distract_type']
                        exact_match_score = log['exact_match']

                        if distract_type not in distract_type_scores:
                            distract_type_scores[distract_type] = []

                        distract_type_scores[distract_type].append(exact_match_score)

                    # Calculate average exact match for each distract type
                    distract_type_averages = {
                        dtype: sum(scores)/len(scores) for dtype, scores in distract_type_scores.items()
                    }
                    
                    results[file_path] = distract_type_averages

    # Save the results to a file in the base path
    with open(os.path.join(base_path, 'distraction_type_scores.json'), 'w') as f:
        json.dump(results, f, indent=4)

    return "Scores calculated and saved successfully."

# Assuming the base path is provided or set to current working directory
base_path = '.'  # This would be set to the appropriate path where the folders are located
calculate_exact_match_scores_modified(base_path)
