import os
import yaml

# Navigate to the directory containing your YAML files
directory = '.'  # Change this to your directory path

# Loop through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('_few_shot.yaml'):  # Select only the modified 'few_shot' files
        filepath = os.path.join(directory, filename)
        
        # Read the YAML file
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        
        # Check if the section exists and add the new parameter
        if 'model_specific_generation_kwargs' in data:
            data['model_specific_generation_kwargs']['default']['num_shot'] = 2  # Add the new parameter
        
        # Write the modified data back to the YAML file
        with open(filepath, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False)

print("Updated files successfully.")
