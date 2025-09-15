import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import json
from collections import defaultdict
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    # Qwen2VLForConditionalGeneration  # Ensure this class is correctly imported
)
from PIL import Image
import torch
import evaluate
import re
from tqdm import tqdm  # Optional: For progress bars

# ----------------------------
# 1. Setup and Initialization
# ----------------------------

# Set device: GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model configurations
model_configs = [
    # {
    #     "name": "Qwen/Qwen2-VL-2B-Instruct",
    #     "model_class": "Qwen2VLForConditionalGeneration",
    #     "processor_class": "AutoProcessor",
    #     "torch_dtype": "auto",
    #     "device_map": "auto"
    # },
    {
        "name": "THUDM/cogvlm2-llama3-chat-19B",
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoProcessor",
        "torch_dtype": "float16",
        "device_map": "auto"
    }
]

# Initialize dictionaries to hold models and processors
models = {}
processors = {}

# Function to dynamically load model classes
def get_model_class(class_name):
    if class_name == "AutoModelForCausalLM":
        return AutoModelForCausalLM
    elif class_name == "Qwen2VLForConditionalGeneration":
        return Qwen2VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model class: {class_name}")

# Load all models and processors
for config in model_configs:
    model_class = get_model_class(config["model_class"])
    try:
        if config["torch_dtype"] == "auto":
            model = model_class.from_pretrained(
                config["name"],
                torch_dtype="auto",
                device_map=config["device_map"],
                trust_remote_code=True
            )
        else:
            dtype = getattr(torch, config["torch_dtype"])
            model = model_class.from_pretrained(
                config["name"],
                torch_dtype=dtype,
                device_map=config["device_map"],
                trust_remote_code=True
            )
        models[config["name"]] = model
    except Exception as e:
        print(f"Error loading model '{config['name']}': {e}")
        continue  # Skip loading this model if there's an error

    try:
        processor = AutoProcessor.from_pretrained(config["name"], trust_remote_code=True)
        processors[config["name"]] = processor
    except Exception as e:
        print(f"Error loading processor for model '{config['name']}': {e}")
        continue  # Skip loading processor if there's an error

# Define the sections of the dataset to load
sections_with_distract = ["add_image", "insert_image", "add_hint", "insert_hint"]
model_configs = model_configs[:1]
sections_without_distract = ["original_add_image", "original_insert_image", "original_add_hint", "original_insert_context"]

# Combine all sections
all_sections = sections_with_distract 

datasets = {}

# Load each section's test split
for section in all_sections:
    try:
        datasets[section] = load_dataset("pkulium/I-ScienceQA", section)["test"]
    except Exception as e:
        print(f"Error loading section '{section}': {e}")
        continue  # Skip sections that fail to load

# ----------------------------
# 2. Define the Inference Function
# ----------------------------

def run_inference(example, processor, model):
    """
    Processes a single example by formatting the prompt, passing it to the model,
    and extracting both the predicted answer and the raw output text.
    """
    # 1. Image Processing: Convert to RGB
    if example["image"]:
        image = example["image"].convert("RGB")
    else:
        image = None

    # 2. Extract Fields: Context (hint), Question, Choices, and Answer Index
    context = example.get("hint", "")
    question = example.get("question", "")
    choices = example.get("choices", [])

    # 3. Format Choices with Labels (A, B, C, ...)
    options = [chr(ord("A") + i) for i in range(len(choices))]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])

    # 4. Construct the Full Prompt
    if context:
        context_text = f"Context: {context}\n"
    else:
        context_text = ""
        
    post_prompt = (
        "\n- Carefully read and focus on the main question.\n"
        "- Disregard any distracting elements in the image and ignore irrelevant textual hints.\n"
        "- Select the correct answer by providing the option's letter from the given choices first.\n"
        "- Then, provide a brief reasoning for your selection.\n"
        "- Remeber, begin your answer with a single letter from the given choices first."
    )
    prompt = f"{context_text}{question}\n{choices_str}{post_prompt}"

    # 5. Prepare Inputs for the Model
    input_by_model = model.build_conversation_input_ids(
        processor,
        query=prompt,
        history=[],
        images=[image] if image else None
    )

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
        'images': [[input_by_model['images'][0].to(device = device, dtype = torch.float16)]] if image else None,
    }

    # 6. Generate the Response
    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=128)
        output_text = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    # 7. Extract the Predicted Answer Using Regex
    match = re.match(r'^([A-Z])\b', output_text[len(prompt):].strip())
    if match:
        predicted_answer = match.group(1)
    else:
        # Fallback: Extract the first character of the first word and convert to uppercase
        predicted_answer = output_text.strip().split()[0][0].upper() if output_text.strip() else ""

    return predicted_answer, output_text

# ----------------------------
# 3. Initialize Evaluation Metrics
# ----------------------------

exact_match_metric = evaluate.load("exact_match")

# ----------------------------
# 4. Initialize Results Dictionary
# ----------------------------

# Structure:
# results[model_name][section][distract_type] = {
#     "exact_match": cumulative_score,
#     "count": number_of_samples,
#     "samples": [sample_info, ...]
# }
# Additionally, include 'overall' as a distract_type to aggregate all distract types within a section

results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"exact_match": 0, "count": 0, "samples": []})))

# ----------------------------
# 5. Iterate Through Each Model, Section, and Dataset
# ----------------------------

for model_name, model in models.items():
    processor = processors.get(model_name, None)
    if processor is None:
        print(f"No processor found for model '{model_name}'. Skipping...")
        continue  # Skip if processor is not found

    print(f"\n=== Evaluating Model: {model_name} ===")

    for section, dataset in datasets.items():
        print(f"Processing section: {section}")
        for example in tqdm(dataset, desc=f"Model: {model_name} | Section: {section}"):
            try:
                # Determine if 'distract_type' exists
                distract_type = example.get("distract_type", "no_distract_type")  # Assign default if missing

                # Run Inference and Capture Both Predicted Answer and Output Text
                prediction, output_text = run_inference(example, processor, model)

                # Format Reference Answer
                if isinstance(example["answer"], int):
                    reference = chr(ord("A") + example["answer"])
                else:
                    reference = example["answer"]

                # Determine Exact Match
                exact_match = 1.0 if prediction == reference else 0.0

                # Update Results for Distract Type
                results[model_name][section][distract_type]["exact_match"] += exact_match
                results[model_name][section][distract_type]["count"] += 1

                # Update Results for Overall Section
                results[model_name][section]["overall"]["exact_match"] += exact_match
                results[model_name][section]["overall"]["count"] += 1

                # Save Sample Information, Including output_text
                sample_info = {
                    "question": example.get("question", ""),
                    "context": example.get("hint", ""),
                    "choices": example.get("choices", []),
                    "reference": reference,
                    "prediction": prediction,
                    "output_text": output_text,  # Saving the raw output text
                    "exact_match": exact_match
                }
                results[model_name][section][distract_type]["samples"].append(sample_info)
                results[model_name][section]["overall"]["samples"].append(sample_info)

            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

# ----------------------------
# 6. Calculate Final Exact Match Scores
# ----------------------------

final_results = {}
for model_name, sections in results.items():
    final_results[model_name] = {}
    for section, distract_types in sections.items():
        final_results[model_name][section] = {}
        for distract_type, data in distract_types.items():
            if data["count"] == 0:
                score = 0.0
            else:
                score = data["exact_match"] / data["count"]
            final_results[model_name][section][distract_type] = {
                "exact_match": score,
                "count": data["count"]
            }

# ----------------------------
# 7. Print Results
# ----------------------------

print("\n=== Exact Match Scores by Model, Section, and Distract Type ===")
for model_name, sections in final_results.items():
    print(f"\n--- Model: {model_name} ---")
    for section, distract_types in sections.items():
        print(f"\nSection: {section}")
        for distract_type, metrics in distract_types.items():
            if distract_type == "overall":
                print(f"  Overall:")
            else:
                print(f"  Distract Type: {distract_type}")
            print(f"    Exact Match Score: {metrics['exact_match']:.2f} ({metrics['count']} samples)")

# ----------------------------
# 8. Save Detailed Results to a JSON File
# ----------------------------

output_data = {}
for model_name, sections in results.items():
    output_data[model_name] = {}
    for section, distract_types in sections.items():
        output_data[model_name][section] = {}
        for distract_type, data in distract_types.items():
            output_data[model_name][section][distract_type] = {
                "exact_match_score": data["exact_match"] / data["count"] if data["count"] > 0 else 0.0,
                "total_samples": data["count"],
                "samples": data["samples"]
            }

# Specify the Output File Path
output_file_path = "evaluation_results_by_model_section_distract_type_defense_19B.json"

# Save to JSON
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"\nDetailed results have been saved to '{output_file_path}'.")
