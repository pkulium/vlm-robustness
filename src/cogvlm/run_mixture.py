import json
from collections import defaultdict
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration
)
from PIL import Image
import torch
import evaluate
import re
from tqdm import tqdm
import random
import copy
# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model configurations
model_configs = [
    {
        "name": "Qwen/Qwen2-VL-2B-Instruct",
        "model_class": "Qwen2VLForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "torch_dtype": "auto",
        "device_map": "auto"
    },
    {
        "name": "Qwen/Qwen2-VL-7B-Instruct",
        "model_class": "Qwen2VLForConditionalGeneration",
        "processor_class": "AutoProcessor",
        "torch_dtype": "auto",
        "device_map": "auto"
    }
]

# Initialize models and processors
models = {}
processors = {}

def get_model_class(class_name):
    if class_name == "AutoModelForCausalLM":
        return AutoModelForCausalLM
    elif class_name == "Qwen2VLForConditionalGeneration":
        return Qwen2VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model class: {class_name}")

# Load models and processors
for config in model_configs:
    model_class = get_model_class(config["model_class"])
    try:
        if config["torch_dtype"] == "auto":
            model = model_class.from_pretrained(
                config["name"],
                torch_dtype="auto",
                device_map=config["device_map"]
            )
        else:
            dtype = getattr(torch, config["torch_dtype"])
            model = model_class.from_pretrained(
                config["name"],
                torch_dtype=dtype,
                device_map=config["device_map"]
            )
        models[config["name"]] = model
    except Exception as e:
        print(f"Error loading model '{config['name']}': {e}")
        continue

    try:
        processor = AutoProcessor.from_pretrained(config["name"])
        processors[config["name"]] = processor
    except Exception as e:
        print(f"Error loading processor for model '{config['name']}': {e}")
        continue

# Load datasets
sections_to_load = ["add_hint", "insert_hint", "add_image"]
datasets = {}
for section in sections_to_load:
    try:
        datasets[section] = load_dataset("pkulium/I-ScienceQA", section)["test"]
    except Exception as e:
        print(f"Error loading section '{section}': {e}")
        continue

# Filter samples without images
def filter_samples_without_image(dataset):
    return [sample for sample in dataset if sample.get("image") is None]

add_hint_samples = filter_samples_without_image(datasets["add_hint"])
insert_hint_samples = filter_samples_without_image(datasets["insert_hint"])

# Get all images from add_image dataset
add_image_samples = datasets["add_image"]
image_pool = [sample["image"] for sample in add_image_samples if sample.get("image") is not None]


# Add random images to samples with a fixed seed for reproducibility
def add_random_images(samples, image_pool, seed=42):
    random.seed(seed)  # Set the random seed for repeatability
    for sample in samples:
        sample["image"] = random.choice(image_pool)
    return samples


# Create datasets for evaluation
evaluation_datasets = {
    "add_hint_no_image": add_hint_samples,
    "insert_hint_no_image": insert_hint_samples,
    "add_hint_with_image": add_random_images(copy.deepcopy(add_hint_samples), image_pool),
    "insert_hint_with_image": add_random_images(copy.deepcopy(insert_hint_samples), image_pool)
}
import pdb 
pdb.set_trace()

# Inference function
def run_inference(example, processor, model):
    if example["image"]:
        image = example["image"].convert("RGB")
    else:
        image = None

    context = example.get("hint", "")
    question = example.get("question", "")
    choices = example.get("choices", [])

    options = [chr(ord("A") + i) for i in range(len(choices))]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])

    pre_prompt = ""
    post_prompt = ("\nProvide the answer by selecting the option's letter from the given choices first, "
                   "then give a brief reasoning for your selection.")

    context_text = f"Context: {context}\n" if context else ""
    prompt = f"{pre_prompt}{context_text}{question}\n{choices_str}{post_prompt}"

    inputs = processor(
        text=[prompt],
        images=[image] if image else None,
        return_tensors="pt"
    ).to(device)

    generate_ids = model.generate(**inputs, max_new_tokens=128)
    output_text = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    match = re.match(r'^([A-Z])\b', output_text.strip())
    if match:
        predicted_answer = match.group(1)
    else:
        predicted_answer = output_text.strip().split()[0][0].upper() if output_text.strip() else ""

    return predicted_answer, output_text

# Initialize results dictionary
results = defaultdict(lambda: defaultdict(lambda: {"overall": {"exact_match": 0, "count": 0, "samples": []}}))

# Evaluation loop
for model_name, model in models.items():
    processor = processors.get(model_name, None)
    if processor is None:
        print(f"No processor found for model '{model_name}'. Skipping...")
        continue

    print(f"\n=== Evaluating Model: {model_name} ===")

    for dataset_name, dataset in evaluation_datasets.items():
        print(f"Processing dataset: {dataset_name}")
        
        for example in tqdm(dataset, desc=f"Model: {model_name} | Dataset: {dataset_name}"):
            try:
                prediction, output_text = run_inference(example, processor, model)

                if isinstance(example["answer"], int):
                    reference = chr(ord("A") + example["answer"])
                else:
                    reference = example["answer"]

                exact_match = 1.0 if prediction == reference else 0.0

                results[model_name][dataset_name]["overall"]["exact_match"] += exact_match
                results[model_name][dataset_name]["overall"]["count"] += 1

                sample_info = {
                    "question": example.get("question", ""),
                    "context": example.get("hint", ""),
                    "choices": example.get("choices", []),
                    "reference": reference,
                    "prediction": prediction,
                    "output_text": output_text,
                    "exact_match": exact_match,
                    "has_image": example.get("image") is not None
                }
                results[model_name][dataset_name]["overall"]["samples"].append(sample_info)

            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

# Calculate final exact match scores
final_results = {}
for model_name, datasets in results.items():
    final_results[model_name] = {}
    for dataset_name, data in datasets.items():
        if data["overall"]["count"] == 0:
            score = 0.0
        else:
            score = data["overall"]["exact_match"] / data["overall"]["count"]
        final_results[model_name][dataset_name] = {
            "exact_match": score,
            "count": data["overall"]["count"]
        }

# Print results
print("\n=== Exact Match Scores by Model and Dataset ===")
for model_name, datasets in final_results.items():
    print(f"\n--- Model: {model_name} ---")
    for dataset_name, metrics in datasets.items():
        print(f"Dataset: {dataset_name}")
        print(f"Exact Match Score: {metrics['exact_match']:.2f} ({metrics['count']} samples)")

# Save detailed results to JSON
output_data = {}
for model_name, datasets in results.items():
    output_data[model_name] = {}
    for dataset_name, data in datasets.items():
        output_data[model_name][dataset_name] = {
            "exact_match_score": data["overall"]["exact_match"] / data["overall"]["count"] if data["overall"]["count"] > 0 else 0.0,
            "total_samples": data["overall"]["count"],
            "samples": data["overall"]["samples"]
        }

output_file_path = "evaluation_results_by_model_and_dataset_mixture.json"

with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"\nDetailed results have been saved to '{output_file_path}'.")