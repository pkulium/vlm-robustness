# import torch
# import numpy as np
# import copy
# from PIL import Image
# import torch.nn.functional as F

# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# from datasets import load_dataset
# from PIL import Image

# device = "cuda" if torch.cuda.is_available() else "cpu"

# class Visualizer:
#     def __init__(self, model, processor) -> None:
#         self.model = model
#         self.processor = processor

#     def _map_subwords_to_words(self, sentence: str):
#         tokens = self.processor.tokenizer.tokenize(sentence)
#         mapping = []
#         word_idx = 0
#         for token in tokens:
#             if token.startswith("##"):
#                 mapping.append(word_idx - 1)
#             else:
#                 mapping.append(word_idx)
#                 word_idx += 1
#         return mapping, tokens

#     def _normalize_importance(self, word_importance):
#         min_importance = np.min(word_importance)
#         max_importance = np.max(word_importance)
#         return (word_importance - min_importance) / (max_importance - min_importance)

#     def vis_by_grad(self, input_sentence: str, image: Image.Image, label: str) -> dict:
#         self.model.eval()

#         mapping, tokens = self._map_subwords_to_words(input_sentence)
#         words = input_sentence.split()

#         inputs = self.processor(text=[input_sentence], images=[image] if image else None, return_tensors="pt").to(device)
#         input_ids = inputs['input_ids']
#         pixel_values = inputs['pixel_values']

#         embeddings = self.model.get_input_embeddings()(input_ids)
#         embeddings.requires_grad_()
#         embeddings.retain_grad()

#         label_index = ord(label) - ord('A')
#         label_tensor = torch.tensor([label_index], device=device)
#         outputs = self.model(inputs_embeds=embeddings, pixel_values=pixel_values)
#         logits = outputs.logits[:, -1, :]
#         loss = F.cross_entropy(logits, label_tensor)
#         loss.backward()

#         grads = embeddings.grad
#         word_grads = [torch.zeros_like(grads[0][0]) for _ in range(len(words))]

#         for idx, grad in enumerate(grads[0][:len(mapping)]):
#             word_grads[mapping[idx]] += grad

#         words_importance = [grad.norm().item() for grad in word_grads]
#         normalized_importance = self._normalize_importance(words_importance)

#         return dict(zip(words, normalized_importance))

#     def vis_by_delete(self, input_sentence: str, image: Image.Image, label: str) -> dict:
#         words = input_sentence.split()
#         label_index = ord(label) - ord('A')
#         label_tensor = torch.tensor([label_index], device=device)

#         inputs = self.processor(text=[input_sentence], images=[image] if image else None, return_tensors="pt").to(device)
#         outputs = self.model(**inputs)
#         logits = outputs.logits[:, -1, :]
#         original_loss = F.cross_entropy(logits, label_tensor).item()

#         word_importance = []
#         for i in range(len(words)):
#             new_words = copy.deepcopy(words)
#             del new_words[i]
#             new_sentence = ' '.join(new_words)
#             inputs = self.processor(text=[new_sentence], images=[image] if image else None, return_tensors="pt").to(device)
#             outputs = self.model(**inputs)
#             logits = outputs.logits[:, -1, :]
#             new_loss = F.cross_entropy(logits, label_tensor).item()

#             importance = abs(new_loss - original_loss)
#             word_importance.append(importance)

#         normalized_importance = self._normalize_importance(word_importance)

#         return dict(zip(words, normalized_importance))

# # Initialize the model and processor
# model_name = "Qwen/Qwen2-VL-7B-Instruct"
# model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
# processor = AutoProcessor.from_pretrained(model_name)

# # Initialize the Visualizer
# visualizer = Visualizer(model, processor)

# # Load the dataset sections
# sections = ["add_hint", "insert_hint"]
# datasets = {}
# for section in sections:
#     datasets[section] = load_dataset("pkulium/I-ScienceQA", section)["test"]

# # Load the original dataset sections
# original_sections = ["original_add_hint", "original_insert_context"]
# original_datasets = {}
# for section in original_sections:
#     original_datasets[section] = load_dataset("pkulium/I-ScienceQA", section)["test"]

# def find_counterpart(example, original_dataset):
#     for orig_example in original_dataset:
#         if (example['question'] == orig_example['question'] and
#             example['choices'] == orig_example['choices'] and
#             example['answer'] == orig_example['answer']):
#             return orig_example
#     return None  # Return None if no match is found

# def process_example(example, section, original_example):
#     image = None
#     if "image" in example and example["image"] is not None:
#         image = example["image"].convert("RGB")
#     input_sentence = example["question"]
#     label = chr(ord('A') + example["answer"])  # Convert numeric answer to letter
    
#     if not label:
#         print("Warning: Empty label detected")
#         return  # Skip this example or handle it appropriately

#     print(f"\nSection: {section}")
#     print(f"Question: {input_sentence}")
#     print(f"Hint: {example['hint']}")
#     print(f"Choices: {example['choices']}")
#     print(f"Correct Answer: {label}")

#     # Print original counterpart
#     if original_example:
#         print("\nOriginal Counterpart:")
#         print(f"Question: {original_example['question']}")
#         print(f"Hint: {original_example.get('hint', 'N/A')}")
#         print(f"Choices: {original_example['choices']}")
#         print(f"Correct Answer: {chr(ord('A') + original_example['answer'])}")
#     else:
#         print("\nNo matching original counterpart found.")

#     # Extract Fields: Context (hint), Question, Choices, and Answer Index
#     context = example.get("hint", "")
#     question = example.get("question", "")
#     choices = example.get("choices", [])

#     # Format Choices with Labels (A, B, C, ...)
#     options = [chr(ord("A") + i) for i in range(len(choices))]
#     choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])

#     # Define Prompts: Pre and Post
#     pre_prompt = ""
#     post_prompt = (
#         "\nProvide the answer by selecting the option's letter from the given choices first, "
#         "then give a brief reasoning for your selection."
#     )

#     # Construct the Full Prompt
#     if context:
#         context_text = f"Context: {context}\n"
#     else:
#         context_text = ""

#     prompt = f"{pre_prompt}{context_text}{question}\n{choices_str}{post_prompt}"
    
#     try:
#         # Visualize using deletion method
#         delete_results = visualizer.vis_by_delete(prompt, image, label)
#         print("Deletion-based visualization:")
#         for word, importance in delete_results.items():
#             print(f"  {word}: {importance:.4f}")
#     except Exception as e:
#         print(f"Error processing example: {e}")
    
#     print("-"*50)

# # Process a few samples from each section
# num_samples = 3  # Number of samples to process from each section

# for section, dataset in datasets.items():
#     print(f"\nProcessing {num_samples} samples from {section} section:")
#     original_section = "original_add_hint" if section == "add_hint" else "original_insert_context"
#     original_dataset = original_datasets[original_section]
    
#     processed_count = 0
#     for example in dataset:
#         original_example = find_counterpart(example, original_dataset)
#         if original_example:
#             process_example(example, section, original_example)
#             processed_count += 1
#             if processed_count >= num_samples:
#                 break
#         else:
#             print(f"Warning: No matching counterpart found for example in {section}")

import torch
import numpy as np
import copy
from PIL import Image
import torch.nn.functional as F

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from datasets import load_dataset
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

class Visualizer:
    def __init__(self, model, processor) -> None:
        self.model = model
        self.processor = processor

    def _map_subwords_to_words(self, sentence: str):
        tokens = self.processor.tokenizer.tokenize(sentence)
        mapping = []
        word_idx = 0
        for token in tokens:
            if token.startswith("##"):
                mapping.append(word_idx - 1)
            else:
                mapping.append(word_idx)
                word_idx += 1
        return mapping, tokens

    def _normalize_importance(self, word_importance):
        min_importance = np.min(word_importance)
        max_importance = np.max(word_importance)
        return (word_importance - min_importance) / (max_importance - min_importance)

    def vis_by_grad(self, input_sentence: str, image: Image.Image, label: str) -> dict:
        self.model.eval()

        mapping, tokens = self._map_subwords_to_words(input_sentence)
        words = input_sentence.split()

        inputs = self.processor(text=[input_sentence], images=[image] if image else None, return_tensors="pt").to(device)
        input_ids = inputs['input_ids']
        pixel_values = inputs['pixel_values']

        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_()
        embeddings.retain_grad()

        label_index = ord(label) - ord('A')
        label_tensor = torch.tensor([label_index], device=device)
        outputs = self.model(inputs_embeds=embeddings, pixel_values=pixel_values)
        logits = outputs.logits[:, -1, :]
        loss = F.cross_entropy(logits, label_tensor)
        loss.backward()

        grads = embeddings.grad
        word_grads = [torch.zeros_like(grads[0][0]) for _ in range(len(words))]

        for idx, grad in enumerate(grads[0][:len(mapping)]):
            word_grads[mapping[idx]] += grad

        words_importance = [grad.norm().item() for grad in word_grads]
        normalized_importance = self._normalize_importance(words_importance)

        return dict(zip(words, normalized_importance))

    def vis_by_delete(self, input_sentence: str, image: Image.Image, label: str) -> dict:
        words = input_sentence.split()
        label_index = ord(label) - ord('A')
        label_tensor = torch.tensor([label_index], device=device)

        inputs = self.processor(text=[input_sentence], images=[image] if image else None, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        original_loss = F.cross_entropy(logits, label_tensor).item()

        word_importance = []
        for i in range(len(words)):
            new_words = copy.deepcopy(words)
            del new_words[i]
            new_sentence = ' '.join(new_words)
            inputs = self.processor(text=[new_sentence], images=[image] if image else None, return_tensors="pt").to(device)
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            new_loss = F.cross_entropy(logits, label_tensor).item()

            importance = abs(new_loss - original_loss)
            word_importance.append(importance)

        normalized_importance = self._normalize_importance(word_importance)

        return dict(zip(words, normalized_importance))

# Initialize the model and processor
model_name = "Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_name)

# Initialize the Visualizer
visualizer = Visualizer(model, processor)

# Load the dataset sections
sections = ["add_hint", "insert_hint"]
datasets = {}
for section in sections:
    datasets[section] = load_dataset("pkulium/I-ScienceQA", section)["test"]

# Load the original dataset sections
original_sections = ["original_add_hint", "original_insert_context"]
original_datasets = {}
for section in original_sections:
    original_datasets[section] = load_dataset("pkulium/I-ScienceQA", section)["test"]

def find_counterpart(example, original_dataset):
    for orig_example in original_dataset:
        if (example['question'] == orig_example['question'] and
            example['choices'] == orig_example['choices'] and
            example['answer'] == orig_example['answer']):
            return orig_example
    return None  # Return None if no match is found

def process_example(example, section, original_example):
    image = None
    if "image" in example and example["image"] is not None:
        image = example["image"].convert("RGB")
    input_sentence = example["question"]
    label = chr(ord('A') + example["answer"])  # Convert numeric answer to letter
    
    if not label:
        print("Warning: Empty label detected")
        return  # Skip this example or handle it appropriately

    print(f"\nSection: {section}")
    print(f"Question: {input_sentence}")
    print(f"Hint: {example['hint']}")
    print(f"Choices: {example['choices']}")
    print(f"Correct Answer: {label}")

    # Print original counterpart
    # if original_example:
    #     print("\nOriginal Counterpart:")
    #     print(f"Question: {original_example['question']}")
    #     print(f"Hint: {original_example.get('hint', 'N/A')}")
    #     print(f"Choices: {original_example['choices']}")
    #     print(f"Correct Answer: {chr(ord('A') + original_example['answer'])}")
    # else:
    #     print("\nNo matching original counterpart found.")

    # Extract Fields: Context (hint), Question, Choices, and Answer Index
    context = example.get("hint", "")
    question = example.get("question", "")
    choices = example.get("choices", [])

    # Format Choices with Labels (A, B, C, ...)
    options = [chr(ord("A") + i) for i in range(len(choices))]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])

    # Define Prompts: Pre and Post
    pre_prompt = ""
    post_prompt = (
        "\nProvide the answer by selecting the option's letter from the given choices first, "
        "then give a brief reasoning for your selection."
    )

    # Construct the Full Prompt
    if context:
        context_text = f"Context: {context}\n"
    else:
        context_text = ""

    prompt = f"{pre_prompt}{context_text}{question}\n{choices_str}{post_prompt}"
    
    try:
        # Visualize using deletion method
        delete_results = visualizer.vis_by_delete(prompt, image, label)
        print("Deletion-based visualization:")
        for word, importance in delete_results.items():
            print(f"  {word}: {importance:.4f}")
    except Exception as e:
        print(f"Error processing example: {e}")
    
# Process a few samples from each section
num_samples = 3  # Number of samples to process from each section

for section, dataset in datasets.items():
    print(f"\nProcessing {num_samples} samples from {section} section:")
    original_section = "original_add_hint" if section == "add_hint" else "original_insert_context"
    original_dataset = original_datasets[original_section]
    
    processed_count = 0
    for example in dataset:
        original_example = find_counterpart(example, original_dataset)
        if original_example:
            process_example(example, section, original_example)
            process_example(original_example, section, example)
            print("-"*50)
            processed_count += 1
            if processed_count >= num_samples:
                break
        else:
            print(f"Warning: No matching counterpart found for example in {section}")
