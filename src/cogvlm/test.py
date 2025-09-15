# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class Visualizer:
    def __init__(self, model_name: str) -> None:
        """
        Initialize the Visualizer class with a Hugging Face model.

        Parameters:
        - model_name (str): The name or path of the Hugging Face model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def _map_subwords_to_words(self, sentence: str):
        """
        Convert a sentence into tokens and map subword tokens to their corresponding words.

        Parameters:
        - sentence (str): The input sentence.

        Returns:
        - mapping (list): List mapping subword tokens to word indices.
        - tokens (list): Tokenized version of the input sentence.
        """
        tokens = self.tokenizer.tokenize(sentence)
        mapping = []
        word_idx = 0
        for token in tokens:
            if token.startswith("▁") or token.startswith("##"):  # Adjusted for different tokenizers
                mapping.append(word_idx)
                word_idx += 1
            else:
                mapping.append(word_idx - 1)
        return mapping, tokens

    def _normalize_importance(self, word_importance):
        """
        Normalize importance values of words in a sentence using min-max scaling.

        Parameters:
        - word_importance (list): List of importance values for each word.

        Returns:
        - list: Normalized importance values for each word.
        """
        min_importance = np.min(word_importance)
        max_importance = np.max(word_importance)
        if max_importance - min_importance == 0:
            return [0.0 for _ in word_importance]
        return (word_importance - min_importance) / (max_importance - min_importance)

    def vis_by_grad(self, input_sentence: str, label: int) -> dict:
        """
        Visualize word importance in an input sentence based on gradient information.

        This method uses the gradients of the model's outputs with respect to its 
        input embeddings to estimate word importance.

        Parameters:
        - input_sentence (str): The input sentence.
        - label (int): The target label index.

        Returns:
        - dict: Dictionary with words as keys and their normalized importance as values.
        """        
        self.model.eval()

        mapping, tokens = self._map_subwords_to_words(input_sentence)
        words = input_sentence.split()

        inputs = self.tokenizer(input_sentence, return_tensors="pt")
        input_ids = inputs['input_ids']
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_()
        embeddings.retain_grad()

        outputs = self.model(inputs_embeds=embeddings, attention_mask=inputs['attention_mask'], labels=input_ids.new([label]))
        loss = outputs.loss
        loss.backward()

        grads = embeddings.grad  # Shape: (1, seq_len, hidden_size)
        word_grads = [torch.zeros_like(grads[0][0]) for _ in range(len(words))]  # Initialize gradient vectors for each word

        # Aggregate gradients for each word
        for idx, grad in enumerate(grads[0][:len(mapping)]):
            word_grads[mapping[idx]] += grad

        words_importance = [grad.norm().item() for grad in word_grads]
        normalized_importance = self._normalize_importance(words_importance)

        return dict(zip(words, normalized_importance))

    def vis_by_delete(self, input_sentence: str, label: int) -> dict:
        """
        Visualize word importance in an input sentence by deletion method.

        For each word in the sentence, the method deletes it and measures the 
        change in the model's output. A higher change indicates higher importance.

        Parameters:
        - input_sentence (str): The input sentence.
        - label (int): The target label index.

        Returns:
        - dict: Dictionary with words as keys and their normalized importance as values.
        """        
        words = input_sentence.split()
        encoded_label = torch.tensor([label]).unsqueeze(0)
        inputs = self.tokenizer(input_sentence, return_tensors="pt")
        with torch.no_grad():
            original_outputs = self.model(**inputs, labels=encoded_label)
            original_loss = original_outputs.loss.item()

        word_importance = []
        for i in range(len(words)):
            new_words = copy.deepcopy(words)
            del new_words[i]
            new_sentence = ' '.join(new_words)
            inputs = self.tokenizer(new_sentence, return_tensors="pt")
            with torch.no_grad():
                new_outputs = self.model(**inputs, labels=encoded_label)
                new_loss = new_outputs.loss.item()

            importance = abs(new_loss - original_loss)
            word_importance.append(importance)

        normalized_importance = self._normalize_importance(word_importance)

        return dict(zip(words, normalized_importance))


import torch

def main():
    # Specify the Hugging Face model name
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    # Initialize the Visualizer with the specified model
    visualizer = Visualizer(model_name=model_name)

    # Example sentence
    sentence = "I absolutely loved the new movie; it was fantastic and thrilling!"

    # The SST-2 dataset has two labels: 0 for negative and 1 for positive
    # Let's assume we are interested in the positive sentiment
    label = 1

    # Get word importance based on gradients
    grad_importance = visualizer.vis_by_grad(input_sentence=sentence, label=label)
    print("Word Importance based on Gradients:")
    for word, importance in grad_importance.items():
        print(f"{word}: {importance:.4f}")

    print("\n" + "-"*50 + "\n")

    # Get word importance based on deletion
    delete_importance = visualizer.vis_by_delete(input_sentence=sentence, label=label)
    print("Word Importance based on Deletion:")
    for word, importance in delete_importance.items():
        print(f"{word}: {importance:.4f}")

if __name__ == "__main__":
    main()
