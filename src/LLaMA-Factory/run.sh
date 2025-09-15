#!/bin/bash
llamafactory-cli train examples/train_lora/qwen2vl_add_image.yaml
llamafactory-cli train examples/train_lora/qwen2vl_insert_image.yaml
llamafactory-cli train examples/train_lora/llava_add_image.yaml
llamafactory-cli train examples/train_lora/llava_insert_image.yaml
