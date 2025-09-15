#!/bin/bash

MODEL_KEYS=(
    "llava"
    "internvl2"
    "gpt4v"
)

# Combined tasks
TASKS="I-ScienceQA_insert_hint","I-scienceqa_add_hint","I-ScienceQA_insert_hint_original","I-scienceqa_add_hint_original"


# Loop through the list of model keys
for model_key in "${MODEL_KEYS[@]}"; do
    echo "Launching model: $model_key..."
    if [ "$model_key" == "llava" ]; then
        # Launch llava with the 7b model
        echo "Launching llava with 7b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="liuhaotian/llava-v1.5-7b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_7b" --output_path ./logs/

        # Launch llava with the 13b model
        echo "Launching llava with 13b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="liuhaotian/llava-v1.5-13b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_13b" --output_path ./logs/
    elif [ "$model_key" == "instructblip" ]; then
        # Launch instructblip with the 7b model
        echo "Launching instructblip with 7b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="Salesforce/instructblip-vicuna-7b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_7b" --output_path ./logs/

        # Launch instructblip with the 13b model
        echo "Launching instructblip with 13b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="Salesforce/instructblip-vicuna-13b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_13b" --output_path ./logs/
    elif [ "$model_key" == "internvl2" ]; then
        echo "Launching internvl2 with 2b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="OpenGVLab/InternVL2-2B" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_7b" --output_path ./logs/

        echo "Launching internvl2 with 8b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="OpenGVLab/InternVL2-8B" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_13b" --output_path ./logs/
    else
        # General case for other models
        echo "Launching $model_key for combined tasks..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}" --output_path ./logs/
    fi
done
