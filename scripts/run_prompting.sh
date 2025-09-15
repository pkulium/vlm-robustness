#!/bin/bash
# List of model keys
MODEL_KEYS=(
    "llava_hf"
    "internvl2"
    "gpt4v"
)

# Combined tasks
TASKS="I-ScienceQA_add_image_prompting,I-Scienceqa_insert_image_prompting,I-ScienceQA_insert_hint_prompting,I-Scienceqa_add_hint_prompting"

# Loop through the list of model keys
for model_key in "${MODEL_KEYS[@]}"; do
    echo "Launching model: $model_key..."
    if [ "$model_key" == "llava_hf" ]; then
        # Launch llava_hf with the 7b model
        echo "Launching llava_hf with 7b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="llava-hf/llava-1.5-7b-hf" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_7b" --output_path ./logs_rebuttal/ --limit 10

        # Launch llava_hf with the 13b model
        echo "Launching llava_hf with 13b model..."
        # accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="llava-hf/llava-1.5-7b-hf" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_13b" --output_path ./logs_rebuttal/ --limit 10
    elif [ "$model_key" == "instructblip" ]; then
        # Launch instructblip with the 7b model
        echo "Launching instructblip with 7b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="Salesforce/instructblip-vicuna-7b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_7b" --output_path ./logs_rebuttal/ --limit 10

        # Launch instructblip with the 13b model
        echo "Launching instructblip with 13b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="Salesforce/instructblip-vicuna-13b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_13b" --output_path ./logs_rebuttal/ --limit 10
    elif [ "$model_key" == "internvl2" ]; then
        echo "Launching internvl2 with 2b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="OpenGVLab/InternVL2-2B" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_7b" --output_path ./logs_rebuttal/ --limit 10

        echo "Launching internvl2 with 8b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="OpenGVLab/InternVL2-8B" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_13b" --output_path ./logs_rebuttal/ --limit 10
    else
        # General case for other models
        echo "Launching $model_key for combined tasks..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}" --output_path ./logs_rebuttal/ --limit 10
    fi
done
 
 