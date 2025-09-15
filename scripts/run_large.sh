#!/bin/bash
# env minigptv
# List of model keys
MODEL_KEYS=(
    "internvl2"
    "llava"
)

# Combined tasks
TASKS="I-ScienceQA_add_image,I-scienceqa_insert_image"

# Loop through the list of model keys
for model_key in "${MODEL_KEYS[@]}"; do
    echo "Launching model: $model_key..."
    if [ "$model_key" == "llava" ]; then
       
        # Launch llava with the 34b model
        echo "Launching llava with 34b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="liuhaotian/llava-v1.6-34b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_34b" --output_path ./logs/
    elif [ "$model_key" == "internvl2" ]; then
        
        echo "Launching internvl2 with 1b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="OpenGVLab/InternVL2-1b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_1b" --output_path ./logs/
    else
        # General case for other models
        echo "Launching $model_key for combined tasks..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}" --output_path ./logs/
    fi
done

#============================================================================
MODEL_KEYS=(
    "internvl2"
    "llava"
)

# Combined tasks
TASKS="I-ScienceQA_insert_hint,I-scienceqa_add_hint"

# Loop through the list of model keys
for model_key in "${MODEL_KEYS[@]}"; do
    echo "Launching model: $model_key..."
    if [ "$model_key" == "llava" ]; then
       
        # Launch llava with the 34b model
        echo "Launching llava with 34b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="liuhaotian/llava-v1.6-34b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_34b" --output_path ./logs/
    elif [ "$model_key" == "internvl2" ]; then
        
        echo "Launching internvl2 with 1b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="OpenGVLab/InternVL2-1b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_1b" --output_path ./logs/
    else
        # General case for other models
        echo "Launching $model_key for combined tasks..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}" --output_path ./logs/
    fi
done



#============================================================================
MODEL_KEYS=(
    "internvl2"
    "llava"
)

# Combined tasks
TASKS="I-ScienceQA_insert_hint_original,I-scienceqa_add_hint_original"

# Loop through the list of model keys
for model_key in "${MODEL_KEYS[@]}"; do
    echo "Launching model: $model_key..."
    if [ "$model_key" == "llava" ]; then
       
        # Launch llava with the 34b model
        echo "Launching llava with 34b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="liuhaotian/llava-v1.6-34b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_34b" --output_path ./logs/
    elif [ "$model_key" == "internvl2" ]; then
        
        echo "Launching internvl2 with 1b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="OpenGVLab/InternVL2-1b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_1b" --output_path ./logs/

        echo "Launching internvl2 with 26b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="OpenGVLab/InternVL2-26b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_26b" --output_path ./logs/
    else
        # General case for other models
        echo "Launching $model_key for combined tasks..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}" --output_path ./logs/
    fi
done


MODEL_KEYS=(
    "internvl2"
)

# Combined tasks
TASKS="I-ScienceQA_add_image_original,I-scienceqa_insert_image_original"

# Loop through the list of model keys
for model_key in "${MODEL_KEYS[@]}"; do
    echo "Launching model: $model_key..."
    if [ "$model_key" == "llava" ]; then
       
        # Launch llava with the 34b model
        echo "Launching llava with 34b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="liuhaotian/llava-v1.6-34b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_34b" --output_path ./logs/
    elif [ "$model_key" == "internvl2" ]; then 
        echo "Launching internvl2 with 1b model..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --model_args pretrained="OpenGVLab/InternVL2-1b" --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}_1b" --output_path ./logs/
    else
        # General case for other models
        echo "Launching $model_key for combined tasks..."
        accelerate launch --num_processes=16 -m lmms_eval --model $model_key --tasks $TASKS --batch_size 1 --log_samples --log_samples_suffix "${model_key}" --output_path ./logs/
    fi
done
