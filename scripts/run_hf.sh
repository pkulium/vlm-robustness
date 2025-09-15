# List of model keys
MODEL_KEYS=(
    "llava_hf"
)

# Array of tasks
TASKS=(
    "I-scienceqa_insert_image"
    "I-ScienceQA_add_image"
    "I-ScienceQA_add_image_unsplash"
    "I-scienceqa_insert_image_original"
)

# Loop through the list of model keys
for model_key in "${MODEL_KEYS[@]}"; do
    echo "Launching model: $model_key..."
    
    # Nested loop for tasks
    for task in "${TASKS[@]}"; do
        echo "Processing task: $task..."
        if [ "$model_key" == "llava" ]; then
            # Launch llava with the 7b model
            echo "Launching llava with 7b model for task: $task..."
            accelerate launch --num_processes=8 -m lmms_eval --model $model_key --model_args pretrained="liuhaotian/llava-v1.5-7b" --tasks $task --batch_size 1 --log_samples --log_samples_suffix "${model_key}_7b_${task}" --output_path ./logs/

            # Launch llava with the 13b model
            echo "Launching llava with 13b model for task: $task..."
            accelerate launch --num_processes=8 -m lmms_eval --model $model_key --model_args pretrained="liuhaotian/llava-v1.5-13b" --tasks $task --batch_size 1 --log_samples --log_samples_suffix "${model_key}_13b_${task}" --output_path ./logs/
        elif [ "$model_key" == "instructblip" ]; then
            # Launch llava with the 7b model
            echo "Launching llava with 7b model for task: $task..."
            accelerate launch --num_processes=8 -m lmms_eval --model $model_key --model_args pretrained="Salesforce/instructblip-vicuna-7b" --tasks $task --batch_size 1 --log_samples --log_samples_suffix "${model_key}_7b_${task}" --output_path ./logs/

            # Launch llava with the 13b model
            echo "Launching llava with 13b model for task: $task..."
            accelerate launch --num_processes=8 -m lmms_eval --model $model_key --model_args pretrained="Salesforce/instructblip-vicuna-7b" --tasks $task --batch_size 1 --log_samples --log_samples_suffix "${model_key}_13b_${task}" --output_path ./logs/
        elif [ "$model_key" == "llava_hf" ]; then
            # Launch llava_hf with different model versions
            echo "Launching llava_hf with mistral-7b model for task: $task..."
            accelerate launch --num_processes=8 -m lmms_eval --model $model_key --model_args pretrained="llava-hf/llava-v1.6-mistral-7b-hf" --tasks $task --batch_size 1 --log_samples --log_samples_suffix "${model_key}_mistral-7b_${task}" --output_path ./logs/

            echo "Launching llava_hf with vicuna-7b model for task: $task..."
            accelerate launch --num_processes=8 -m lmms_eval --model $model_key --model_args pretrained="llava-hf/llava-v1.6-vicuna-7b-hf" --tasks $task --batch_size 1 --log_samples --log_samples_suffix "${model_key}_vicuna-7b_${task}" --output_path ./logs/

            echo "Launching llava_hf with vicuna-13b model for task: $task..."
            accelerate launch --num_processes=8 -m lmms_eval --model $model_key --model_args pretrained="llava-hf/llava-v1.6-vicuna-13b-hf" --tasks $task --batch_size 1 --log_samples --log_samples_suffix "${model_key}_vicuna-13b_${task}" --output_path ./logs/

            echo "Launching llava_hf with 34b model for task: $task..."
            accelerate launch --num_processes=8 -m lmms_eval --model $model_key --model_args pretrained="llava-hf/llava-v1.6-34b-hf" --tasks $task --batch_size 1 --log_samples --log_samples_suffix "${model_key}_34b_${task}" --output_path ./logs/
        else
            # General case for other models
            accelerate launch --num_processes=8 -m lmms_eval --model $model_key --tasks $task --batch_size 1 --log_samples --log_samples_suffix "${model_key}_${task}" --output_path ./logs/
        fi
    done
done
