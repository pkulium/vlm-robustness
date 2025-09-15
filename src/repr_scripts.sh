# install lmms_eval without building dependencies
pip install --no-deps -U -e .

# install LLaVA without building dependencies
cd LLaVA
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install flash-attn --no-build-isolation
pip install accelerate deepspeed --upgrade
pip install git+https://github.com/huggingface/transformers

cd ..
# install all the requirements that require for reproduce llava results

# Run and exactly reproduce llava_v1.5 results!
# mme as an example
accelerate launch --num_processes=1 -m lmms_eval --model llava   --model_args pretrained="liuhaotian/llava-v1.5-7b,use_flash_attention_2=False,device_map=auto"   --tasks mme  --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/
