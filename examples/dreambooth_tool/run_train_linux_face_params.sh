#export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"

# v4 model
# 100 inference steps
CUDA_VISIBLE_DEVICES=$1 \
accelerate launch --main_process_port 2419 \
  /ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/tool.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --seed 1234 \
  --resolution=512 \
  --train_batch_size=1 \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --sample_batch_size=4 \
  --max_train_steps=800 \
  --save_interval 1000 \
  --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/configs/config_linux_mix0.json" \
  --num_inference_steps 100 \
  --n_images_to_generate_for_each_prompt 4

# --num_inference_steps 50 \
# --n_images_to_generate_for_each_prompt 4 \
# --max_train_steps=800 \
#   --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/configs/config_linux0.json" \
#   --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/configs/config_linux_me.json" \
#   --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/configs/config_linux_mani.json" \
#   --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/configs/config_linux_max_friend.json" \
#   --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/configs/config_linux_max.json" \ 
#  --skip_training