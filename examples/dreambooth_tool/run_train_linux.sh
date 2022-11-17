export MODEL_NAME="runwayml/stable-diffusion-v1-5"

# PROD
# memory during training: 13649 gb during training (+ not_cache_latents, -train_text_encoder )
# memory during training: ??? gb during training (+ not_cache_latents )
# baseline
CUDA_VISIBLE_DEVICES=$1 \
accelerate launch /ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/tool.py \
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
  --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/configs/config_linux_arina.json" \
  --num_inference_steps 50 \
  --n_images_to_generate_for_each_prompt 4 \
  --skip_training

# --num_inference_steps 50 \
# --n_images_to_generate_for_each_prompt 4 \
# --max_train_steps=800 \
#   --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/configs/config_linux0.json" \
#  --skip_training