export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/Users/aachigorin/work/code/diffusers/examples/dreambooth_tool/data/images/dog_0/"
export CLASS_DIR="/Users/aachigorin/work/code/diffusers/examples/dreambooth_tool/data/images/dogs/"
export OUTPUT_DIR="/Users/aachigorin/work/code/diffusers/examples/dreambooth_tool/data/saved_models/"

accelerate launch /Users/aachigorin/work/code/diffusers/examples/dreambooth_tool/tool.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --seed 2410 \
  --resolution=128 \
  --train_batch_size=1 \
  --train_text_encoder \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=1 \
  --sample_batch_size=1 \
  --max_train_steps=1 \
  --save_interval 200 \
  --config_path "/Users/aachigorin/work/code/diffusers/examples/dreambooth_tool/data/config0.json"


  # --use_8bit_adam \
  # --resolution=512 \
  # --mixed_precision="fp16" \
  # --instance_data_dir=$INSTANCE_DIR \
  #--class_data_dir=$CLASS_DIR \
  #--output_dir=$OUTPUT_DIR \
  #--save_sample_prompt="photo of sks dog" \
  #--instance_prompt="a photo of sks dog" \
  #--class_prompt="a photo of dog"