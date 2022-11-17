export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/data/images/dog_0/"
export CLASS_DIR="/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/data/images/dogs/"
export OUTPUT_DIR="/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/data/saved_models/"


# memory during training: 13649 gb during training (+ not_cache_latents, - train_text_encoder)
CUDA_VISIBLE_DEVICES=$1 \
accelerate launch /ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/tool.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --seed 2410 \
  --resolution=512 \
  --train_batch_size=1 \
  --not_cache_latents \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=5 \
  --sample_batch_size=4 \
  --max_train_steps=1 \
  --save_interval 1000 \
  --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/configs/config_linux_cola.json" \
  --num_inference_steps 1 \
  --n_images_to_generate_for_each_prompt 1


  # --skip_training_for_debug \
  # --instance_data_dir=$INSTANCE_DIR \
  #--class_data_dir=$CLASS_DIR \
  #--output_dir=$OUTPUT_DIR \
  #--save_sample_prompt="photo of sks dog" \
  #--instance_prompt="a photo of sks dog" \
  #--class_prompt="a photo of dog"
  
  
  # DEBUG MEMORY
# memory usage with deepspeed training (throuh accelerat config): 
# падает по памяти...
# In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
# Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU [4] MPS): 2
# How many different machines will you use (use more than 1 for multi-node training)? [1]:
# Do you want to use DeepSpeed? [yes/NO]: yes
# Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
# What should be your DeepSpeed's ZeRO optimization stage (0, 1, 2, 3)? [2]:
# Where to offload optimizer states? [none/cpu/nvme]: cpu
# Where to offload parameters? [none/cpu/nvme]: cpu
# How many gradient accumulation steps you're passing in your script? [1]:
# Do you want to use gradient clipping? [yes/NO]:
# Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:
# How many GPU(s) should be used for distributed training? [1]:1
# Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: fp16
# same shit
# CUDA_VISIBLE_DEVICES=$1 \
# accelerate launch /ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/tool.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --seed 2410 \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --train_text_encoder \
#   --not_cache_latents \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --mixed_precision="fp16" \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=5 \
#   --sample_batch_size=4 \
#   --max_train_steps=2 \
#   --save_interval 1000 \
#   --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/config0_linux.json" \
#   --num_inference_steps 5 \
#   --n_images_to_generate_for_each_prompt 2

# memory usage with deepspeed training (throuh accelerat config): 
# падает по памяти...
# In which compute environment are you running? ([0] This machine, [1] AWS (Amazon SageMaker)): 0
# Which type of machine are you using? ([0] No distributed training, [1] multi-CPU, [2] multi-GPU, [3] TPU [4] MPS): 2
# How many different machines will you use (use more than 1 for multi-node training)? [1]:
# Do you want to use DeepSpeed? [yes/NO]: yes
# Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
# What should be your DeepSpeed's ZeRO optimization stage (0, 1, 2, 3)? [2]:
# Where to offload optimizer states? [none/cpu/nvme]: cpu
# Where to offload parameters? [none/cpu/nvme]: cpu
# How many gradient accumulation steps you're passing in your script? [1]:
# Do you want to use gradient clipping? [yes/NO]:
# Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:
# How many GPU(s) should be used for distributed training? [1]:1
# Do you wish to use FP16 or BF16 (mixed precision)? [NO/fp16/bf16]: fp16
# same shit
# CUDA_VISIBLE_DEVICES=$1 \
# accelerate launch /ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/tool.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --seed 2410 \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --train_text_encoder \
#   --not_cache_latents \
#   --use_8bit_adam \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --mixed_precision="fp16" \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=5 \
#   --sample_batch_size=4 \
#   --max_train_steps=2 \
#   --save_interval 1000 \
#   --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/config0_linux.json" \
#   --num_inference_steps 5 \
#   --n_images_to_generate_for_each_prompt 2

# memory usage with deepspeed training: >15000 mb (+ not_cache_latents, + accelerate launch --use_deepspeed --zero_stage=2 --gradient_accumulation_steps=1 --offload_param_device=cpu --offload_optimizer_device=cpu)
# падает по памяти...
# CUDA_VISIBLE_DEVICES=$1 \
# accelerate launch --use_deepspeed --zero_stage=2 --gradient_accumulation_steps=1 --offload_param_device=cpu --offload_optimizer_device=cpu /ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/tool.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --seed 2410 \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --train_text_encoder \
#   --not_cache_latents \
#   --use_8bit_adam \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --mixed_precision="fp16" \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=5 \
#   --sample_batch_size=4 \
#   --max_train_steps=2 \
#   --save_interval 1000 \
#   --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/config0_linux.json" \
#   --num_inference_steps 5 \
#   --n_images_to_generate_for_each_prompt 2

# memory during training: 14700 mb during training (+ not_cache_latents)
# works fine
# CUDA_VISIBLE_DEVICES=$1 \
# accelerate launch /ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/tool.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --seed 2410 \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --train_text_encoder \
#   --not_cache_latents \
#   --use_8bit_adam \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --mixed_precision="fp16" \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=5 \
#   --sample_batch_size=4 \
#   --max_train_steps=2 \
#   --save_interval 1000 \
#   --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/config0_linux.json" \
#   --num_inference_steps 5 \
#   --n_images_to_generate_for_each_prompt 2
  
# memory during training: 14538 gb during training (basic setup)
# this one fails during execution
# CUDA_VISIBLE_DEVICES=$1 \
# accelerate launch /ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/tool.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --seed 2410 \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --train_text_encoder \
#   --use_8bit_adam \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --mixed_precision="fp16" \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=5 \
#   --sample_batch_size=4 \
#   --max_train_steps=2 \
#   --save_interval 1000 \
#   --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/config0_linux.json" \
#   --num_inference_steps 5 \
#   --n_images_to_generate_for_each_prompt 2
  
# memory during training: 13649 gb during training (+ not_cache_latents, - train_text_encoder)
# CUDA_VISIBLE_DEVICES=$1 \
# accelerate launch /ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/tool.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
#   --with_prior_preservation \
#   --prior_loss_weight=1.0 \
#   --seed 2410 \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --not_cache_latents \
#   --use_8bit_adam \
#   --gradient_accumulation_steps=1 \
#   --gradient_checkpointing \
#   --mixed_precision="fp16" \
#   --learning_rate=1e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_class_images=5 \
#   --sample_batch_size=4 \
#   --max_train_steps=2 \
#   --save_interval 1000 \
#   --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/config0_linux.json" \
#   --num_inference_steps 5 \
#   --n_images_to_generate_for_each_prompt 2

# this works on T4 GPU
# CUDA_VISIBLE_DEVICES=$1 \
# accelerate launch /ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/tool.py \
  # --pretrained_model_name_or_path=$MODEL_NAME \
  # --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  # --with_prior_preservation \
  # --prior_loss_weight=1.0 \
  # --seed 2410 \
  # --resolution=512 \
  # --train_batch_size=1 \
  # --train_text_encoder \
  # --use_8bit_adam \
  # --gradient_accumulation_steps=1 \
  # --gradient_checkpointing \
  # --mixed_precision="fp16" \
  # --learning_rate=1e-6 \
  # --lr_scheduler="constant" \
  # --lr_warmup_steps=0 \
  # --num_class_images=50 \
  # --sample_batch_size=4 \
  # --max_train_steps=800 \
  # --save_interval 1000 \
  # --config_path "/ssd/aachigorin/code/diffusers_my/examples/dreambooth_tool/config0_linux.json" \
  # --num_inference_steps 50 \
  # --n_images_to_generate_for_each_prompt 4
  
  # accelerate launch --use_deepspeed --zero_stage=2 --gradient_accumulation_steps=1 --offload_param_device=cpu --offload_optimizer_device=cpu tool.py \
  # 
  # --skip_training_for_debug \
  # --instance_data_dir=$INSTANCE_DIR \
  #--class_data_dir=$CLASS_DIR \
  #--output_dir=$OUTPUT_DIR \
  #--save_sample_prompt="photo of sks dog" \
  #--instance_prompt="a photo of sks dog" \
  #--class_prompt="a photo of dog"