import argparse
import json
import os
from pathlib import Path
import hashlib
import torch
import copy

from trainer import DeepBoothTrainer
from generator import DreamBoothGenerator


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        #default="fp16",
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    # parser.add_argument(
    #     "--save_sample_prompt",
    #     type=str,
    #     default=None,
    #     help="The prompt used to generate sample outputs to save.",
    # )
    # parser.add_argument(
    #     "--save_sample_negative_prompt",
    #     type=str,
    #     default=None,
    #     help="The negative prompt used to generate sample outputs to save.",
    # )
    parser.add_argument(
        "--n_save_sample",
        type=int,
        default=4,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--save_guidance_scale",
        type=float,
        default=7.5,
        help="CFG for save sample.",
    )
    parser.add_argument(
        "--save_infer_steps",
        type=int,
        default=50,
        help="The number of inference steps for save sample.",
    )
    parser.add_argument(
        "--pad_tokens",
        default=False,
        action="store_true",
        help="Flag to pad tokens to length 77.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="text-inversion-model",
    #     help="The output directory where the model predictions and checkpoints will be written.",
    # )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--save_interval", type=int, default=10_000, help="Save weights every N steps.")
    parser.add_argument("--save_min_steps", type=int, default=0, help="Start saving weights after N steps.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--not_cache_latents", action="store_true", help="Do not precompute and cache latents from VAE.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # parser.add_argument(
    #     "--concepts_list",
    #     type=str,
    #     default=None,
    #     help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    # )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to json containing multiple queries for dreambooth generation",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of step for generator"
    )
    parser.add_argument(
        "--skip_training", action='store_true', help="Skip training step to speed up debugging"
    )
    parser.add_argument(
        "--n_images_to_generate_for_each_prompt", type=int, default=4, help=""
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main(args):
    # parser.add_argument(
    #     "--instance_data_dir",
    #     type=str,
    #     default=None,
    #     help="A folder containing the training data of instance images.",
    # )
    # parser.add_argument(
    #     "--class_data_dir",
    #     type=str,
    #     default=None,
    #     help="A folder containing the training data of class images.",
    # )
    # parser.add_argument(
    #     "--instance_prompt",
    #     type=str,
    #     default=None,
    #     help="The prompt with identifier specifying the instance",
    # )
    # parser.add_argument(
    #     "--class_prompt",
    #     type=str,
    #     default=None,
    #     help="The prompt to specify images in the same class as provided instance images.",
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default="text-inversion-model",
    #     help="The output directory where the model predictions and checkpoints will be written.",
    # )
    # parser.add_argument(
    #     "--concepts_list",
    #     type=str,
    #     default=None,
    #     help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    # )
    # parser.add_argument(
    #     "--save_sample_prompt",
    #     type=str,
    #     default=None,
    #     help="The prompt used to generate sample outputs to save.",
    # )
    # parser.add_argument(
    #     "--save_sample_negative_prompt",
    #     type=str,
    #     default=None,
    #     help="The negative prompt used to generate sample outputs to save.",
    # )

    with open(args.config_path) as f:
        config = json.load(f)

    global_results_json = {
        'output_dir': config['output_dir'],
        'queries': []
    }
    for idx, query in enumerate(config["queries"]):
        trainer = DeepBoothTrainer()
        args.concepts_list = query['concepts']
        instance_prompts = '_'.join([concept['instance_prompt'] for concept in query['concepts']])
                
        str_args = argparse.Namespace(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                      pretrained_vae_name_or_path=args.pretrained_vae_name_or_path,
                                      tokenizer_name=args.tokenizer_name,
                                      with_prior_preservation=args.with_prior_preservation,
                                      learning_rate=args.learning_rate
                                     )
        instance_prompt_args_md5 = hashlib.md5(instance_prompts.encode()).hexdigest()
        instance_prompt_args_md5 += '_' + hashlib.md5(str(str_args).encode()).hexdigest()
        
        output_dir = Path(config['output_dir']) / instance_prompt_args_md5
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(output_dir)
        if not args.skip_training:
            trainer.train(args)
        del trainer
        torch.cuda.empty_cache() # trying to release the memory as much as possible

        global_results_json['args'] = str(args)
        query_copy = copy.deepcopy(query)
        query_copy['prompts_results'] = []
        model_dir = output_dir / str(args.max_train_steps)
        generator = DreamBoothGenerator(model_dir)
        for prompt in query['prompts']:
            local_results_json = dict()
            local_results_json['concepts'] = query['concepts']
            local_results_json['args'] = str(args)
            local_results_json['prompt'] = prompt
            prompt_md5 = 'prompt_' + hashlib.md5(prompt.encode()).hexdigest()
            cur_output_dir = output_dir / prompt_md5
            query_copy['prompts_results'].append({'prompt': prompt, 'results_dir': str(cur_output_dir)})
            cur_output_dir.mkdir(parents=True, exist_ok=True)            
            generator.generate(prompt, n_images=args.n_images_to_generate_for_each_prompt,
                               save_dir=cur_output_dir, num_inference_steps=args.num_inference_steps)
            with open(cur_output_dir / 'results.json', 'w+') as f:
                json.dump(local_results_json, f)

        global_results_json['queries'].append(query_copy)
        with open(Path(config['output_dir']) / 'results.json', 'w+') as f:
            json.dump(global_results_json, f)

        del generator
        torch.cuda.empty_cache() # trying to release the memory as much as possible

if __name__ == "__main__":
    args = parse_args()
    main(args)