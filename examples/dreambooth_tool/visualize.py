from html4vision import Col, imagetable

import argparse
import json
from pathlib import Path

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        required=True,
        help="Path to the directory with results.json files",
    )
    parser.add_argument(
        "--max_num_results",
        type=int,
        default=50,
        help="Maximum number of resuling concepts visualizations (instances)",
    )
    parser.add_argument(
        "--max_num_prompts",
        type=int,
        default=10,
        help="Maximum number of prompts for one resulting instance",
    )
    args = parser.parse_args()
    return args


def main(args):
    results_paths = Path(args.results_dir).glob('**/results.json')
    for path in results_paths:
        instance_prompts = []
        class_prompts = []
        instance_images_paths = []
        class_images_paths = []
        with open(path) as f:
            r_json = json.load(f)
        for concept in r_json['concepts']:
            instance_prompts += concept['instance_prompt']
            class_prompts += concept['class_prompt']
            for ext in ['*.jpg', '*.jpeg']:
                instance_images_paths += Path(concept['instance_data_dir']).glob(f'*.{ext}')
                class_images_paths += Path(concept['class_data_dir']).glob(f'*.{ext}')
        sample_images_paths = path.parent.glob('*.png')
        prompt = r_json['prompt']


if __name__ == "__main__":
    args = parse_args()
    main(args)
