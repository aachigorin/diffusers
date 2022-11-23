import dominate
from dominate.tags import *

import argparse
import json
from pathlib import Path
import base64
import cv2

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--results_path",
        type=str,
        default=None,
        required=True,
        help="Path to the directory with results.json files",
    )
    parser.add_argument(
        "--n_cols",
        type=int,
        default=6
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=200
    )
    parser.add_argument(
        "--html_path",
        type=str,
        default=None,
        required=False,
        help="Path to the html file with visualization results",
    )
    args = parser.parse_args()
    if args.html_path is None:
        args.html_path = str(Path(args.results_path).parents[0] / 'visualization.html')
    return args

def table_with_images(img_paths, n_cols=4, w=200, h=200, n_max=8):
    with table():
        for idx, path in enumerate(img_paths):
            if idx >= n_max:
                break
            if idx % n_cols == 0:
                row = tr()
            #row += td(img(src=path, width=w, height=h))
            image = cv2.imread(str(path))
            td(img(src=f"data:image/jpeg;base64,{image_to_base64(image)}", width=w, height=h))


def read_img_paths(dir_path):
    img_paths = []
    for ext in ['jpg', 'jpeg', 'png']:
        img_paths += Path(dir_path).glob(f'*.{ext}')
    return img_paths

def image_to_base64(image):
    return base64.b64encode(cv2.imencode('.jpg', image)[1]).decode('utf-8')

def main(args):
    with open(args.results_path) as f:
        results_json = json.load(f)
    #output_dir = results_json['output_dir']
    #results_paths = Path(output_dir).glob('**/results.json')

    doc = dominate.document(title='')
    with doc:
        for query in results_json['queries']:
            instance_images = []
            class_images = []
            instance_prompts = []
            class_prompts = []
            for concept in query['concepts']:
                instance_images = read_img_paths(concept['instance_data_dir'])
                class_images = read_img_paths(concept['class_data_dir'])
                instance_prompts.append(concept['instance_prompt'])
                class_prompts.append(concept['class_prompt'])

            h2('Instance prompts:')
            for prompt in instance_prompts:
                p(prompt)
            h2('Instance images:')
            table_with_images(instance_images, n_cols=args.n_cols, w=args.img_size, h=args.img_size)

            h2('Class prompts:')
            for prompt in class_prompts:
                p(prompt)
            h2('Class images:')
            table_with_images(class_images, n_cols=args.n_cols, w=args.img_size, h=args.img_size, n_max=12)

            h2('Prompts results:')
            for prompt_res in query['prompts_results']:
                p(prompt_res["prompt"])
                prompt_images = read_img_paths(prompt_res['results_dir'])
                table_with_images(prompt_images, n_cols=args.n_cols, w=args.img_size, h=args.img_size, n_max=12)


    with open(args.html_path, 'w+') as f:
        f.write(str(doc))
        print(f'Html is saved into {args.html_path}')


if __name__ == "__main__":
    args = parse_args()
    main(args)