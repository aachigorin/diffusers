import os

from diffusers import StableDiffusionPipeline


class DreamBoothGenerator:
    def __init__(self, model_path):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path).to("cuda")

    def generate(self, prompt, n_images, save_dir):
        images = self.pipe(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=n_images)
        for idx, image in enumerate(images):
            image.save(os.path.join(save_dir, f"sample_{idx}.png"))