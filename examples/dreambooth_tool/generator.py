import os
import torch

from diffusers import StableDiffusionPipeline, DDIMScheduler


class DreamBoothGenerator:
    def __init__(self, model_path):
        #self.pipe = StableDiffusionPipeline.from_pretrained(model_path)        
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        #self.pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16)
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, safety_checker=None)
        if torch.cuda.is_available():
            self.pipe.to("cuda")


    def generate(self, prompt, n_images, save_dir, num_inference_steps=50):
        images = self.pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=7.5,
                           num_images_per_prompt=n_images).images            
        for idx, image in enumerate(images):
            image.save(os.path.join(save_dir, f"sample_{idx}.png"))