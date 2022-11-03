from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

#scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

model_id = "/stor/a.chigorin/projects/deepbooth/saved_models/200/"

#pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16).to("cuda")

#pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")

prompt = "A photo of sks dog in a bucket"
#image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, num_images_per_prompt=1).images[0]
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("/stor/a.chigorin/projects/deepbooth/saved_models/generated_examples/dog-bucket.png")
