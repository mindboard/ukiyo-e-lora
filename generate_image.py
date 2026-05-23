
import argparse

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import LineartDetector
from PIL import Image
import torch

parser = argparse.ArgumentParser(description="Generate an ukiyo-e styled image from a lineart input.")
parser.add_argument("input", help="Path to the input lineart image.")
parser.add_argument("output", help="Path to save the generated image.")
args = parser.parse_args()

ctrl = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_lineart", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=ctrl, torch_dtype=torch.float16,
).to("cuda")

generator = torch.Generator(device="cuda").manual_seed(43)

pipe.load_lora_weights("./output/lora_ukiyoe/step-200", weight_name="pytorch_lora_weights.safetensors")
pipe.fuse_lora(lora_scale=0.6)

lineart = Image.open(args.input).convert("RGB").resize((512, 512))
img = pipe(
    prompt="ukiyoe_style, a beautiful girl in kimono standing under cherry blossoms, majestic mount fuji in the background, spring landscape, masterpiece",
    negative_prompt="blurry, low quality, photo, 3d",
    image=lineart,
    num_inference_steps=40,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.9,
    generator=generator,
).images[0]

img.save(args.output)

