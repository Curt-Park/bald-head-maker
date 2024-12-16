import argparse
import torch

from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image

from stable_hair_sdxl.controlnet import StableHairControlNetModel
from stable_hair_sdxl.pipeline_controlnet import StableHairSDXLControlNetPipeline


parser = argparse.ArgumentParser(description='스테이블 디퓨전 헤어 변환 파이프라인')
parser.add_argument(
    '--pretrained-model-path',
    type=str,
    default="stabilityai/stable-diffusion-xl-base-1.0",
)
parser.add_argument(
    '--controlnet-model-path',
    type=str,
    required=True,
)
parser.add_argument(
    '--input-image',
    type=str,
    required=True,
)
args = parser.parse_args()
device = "cuda" if torch.cuda.is_available else "cpu"


bald_converter = StableHairControlNetModel.from_pretrained(args.controlnet_model_path).to(device)
remove_hair_pipeline = StableHairSDXLControlNetPipeline.from_pretrained(
    args.pretrained_model_path,
    controlnet=bald_converter,
    safety_checker=None,
    torch_dtype=torch.float16,
).to(device)

remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(
    remove_hair_pipeline.scheduler.config
)
remove_hair_pipeline.enable_model_cpu_offload()
control_image = load_image(args.input_image).resize((1024, 1024))

# generate image
generator = torch.manual_seed(0)
image = remove_hair_pipeline("", num_inference_steps=20, generator=generator, image=control_image).images[0]
image.save("./output.png")