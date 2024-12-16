import os
import torch

from diffusers import UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel

from stable_hair_sdxl.controlnet import StableHairControlNetModel
from stable_hair_sdxl.pipeline_controlnet import StableHairSDXLControlNetPipeline


PRETRAINED_MODEL_PATH = "stabilityai/stable-diffusion-xl-base-1.0"
BALD_MODEL_PATH = os.getenv("MODEL_PATH")
device = "cuda" if torch.cuda.is_available else "cpu"


unet = UNet2DConditionModel.from_pretrained(PRETRAINED_MODEL_PATH, subfolder="unet").to(
    device
)
bald_converter = StableHairControlNetModel.from_unet(unet).to(device)
_state_dict = torch.load(BALD_MODEL_PATH)
bald_converter.load_state_dict(_state_dict, strict=False)
del unet


remove_hair_pipeline = StableHairSDXLControlNetPipeline.from_pretrained(
    PRETRAINED_MODEL_PATH,
    controlnet=bald_converter,
    safety_checker=None,
)
remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(
    remove_hair_pipeline.scheduler.config
)
remove_hair_pipeline = remove_hair_pipeline.to(device)
