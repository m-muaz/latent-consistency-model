
import os
from pathlib import Path

# Import pytorch, diffusers, accelerate, transformers
import torch

from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel
)


# Custom main function
def main():
    # pretrained stable diffusion checkpoint path
    pretrained_checkpoint_model = "/home/yifanyang/stable-diffusion-xl-base-1.0"   

    
    # Load VAE from SD-XL checkpoint 
    vae = AutoencoderKL.from_pretrained(
        pretrained_checkpoint_model,
        subfolder="vae",
        )

    


# Main function
if __name__ == "__main__":
    main()