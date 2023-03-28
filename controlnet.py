from typing import List

import torch.nn as nn
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector
from diffusers.utils import load_image


class ControlNet(nn.module):
    """
    """
    def __init__(self, device, cond_type='scribble'):
        self.device = device
        self.cond_type = cond_type

    def get_text_embeds(self, prompt: List[str], negative_prompt: List[str]):
        pass

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100):
        pass
