import os
import argparse

import pandas as pd
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration, LogitsProcessor, LogitsProcessorList
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers.generation.configuration_utils import GenerationConfig
import json
from tqdm import tqdm
import time
from custom_topp import VLALogitsProcessor, VLA
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
import warnings
import logging
class WatermarkUtils:
    def __init__(self, params):
        self.config = params
        self._janus_attention_mask = None
    def init_model(self):
        model_map = {
            "qwen3": "Qwen3-VL-8B-Instruct"
        }
        if self.config.model_name not in model_map:
            raise ValueError(f"Invalid model name: {self.config.model_name}")
        model_path = self.config.model_path
        if self.config.model_name == "qwen3":
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, 
                device_map="auto", 
                attn_implementation="eager"
            )
            processor = AutoProcessor.from_pretrained(model_path)
        else:
            raise ValueError(f"Invalid model name: {self.config.model_name}")
        
