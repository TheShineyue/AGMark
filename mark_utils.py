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
        return model, processor
    
    def init_logits_processor(self, model, processor): 
        special_tokens = [
            processor.tokenizer.eos_token_id, processor.tokenizer.bos_token_id
        ] if processor.tokenizer.pad_token_id is None else [
            processor.tokenizer.eos_token_id, processor.tokenizer.bos_token_id, processor.tokenizer.pad_token_id]
        embedding_matrix = model.get_input_embeddings().weight
        
        if embedding_matrix.device.type != "cuda":
            embedding_matrix = embedding_matrix.to("cuda")
        kwargs = {
            "embedding_matrix": embedding_matrix,
            "vocab_size": processor.tokenizer.vocab_size,
            "special_tokens": special_tokens,
            "similarity_scheme": self.config.similarity_scheme,
            "split_x": getattr(self.config, 'split_x', 2),
            "atten_delta" : self.config.atten_delta,
            "sct_radio" : self.config.sct_radio,
            "alpha" : self.config.alpha,
            "trans_type" : self.config.trans_type,
            "attn_threshold" : self.config.attn_threshold
        }
        
        return VLALogitsProcessor(**kwargs)
    
    def init_detector(self, model, processor):
        embedding_matrix = model.get_input_embeddings().weight
        kwargs = {
            "processor": processor,
            "embedding_matrix": embedding_matrix,
            "is_hard": False,
            "similarity_scheme": self.config.similarity_scheme,
            "atten_delta" : self.config.atten_delta,
            "sct_radio" : self.config.sct_radio,
            "alpha" : self.config.alpha,
            "trans_type" : self.config.trans_type,
            "attn_threshold" : self.config.attn_threshold
        }
        kwargs.update({
            "model": model,
            "model_name": self.config.model_name
        })
        return VLA(**kwargs)
    
    def _custom_qwen_responses(self, model, processor, inputs, logits_processor=None) -> str:
        inputs = inputs.to("cuda")
        generation_config = {
            "max_new_tokens": self.config.max_tokens,
            "min_new_tokens": self.config.min_tokens,
            "temperature": 0,
            "do_sample": False,
            "num_beams": 1,
        }
        if logits_processor is not None: 
            generation_config["logits_processor"] = LogitsProcessorList([logits_processor])
        outputs = model.generate(
            **inputs, **generation_config)
        origin_generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        if "ASSISTANT:" in origin_generated_text:
            generated_text = origin_generated_text.split("ASSISTANT:")[1].strip()
        if "\nassistant\n" in origin_generated_text:
            generated_text = origin_generated_text.split("\nassistant\n")[1].strip()
        return generated_text    
    
    def get_only_image_atten(self, model, inputs, image_pos):
        with torch.no_grad():
            outputs = model(
                **inputs,
                return_dict=True,
                output_hidden_states=True,
                output_attentions=True
            )
            last_token_attn = outputs.attentions[-1][:, :, -1, :]
            attention_scores = last_token_attn.mean(dim=1)
            hidden_states = outputs.hidden_states[-1][:, -1, :]
        return attention_scores, hidden_states    
    
    def get_inputs(self, processor, image, question):
        if self.config.model_name in ["qwen", "qwen3", "qwen3_4b", "llava_next", "internvl3_5"]:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=[text_prompt], images=[image], padding=True, return_tensors="pt"
            ).to("cuda")
        else:
            prompt = f"USER: <image>\n{question} ASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
        return inputs
    
    def get_all_inputs(self, processor, image, question, response):
        if self.config.model_name in ["qwen", "qwen3", "qwen3_4b"]:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{question}"},
                    ],
                },
                {
                    "role": "assistant",
                    "content" : [{"type":"text", "text": response}]
                }
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=[text_prompt], images=[image], padding=True, return_tensors="pt"
            ).to("cuda")
        return inputs    
    
    def custom_generate_responses(self, model, processor, inputs, logits_processor=None) -> str:
        response_map = {
            "qwen3" : self._custom_qwen_responses,
        }
        response_func = response_map.get(self.config.model_name, None) 
        return response_func(model, processor, inputs, logits_processor)
    
    def _image_feature(self, model, inputs, image_pos=None):
        if self.config.model_name in ["qwen3", "qwen3_4b"]:
            image_feature, deepstack_image_embeds = model.model.get_image_features(
                pixel_values = inputs['pixel_values'],
                image_grid_thw = inputs['image_grid_thw'],
            )
            return image_feature[0]
        else:
            raise ValueError(f"Invalid model name: {self.config.model_name}")

    def get_feature(self, model, inputs, image_pos):
        return self._image_feature(model, inputs, image_pos)
