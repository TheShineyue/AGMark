import json
import os
import argparse
from tqdm import tqdm
import time
import logging
import pandas as pd
import torch
from PIL import Image
from mark_utils import WatermarkUtils
# eggs
def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Experiments parameters')
    aa("--json_path", type=str, default="input/")
    aa("--image_dir", type=str, default="input/")
    aa("--model_path", type=str, default="input/")
    aa("--range_num", type=int, default=100)
    aa("--model_name", type=str, default="llava")
    aa("--task_name", type=str, default="AMBER")
    aa("--data_suffix", type=str, default=".jpg")
    aa("--similarity_scheme", type=str, default="cosine")
    aa("--max_tokens", type=int, default=128)
    aa("--min_tokens", type=int, default=40)
    aa("--save_res_dir",type=str,default=None)
    aa("--atten_delta", type=float,default = 0.02)
    aa("--unwater_flag", action='store_false')
    aa("--unwater_dir", type =str, default = None)
    aa("--alpha", type =float, default = 0.4)
    aa("--sct_radio", type=float,default = 0.025)
    aa("--trans_type", type =str, default = None) 
    aa("--attn_threshold", type =float, default = 0.15)
    return parser

def refresh_logits_processor(model, logits_processor, inputs, utils=None, image_pos= None, threshold= 0):
    """刷新logits processor用于生成带水印文本"""
    logits_processor.set_img_pos(image_pos)
    feature = utils.get_feature(model, inputs, image_pos)
    atten_scores, hidden_states = utils.get_only_image_atten(model, inputs, image_pos) # , threshold = threshold
    logits_processor.set_attention_scores(atten_scores)
    logits_processor.set_hidden_states(hidden_states)
    logits_processor.refresh_msg(feature)
    logits_processor.b_list = []
    return logits_processor

def get_image_pos(inputs, model, model_name):
    if model_name == 'qwen3':
        start_idx = (
            inputs['input_ids'][0] == model.config.vision_start_token_id
        ).nonzero(as_tuple=True)[0][0].item()
        end_idx = (
            inputs['input_ids'][0] == model.config.vision_end_token_id
        ).nonzero(as_tuple=True)[0][0].item()
    return (start_idx, end_idx)

def pipeline(utils, params):
    df = pd.read_json(params.json_path)

    model, processor = utils.init_model()
    logits_processor = utils.init_logits_processor(model, processor)
    for i in tqdm(range(min(params.range_num, len(df)))):
        row = df.iloc[i]
        if 'image' in row:
            question = row['query']
            image_id = row['image']
            print(row['image'])
            image_path = os.path.join(params.image_dir, image_id)
            image = Image.open(image_path)
            image = image.resize((336, 336), Image.Resampling.LANCZOS)
            inputs = utils.get_inputs(processor, image, question)
        
        image_pos = get_image_pos(inputs, model, params.model_name)
        # print(image_pos)
        # assert 1==0

        with torch.no_grad():
            # refreshed_logits_processor = refresh_logits_processor(model, logits_processor, inputs, utils, image_pos, params.attn_threshold)
            unwatermarked_response = utils.custom_generate_responses(
                    model, processor.tokenizer, inputs, None)
            torch.cuda.empty_cache()
            
            refreshed_logits_processor = refresh_logits_processor(model, logits_processor, inputs, utils, image_pos, params.attn_threshold)

            watermarked_response = utils.custom_generate_responses(
                model, processor.tokenizer, inputs, 
                refreshed_logits_processor
            ) 
            torch.cuda.empty_cache()

            print(unwatermarked_response, watermarked_response)
    del model

def main(params):
    logging.info(f"CUDA is available: {torch.cuda.is_available()}")
    utils = WatermarkUtils(params)
    pipeline(utils, params)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = get_parser()
    params = parser.parse_args()
    
    main(params)
