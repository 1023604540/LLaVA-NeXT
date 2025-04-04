from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
import math

#import memory module
from memory import FIFOMemory
from memory import KMeansMemory



print("load model")
warnings.filterwarnings("ignore")
# Load the OneVision model
pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"   # Use this for 7B model

# pretrained = "/hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/checkpoints/llava-onevision-correct_memory_adapter/checkpoint-6000"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)

save_dir = "/hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/checkpoints/custom_model_0.5b"
special_tokens_dict = {'additional_special_tokens': ['<memory_sep>']}
tokenizer.add_special_tokens(special_tokens_dict)
sep_token_id = tokenizer.convert_tokens_to_ids('<memory_sep>')
print("SEP token ID:", sep_token_id)

model.resize_token_embeddings(len(tokenizer))
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
image_processor.save_pretrained(save_dir)

