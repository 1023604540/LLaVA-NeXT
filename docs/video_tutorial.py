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
# pretrained = "/hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/checkpoints/llava-onevision-0.5b-qwen2_KIT_position_8tokens_initial/checkpoint-300"
# pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"   # Use this for 7B model
pretrained = "/anvme/workspace/b232dd16-LLaVA-OV/llava-onevision-qwen2-0.5b-ov"   # Use this for 7B model
# pretrained = "/hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/checkpoints/llava-onevision-qwen2-0.5b-ov"   # Use this for 7B model
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="flash_attention_2", **llava_model_args)

# for idx, (name, param) in enumerate(model.named_parameters()):
#     print(idx, name, param.shape)

model.eval()


# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

def load_full_video(video_path):
    # Using this would give you too many frames, leading to cuda OOM
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    frame_idx = [i for i in range(0, total_frame_num)]
    dense_frames = vr.get_batch(frame_idx).asnumpy()
    return dense_frames  # (frames, height, width, channels)


def load_sampled_video(video_path, sample_fps=10):
    # 初始化 VideoReader 对象
    vr = VideoReader(video_path, ctx=cpu(0))

    # 获取视频的原始帧率
    original_fps = vr.get_avg_fps()

    # 计算采样间隔
    sample_interval = int(original_fps // sample_fps)

    # 获取视频的总帧数
    total_frame_num = len(vr)

    # 生成需要采样的帧索引列表
    frame_idx = list(range(0, total_frame_num, sample_interval))

    # 获取采样的帧
    sampled_frames = vr.get_batch(frame_idx).asnumpy()

    return sampled_frames  # 返回采样的帧 (frames, height, width, channels)


def dynamic_load_video(video_path):
    # 初始化 VideoReader 对象
    vr = VideoReader(video_path, ctx=cpu(0))

    # 获取视频总帧数和原始帧率
    total_frame_num = len(vr)
    original_fps = vr.get_avg_fps()

    # 计算视频时长（秒）
    duration = total_frame_num / original_fps
    if total_frame_num < 10:
        # 如果视频总帧数不足10帧，补足到10帧
        frame_idx = list(range(total_frame_num)) + [total_frame_num - 1] * (10 - total_frame_num)
    elif total_frame_num < 100:
        # 如果视频总帧数不足100帧，则直接返回所有帧
        frame_idx = list(range(total_frame_num))
    elif duration >= 100:
        # 长视频：每秒采样1帧
        interval = int(original_fps)  # 每秒采一帧
        frame_idx = list(range(0, total_frame_num, interval))
    else:
        # 短视频：计算每秒需要采样的帧数，以确保采样总帧数不少于100
        effective_sample_rate = math.ceil(100 / duration)
        # 计算采样间隔（每隔多少帧采一帧），保证至少采样每一帧
        interval = max(1, int(original_fps / effective_sample_rate))
        frame_idx = list(range(0, total_frame_num, interval))

    # 获取采样的帧
    sampled_frames = vr.get_batch(frame_idx).asnumpy()
    return sampled_frames

print("load video")
# Load and process video
video_path = "docs/jobs.mp4"
# video_path = "/hkfs/work/workspace/scratch/tum_tyz7686-LLaVA-OV/QJ6sjg7SXOQ.mp4"
# video_frames = load_video(video_path, 64)
video_frames = load_video(video_path, 128)
print(video_frames.shape) # (16, 1024, 576, 3)
image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
print(frames.shape) # torch.Size([16, 3, 384, 384])
image_tensors.append(frames)



# Prepare conversation input
conv_template = "qwen_1_5"

# question = f"{DEFAULT_IMAGE_TOKEN}\n The screen is split into two sections, and in the small section on the far right, what is the man wearing a hat doing in front of a brown horse?\nA. Walking the horse\nB. Punching towards the camera\nC. Riding the horse\nD. Kneeling down\nE. Extending his palm forward while facing the camera\nAnswer with the option's letter from the given choices directly.\n"
question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe what's happening in this video."
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
print(prompt_question)
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [frame.size for frame in video_frames]  # (width * height * 3)


# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=1024,
    modalities=["video"],
    repetition_penalty=1.2,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])

