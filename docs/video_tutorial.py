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
pretrained = "/anvme/workspace/b232dd16-LLaVA-OV/checkpoints/llava-onevision-google_siglip-so400m-patch14-384-Qwen_Qwen2-7B-Instruct-memory_adapter_2nd"   # Use this for 7B model
# pretrained = "/anvme/workspace/b232dd16-LLaVA-OV/llava-onevision-qwen2-7b-ov"   # Use this for 7B model
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

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
video_path = "/home/hpc/b232dd/b232dd16/LLaVA-OV/docs/jobs.mp4"
# video_frames = load_video(video_path, 32)
video_frames = load_video(video_path, 32)
print(video_frames.shape) # (16, 1024, 576, 3)
image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
print(frames.shape) # torch.Size([16, 3, 384, 384])
image_tensors.append(frames)

# memory_inserted = False
# fifo = False
# kmeans = False
# if memory_inserted:
#     ##### Insert memory module #####
#     print(len(image_tensors)) # 1
#     print(f"Shape: {image_tensors[0].shape}, Dtype: {image_tensors[0].dtype}")  # Shape: torch.Size([16, 3, 384, 384]), Dtype: torch.float16
#     image_tensors = torch.cat(image_tensors, dim=0)
#     if fifo:
#         fifo_memory = FIFOMemory(max_size=10, tensor_shape=(3, 384, 384), device=device)
#         fifo_memory.add_tensor(image_tensors)
#         fifo_output = fifo_memory.get_tensors()
#
#     if kmeans:
#         kmeans_memory = KMeansMemory(max_size=10, tensor_shape=(3, 384, 384), device=device)
#         kmeans_memory.add_tensor(image_tensors)
#         kmeans_output = kmeans_memory.memory
#
#     image_tensors = (fifo_output + kmeans_output) if fifo and kmeans else (
#         fifo_output if fifo else kmeans_output if kmeans else image_tensors)
#     image_tensors = [image_tensors.to(dtype=torch.float16)]
#
#     print(f"Shape: {image_tensors[0].shape}, Dtype: {image_tensors[0].dtype}")
    ##### Insert memory module #####



# Prepare conversation input
conv_template = "qwen_1_5"

question = f"{DEFAULT_IMAGE_TOKEN}\nDescribe all the content of the video."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [frame.size for frame in video_frames]  # (width * height * 3)


# Your custom hidden-state modification
def modify_hidden_state(hidden_states):
    # For instance: scale hidden states or add noise
    return hidden_states * 1  # Example scaling

# Choose the layer you want to hook into
layer_to_hook = -2  # second-to-last layer as an example

# Hook function
def hidden_state_hook(module, input, output):
    modified_output = modify_hidden_state(output[0])
    return (modified_output,) + output[1:]

# Register the hook before inference/generation
hook_handle = model.model.layers[layer_to_hook].register_forward_hook(hidden_state_hook)

# Cleanup: Remove the hook after you're done
hook_handle.remove()





# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=1024,
    modalities=["video"],
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])

