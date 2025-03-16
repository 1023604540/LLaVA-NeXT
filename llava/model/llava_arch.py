#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
from llava.model.memory_module.memory_builder import NeuralTuringMachine, MultimodalOpsMixin
from llava.model.memory_module.segment import segment, adjusted_segment
import heapq
import numpy as np


################################################################
# Llava OneVision config
  # "mm_newline_position":"one_token",
  # "attention_dropout": 0.0,
  # "bos_token_id": 151643,
  # "eos_token_id": 151645,
  # "hidden_act": "silu",
  # "hidden_size": 3584,
  # "image_token_index": 151646,
  # "image_aspect_ratio": "anyres_max_9",
  # "image_crop_resolution": null,
  # "image_grid_pinpoints": ...,
  # "image_split_resolution": null,
  # "initializer_range": 0.02,
  # "intermediate_size": 18944,
  # "max_position_embeddings": 32768,
  # "max_window_layers": 28,
  # "mm_hidden_size": 1152,
  # "mm_patch_merge_type": "spatial_unpad",
  # "mm_projector_lr": null,
  # "mm_projector_type": "mlp2x_gelu",
  # "mm_resampler_type": null,
  # "mm_spatial_pool_mode": "bilinear",
  # "mm_tunable_parts": "mm_vision_tower,mm_mlp_adapter,mm_language_model",
  # "mm_use_im_patch_token": false,
  # "mm_use_im_start_end": false,
  # "mm_vision_select_feature": "patch",
  # "mm_vision_select_layer": -2,
  # "mm_vision_tower": "google/siglip-so400m-patch14-384",
  # "mm_vision_tower_lr": 2e-06,
  # "model_type": "llava",
  # "num_attention_heads": 28,
  # "num_hidden_layers": 28,
  # "num_key_value_heads": 4,
  # "pos_skipping_range": 4096,
  # "rms_norm_eps": 1e-06,
  # "rope_scaling": null,
  # "rope_theta": 1000000.0,
  # "sliding_window": 131072,
  # "tie_word_embeddings": false,
  # "tokenizer_model_max_length": 32768,
  # "tokenizer_padding_side": "right",
  # "torch_dtype": "bfloat16",
  # "transformers_version": "4.40.0.dev0",
  # "use_cache": true,
  # "use_mm_proj": true,
  # "use_pos_skipping": false,
  # "use_sliding_window": false,
  # "vision_tower_pretrained": null,
  # "vocab_size": 152064
################################################################
class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

        self.mm_input_dim = getattr(config, "ntm_hidden_size", 1152)
        compress_Turing_hidden_dim = getattr(config, "compress_Turing_hidden_dim", 32)
        # Now init in memory_builder
        self.attention_model = NeuralTuringMachine(self.mm_input_dim, compress_Turing_hidden_dim).to(self.device)
        self.memory_mlp = nn.Sequential(
            nn.Linear(1152, 1152),
            nn.GELU(),
            nn.Linear(1152, 1152),
        ).to(self.device)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type


        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor



class LlavaMetaForCausalLM(MultimodalOpsMixin, ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            # [num_frame, channels, height, width]->[num_frame, channels, height/2, width/2]
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1) # [num_frame, height/2, width/2, channels]
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        # image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):

            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat != 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()  # [3584, frame_num, 196]
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)  # [3584, frame_num, 197]
        image_feature = image_feature.permute(1, 2, 0).contiguous()  # [frame_num, 197, 3584]
        return image_feature


    def uniform_sample_frames(self, tensor, num_samples=32):
        """
        Uniformly samples frames from a 4D tensor.

        Args:
            tensor (torch.Tensor): Input tensor of shape (F, P, D)
            num_samples (int): Number of frames to sample

        Returns:
            torch.Tensor: Sampled tensor of shape (num_samples, P, D)
        """
        frame_num = tensor.shape[0]  # Total frames
        if frame_num > num_samples:
            indices = torch.linspace(0, frame_num - 1, num_samples).long()  # Uniformly spaced indices
            return tensor[indices]
        else:
            return tensor



    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):

        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            # Initialize lists to collect non-video images and their original indices.
            non_video_images = []
            non_video_positions = []
            boundary_list = []

            for idx, image in enumerate(images_list):
                # If it is not a video feature, we don't need to process it
                if idx not in video_idx_in_batch:
                    non_video_images.append(image)
                    non_video_positions.append(idx)
                    continue
                # boundaries = adjusted_segment(image.mean(dim=1).flatten(1,2))
                boundaries = segment(image.mean(dim=1).flatten(1, 2))

                #print(f"boundaries:{len(boundaries)}")
                #print(f"boundaries:{boundaries}")

                segment_memory = []
                if image.shape[0] >= 256:
                    print(f"Image shape : {image.shape}")
                    seperate_video = []
                    # split the image feature and encode them separately
                    for i in range(0, image.shape[0], 256):
                        encoded_segment = self.encode_images(image[i:i + 256])
                        print(f"Encoded segment shape : {encoded_segment.shape}")
                        seperate_video.append(encoded_segment)
                    encoded_features = torch.cat(seperate_video, dim=0)
                else:
                    encoded_features = self.encode_images(image)
                # encoded_features = encoded_features.requires_grad_()
                # print(
                #     f"[DEBUG] Vision output requires_grad={encoded_features.requires_grad}, grad_fn={encoded_features.grad_fn}")
                # torch.cuda.synchronize()
                # print("Before attention_model forward pass")
                image_segments = [encoded_features[boundaries[i]:boundaries[i+1]] for i in range(len(boundaries) - 1)]
                memory_boundary = [0]
                current_segment = 0
                for image_segment in image_segments:
                    #print(f"Image segment shape : {image_segment.shape}")
                    #print(f"Encoded segment shape : {encoded_segment.shape}")
                    # segment_memory += (self.compress_temporal_features([image_segment], video_idx_in_batch, all_video=True))
                    segment_memory += [image_segment]
                    current_segment += segment_memory[-1].shape[0]
                    memory_boundary.append(current_segment)
                boundary_list.append(memory_boundary)

                #print(f"Segment memory : {[x.shape for x in segment_memory if x is not None]}")
                # torch.cuda.synchronize()
                # print("After attention_model forward pass")

                cat_segment_memory = torch.cat([image for image in segment_memory], dim=0)
                rank0_print(f"cat_segment_memory shape : {cat_segment_memory.shape}")
                if torch.isnan(cat_segment_memory).any():
                    raise ValueError("NaNs detected in attention_model output!")
                # rank0_print(f"cat_segment_memory shape : {cat_segment_memory.shape}")
                # rank0_print(
                #     f"[attention_model] output requires_grad={cat_segment_memory.requires_grad}, grad_fn={cat_segment_memory.grad_fn}")
                images_list[idx] = cat_segment_memory

            # Now process all non-video images together.
            if non_video_images:
                # Record the original batch sizes of each non-video image tensor.
                original_lengths = [img.shape[0] for img in non_video_images]
                # Concatenate them along the batch dimension.
                concatenated = torch.cat(non_video_images, dim=0)
                # Encode the concatenated tensor.
                encoded = self.encode_images(concatenated)
                # Split the encoded tensor back into individual parts.
                splits = torch.split(encoded, original_lengths, dim=0)
                # Place the processed tensors back to their original positions.
                for pos, enc in zip(non_video_positions, splits):
                    images_list[pos] = enc

            # Apply mm_projector
            split_sizes = [image.shape[0] for image in images_list]
            projected_feature = self.get_model().mm_projector(torch.cat([image for image in images_list], dim=0))
            image_features = torch.split(projected_feature, split_sizes)
            rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")  # [frame_num, 729, 3584]

            new_image_features = []
            for idx, image_feat in enumerate(image_features):
                if idx in video_idx_in_batch:
                    new_image_features.append(self.get_2dPool(image_feat))
                else:
                    new_image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            #rank_print(f"Encoded image feats after 2dPool : {[x.shape for x in new_image_features]}")  # [frame_num, 196, 3584]
            # image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = new_image_features

            # Key Memory Selection Module
            for index, image_feature in enumerate(image_features):
                print(input_ids.shape)
                cur_input_ids = input_ids[index]
                print(cur_input_ids)

                ############################## Conversation Template
                # < | im_start | > system
                # You are a helpful assistant. < | im_end | >
                # < | im_start | > user
                # < image >
                # tell me what is going on in this video. < | im_end | >
                # < | im_start | > assistant
                ##############################
                IM_END_TOKEN_ID = 151645  # "<|im_end|>"
                IMAGE_TOKEN_ID = -200  # "<image>"

                def extract_user_query_tokens(input_ids, image_token_id=IMAGE_TOKEN_ID,
                                              im_end_token_id=IM_END_TOKEN_ID):
                    """
                    Extract tokens from the user message that come after the <image> token
                    and before the next <|im_end|> token.
                    """
                    # Ensure we work with a list of ints.
                    if isinstance(input_ids, torch.Tensor):
                        tokens = input_ids.tolist()
                    else:
                        tokens = input_ids

                    try:
                        # Find the first occurrence of the <image> token.
                        idx_image = tokens.index(image_token_id)
                        # Then find the next occurrence of the <|im_end|> token after the <image> token.
                        idx_im_end = tokens.index(im_end_token_id, idx_image)
                        # Extract tokens after the <image> token up to (but not including) the <|im_end|> token.
                        query_tokens = tokens[idx_image + 2: idx_im_end]  # Skip the <image> token and the space token.
                        query_tensor = torch.tensor(query_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
                        return query_tensor
                    except ValueError:
                        # If the expected tokens are not found, return an empty list.
                        return []

                query = extract_user_query_tokens(cur_input_ids)
                print(f"the query is : {query}")
                query_feature = self.get_model().embed_tokens(query)
                print(query_feature.shape)  # [1, n, 3584]

                def compute_frame_score(frame_feature, query_embedding, reduction='max'):
                    """
                    计算单个frame与query之间的相似度。

                    参数:
                        frame_feature: tensor, 形状为 (196, 3584)
                        query_embedding: tensor, 形状为 (token_num, 3584)
                        reduction: str, 归约方式，可以选择 'max' 或 'mean'

                    返回:
                        score: float, 该frame与query的相似度分数
                    """
                    # 对frame的patch特征进行归一化 (在最后一维)
                    frame_norm = F.normalize(frame_feature, dim=-1)  # (196, 3584)
                    # 对query的token特征进行归一化
                    query_norm = F.normalize(query_embedding, dim=-1)  # (token_num, 3584)

                    # 计算余弦相似度矩阵，结果形状为 (196, token_num)
                    # 每个元素表示某个patch与某个token之间的相似度
                    sim_matrix = torch.matmul(frame_norm, query_norm.T)

                    # 根据归约方式将 (196, token_num) 的相似度矩阵化为单一分数
                    if reduction == 'max':
                        score = sim_matrix.max()  # 取所有patch和token中的最大值
                    elif reduction == 'mean':
                        score = sim_matrix.mean()  # 取平均值
                    else:
                        raise ValueError("Unknown reduction method: choose 'max' or 'mean'")

                    return score.item()

                def compute_all_frame_scores(image_features, query_embedding, reduction='max'):
                    """
                    对每一帧进行相似度计算，返回 (frame_index, score) 的列表。

                    参数:
                        image_features: tensor, 形状为 (frame_num, 196, 3584)
                        query_embedding: tensor, 形状为 (token_num, 3584)
                        reduction: str, 归约方式 ('max' 或 'mean')

                    返回:
                        frame_scores: list of tuple, 每个元素为 (frame_index, score)
                    """
                    frame_scores = []
                    for idx, frame_feature in enumerate(image_features):
                        score = compute_frame_score(frame_feature, query_embedding, reduction)
                        frame_scores.append((idx, score))
                    return frame_scores

                scores = compute_all_frame_scores(image_feature, query_feature.squeeze(0), reduction='mean')
                # print("Frame Scores:")
                #
                # def meanstd(len_scores, dic_scores, n, fns, t1, t2, all_depth):
                #     split_scores = []
                #     split_fn = []
                #     no_split_scores = []
                #     no_split_fn = []
                #     i = 0
                #     for dic_score, fn in zip(dic_scores, fns):
                #         # normalized_data = (score - np.min(score)) / (np.max(score) - np.min(score))
                #         score = dic_score['score']
                #         depth = dic_score['depth']
                #         mean = np.mean(score)
                #         std = np.std(score)
                #
                #         top_n = heapq.nlargest(n, range(len(score)), score.__getitem__)
                #         top_score = [score[t] for t in top_n]
                #         # print(f"split {i}: ",len(score))
                #         i += 1
                #         mean_diff = np.mean(top_score) - mean
                #         if mean_diff > t1 and std > t2:
                #             no_split_scores.append(dic_score)
                #             no_split_fn.append(fn)
                #         elif depth < all_depth:
                #             # elif len(score)>(len_scores/n)*2 and len(score) >= 8:
                #             score1 = score[:len(score) // 2]
                #             score2 = score[len(score) // 2:]
                #             fn1 = fn[:len(score) // 2]
                #             fn2 = fn[len(score) // 2:]
                #             split_scores.append(dict(score=score1, depth=depth + 1))
                #             split_scores.append(dict(score=score2, depth=depth + 1))
                #             split_fn.append(fn1)
                #             split_fn.append(fn2)
                #         else:
                #             no_split_scores.append(dic_score)
                #             no_split_fn.append(fn)
                #     if len(split_scores) > 0:
                #         all_split_score, all_split_fn = meanstd(len_scores, split_scores, n, split_fn, t1, t2,
                #                                                 all_depth)
                #     else:
                #         all_split_score = []
                #         all_split_fn = []
                #     all_split_score = no_split_scores + all_split_score
                #     all_split_fn = no_split_fn + all_split_fn
                #
                #     return all_split_score, all_split_fn
                # max_num_frames = 16
                # t1 = 0.8
                # t2 = -100
                # all_depth = 3
                boundaries = boundary_list[index]
                print(boundaries)
                for idx, score in scores:
                    print(f"Frame {idx}: score = {score}")
                # -------------------- 关键帧挑选部分 --------------------
                # 将 (frame_index, score) 分离为两个列表
                frame_score_values = [score for frame_idx, score in scores]
                segment_scores = []
                for i in range(len(boundaries) - 1):
                    start = boundaries[i]
                    end = boundaries[i + 1]
                    # 计算该 segment 内所有帧的平均相似度作为该 segment 的分数
                    segment_score = np.mean(frame_score_values[start:end])
                    segment_scores.append(segment_score)
                    print(f"Segment {i} (frames {start}-{end - 1}) score: {segment_score}")

                # 选择最相关的 segment（分数最高）
                selected_segment_index = np.argmax(segment_scores)
                selected_segment_start = boundaries[selected_segment_index]
                selected_segment_end = boundaries[selected_segment_index + 1]
                print(
                    f"Selected Segment: Index {selected_segment_index} with frame range {selected_segment_start} to {selected_segment_end - 1}")

                # 从 image_feature 中提取出该 segment 的所有帧
                selected_segment_feature = image_feature[selected_segment_start:selected_segment_end]

                # 直接更新 image_features 列表中的元素
                image_features[index] = selected_segment_feature

                # # 如果帧数超过阈值，则进行关键帧挑选，否则全部保留
                # if len(frame_score_values) >= max_num_frames:
                #     # 归一化分数到 [0,1] 区间
                #     score_arr = np.array(frame_score_values)
                #     normalized_scores = (score_arr - np.min(score_arr)) / (np.max(score_arr) - np.min(score_arr))
                #
                #     # 构造初始的分数字典，深度为0
                #     initial_score_dict = dict(score=normalized_scores.tolist(), depth=0)
                #     # 同时传入所有帧对应的索引列表
                #     selected_score_dicts, selected_frame_indices = meanstd(len(normalized_scores),
                #                                                            [initial_score_dict],
                #                                                            max_num_frames,
                #                                                            [frame_indices],
                #                                                            t1, t2, all_depth)
                #
                #     # 根据每个分割段的深度决定挑选的帧数
                #     selected_frames = []
                #     for seg, seg_indices in zip(selected_score_dicts, selected_frame_indices):
                #         f_num = int(max_num_frames / (2 ** seg['depth']))
                #         # 从该段中挑选得分最高的 f_num 个帧
                #         topk_indices = heapq.nlargest(f_num, range(len(seg['score'])), seg['score'].__getitem__)
                #         selected_frames.extend([seg_indices[i] for i in topk_indices])
                #     selected_frames.sort()
                # else:
                #     selected_frames = frame_indices
                #
                # print("Selected Key Frames:", selected_frames)
                # selected_image_features = image_feature[selected_frames]
                # print(index)
                # image_features[index] = selected_image_features

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
                #rank_print(f"Image feature shape flat : {image_features[0].shape}")

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    # import pdb; pdb.set_trace()
                    if image_idx in video_idx_in_batch:  # video operations
                        rank0_print("Video in batch")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            # 模型将视频帧划分为多个网格（grid），并在每个网格位置添加一个视觉 token
                            image_feature = self.add_token_per_grid(image_feature)
                            #rank_print(f"Image feature shape grid : {image_feature.shape}")
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")

                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            # 直接对视频帧进行处理，每个帧被表示为一个视觉 token
                            image_feature = self.add_token_per_frame(image_feature)  # [frame_num, 197, 3584]

                            new_image_features.append(image_feature.flatten(0, 1))
                            #rank_print(f"Image feature shape frame : {new_image_features[0].shape}")  # [n, 3584]
                        elif mm_newline_position == "one_token":
                            # one-token
                            # 模型将整个视频序列展平成一个单一的视觉 token

                            # Add hierarchical memory module
                            # frame_memory = self.compress_temporal_features(image_feature)
                            rank_print(f"Image feature shape one_token before flatten: {image_feature.shape}")  # [frame_num*196, 3584]
                            image_feature = image_feature.flatten(0, 1)
                            # image_feature = torch.cat((image_feature, frame_memory[0]), dim=0)
                            #rank_print(f"Image feature shape one_token : {image_feature.shape}")  # [frame_num*196, 3584]
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device) # Adds a new dimension at the beginning of the tensor
                                ), dim=0)
                            #rank_print(f"Image feature shape one_token after unpad: {image_feature.shape}")  # [frame_num*196+1, 3584]
                            new_image_features.append(image_feature)
                            #rank_print(f"new_image_features length: {len(new_image_features)}")
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        rank0_print("Images in batch")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        rank0_print("Single-image in batch")
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        #rank_print(f"Total images : {len(image_features)} with shape {[x.shape for x in image_features]}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        #rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # Concatenate text embeddings (cur_input_embeds_1) with an empty image feature placeholder
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            # Adds -1 at the beginning and cur_input_ids.shape[0] at the end to mark start and end positions
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

            # Splits text and label sequences at positions where image tokens exist
            cur_input_ids_noim = []  # Stores text chunks between image tokens
            cur_labels = labels[batch_idx]
            cur_labels_noim = []  # Stores labels for those text chunks
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # Embed the Text Tokens
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            # Insert Image Features into the Text Embeddings
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            # Move to GPU and Concatenate
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        #rank_print("Finishing Inserting")
        for x, modality in zip(new_input_embeds, modalities):
             rank_print(f"New input embeds shape with {modality}: {x.shape}") # [squence_length, 3584]
        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        # Pad Sequences to Max Length
        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        #rank_print(f"New input embeds shape: {new_input_embeds.shape}")  # [batch_size, sequence_length, 3584]

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        rank_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
