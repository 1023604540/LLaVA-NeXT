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
from llava.model.memory_module.segment import segment, adjusted_segment, uniform_segment
import heapq
import numpy as np
from llava.model.memory_module.MemoryController import TransformerProjector
from llava.model.memory_module.bigru import TemporalGRUEncoder
from llava.model.memory_module.position_encoding import TemporalPositionalEncoding
import time


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
# Initialization function
def kaiming_init_linear(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        if layer.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(layer.bias, -bound, bound)
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

        LLM_hidden_dim = getattr(config, "llm_hidden_dim", 896)
        memory_prompt_hidden_dim = getattr(config, "memory_prompt_hidden_dim", 896)
        self.memory_proj_layers = getattr(config, "injected_layers", 10)

        # Define memory projections
        self.memory_projections = nn.ModuleList([
            nn.Linear(LLM_hidden_dim, memory_prompt_hidden_dim).to(dtype=self.dtype,
                                                        device=self.device) for _ in range(self.memory_proj_layers)
        ])
        #self.memory_projections.apply(kaiming_init_linear)

        # Define recurrent memory transformer
        self.recurrent_memory_transformer = TransformerProjector().to(self.device)

        # self.recurrent_memory_transformer.apply(kaiming_init_linear)
        self.memory_readout_cache = None
        # Initialize positional encoding
        self.positional_encoding = TemporalPositionalEncoding(
            max_frames=250,
            embed_dim=LLM_hidden_dim,
            learnable=False
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
        image_features = self.get_model().mm_projector(image_features)
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
        # start = time.time()
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # print("vision_tower is None or images is None or input_ids.shape[1] == 1")
            return None, input_ids, position_ids, attention_mask, past_key_values, None, labels

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


            # # Now support only batch size of 1
            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]

            chunk_wise_encode = False
            if chunk_wise_encode:
                # Encode the images in chunks to save memory
                # Set the chunk size
                chunk_size = 100

                # Store the encoded features
                encoded_chunks = []

                # Loop over the image frames in chunks
                for i in range(0, concat_images.shape[0], chunk_size):
                    chunk = concat_images[i:i + chunk_size]
                    print(f"chunk shape : {chunk.shape}")
                    encoded_chunk = self.encode_images(chunk)
                    encoded_chunks.append(encoded_chunk)

                # Concatenate all the encoded chunks
                encoded_image_features = torch.cat(encoded_chunks, dim=0)
            else:
                encoded_image_features = self.encode_images(concat_images)

            encoded_image_features = torch.split(encoded_image_features, split_sizes)


            # print(f"image_features shape : {[x.shape for x in image_features]}, self.get_model().memory_readout_cache shape : {self.get_model().memory_readout_cache.shape}")
            # rank0_print(f"Encoded image feats : {[x.shape for x in image_features]}, after proj time {time.time() - start}")  # [frame_num, 729, 3584]


            new_image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    #print(f"before_2dpool = {time.time() - start}")
                    new_image_features.append(self.get_2dPool(image_feat))
                    #print(f"after_2dpool = {time.time() - start}")
                else:
                    new_image_features.append(image_feat)
            image_features = new_image_features  # [frame_num, 196, 3584]

            recurrent_memory = None
            memory_augmented_features = []
            for idx, image in enumerate(image_features):
                # If it is not a video feature, we don't need to process it
                if idx not in video_idx_in_batch:
                    non_video_images.append(image)
                    non_video_positions.append(idx)
                    continue
                # Add positional encoding
                image = self.get_model().positional_encoding(image)
                # rank_print(f"image shape after positional encoding: {image.shape}")
                # Init recurrent memory module
                # rank_print(f"image shape : {image.shape}")
                boundaries = uniform_segment(image.mean(dim=1), d=32)
                rank_print(f"boundaries : {boundaries}")
                recurrent_model = self.get_model().recurrent_memory_transformer.to(self.device)
                # Clear the memory cache to avoid memory leak across videos
                updated_image_segment = None
                recurrent_memory = None
                recurrent_model.memory_cache = []

                # print(f"Encoded features shape : {encoded_features.shape}")
                # encoded_features = encoded_features.requires_grad_()
                # rank_print(f"boundaries : {boundaries}")
                image_segments = [image[boundaries[i]:boundaries[i + 1]] for i in range(len(boundaries) - 1)]
                for image_segment in image_segments:
                    # rank_print(f"Image segment shape : {image_segment.shape}")
                    # rank0_print(torch.cuda.memory_allocated() / 1024 ** 3, "GB allocated")
                    # rank0_print(torch.cuda.memory_reserved() / 1024 ** 3, "GB reserved")
                    recurrent_memory, updated_image_segment = recurrent_model(image_segment)
                    # rank_print(f"updated_image_segment shape : {updated_image_segment.shape}")
                    # rank_print(f"recurrent_memory shape : {recurrent_memory.shape}")
                # Branch dropout the updated image segment
                dropout_rate = getattr(self.config, "recurrent_dropout_rate", 0.2)
                if torch.rand(1, device=updated_image_segment.device).item() < dropout_rate:
                    updated_image_segment = torch.zeros(updated_image_segment.shape).to(device=self.device,dtype=self.dtype)
                    rank_print(f"updated_image_segment dropout")
                memory_augmented_features.append(updated_image_segment)
            if recurrent_memory is not None:
                self.get_model().memory_readout_cache = recurrent_memory
            projected_prompts = []
            # Project through each layer's linear projection
            for i in range(self.get_model().memory_proj_layers):
                # (4, 196, 896) → (1, 784, 896)
                projected = self.get_model().memory_projections[i](self.get_model().memory_readout_cache).view(1, -1,self.config.hidden_size)
                projected_prompts.append(projected)

            # Stack into shape: (10, 784, 896)
            memory_prompt_stack = torch.cat(projected_prompts, dim=0)  # shape: (10, 784, 896)




            image_features = memory_augmented_features

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")
            # print(mm_newline_position)
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
                        # rank0_print("Video in batch")
                        if mm_newline_position == "grid":
                            print("Grid-wise")
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
                            # rank_print(f"Image feature shape one_token before flatten: {image_feature.shape}")  # [frame_num*196, 3584]
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
             rank0_print(f"New input embeds shape with {modality}: {x.shape}") # [squence_length, 3584]
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
        # rank_print(f"Finish preparing")
        # print(f"new_input_embeds shape: {new_input_embeds.shape}, new_labels shape: {new_labels.shape if new_labels is not None else None}, position_ids shape: {position_ids.shape if position_ids is not None else None}, attention_mask shape: {attention_mask.shape if attention_mask is not None else None}, past_key_values shape: {past_key_values[0].shape if past_key_values is not None else None}")
        # if past_key_values is None:
        #     if self.get_model().memory_readout_cache is not None:
        #         print("Memory readout injecting")
        #         memory_readout = self.get_model().memory_readout_cache.to(dtype=self.dtype, device=self.device).flatten(0, 1)
        #         print(f"memory_readout shape, {memory_readout.shape}")
        #         T_mem = memory_readout.shape[0]  # memory tokens
        #
        #         # === 1. Inject past_key_values ===
        #         past_key_values = self.inject_memory_as_kv(memory_readout, past_key_values)
        #         self.get_model().memory_readout_cache = None
        # print(f"past_key_values shape: {past_key_values[0][0].shape if past_key_values is not None else None}")

        # num_memory_layers = 4
        # memory_length = 5  # number of memory tokens
        # hidden_size = 896
        # memory_prompt = torch.randn(num_memory_layers, memory_length, hidden_size).to(dtype=self.dtype, device=self.device)
        memory_prompt_stack = None
        # memory_prompt_stack = torch.rand([10, 27840, 896]).to(dtype=self.dtype, device=self.device)
        return memory_prompt_stack, None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def inject_memory_as_kv(self, memory_readout, old_cache=None):
        """
        Inserts memory tensors into the past_key_values (key-value cache) for all layers.

        Args:
            memory_readout (torch.Tensor): Memory tensor of shape [T, hidden_dim].
            old_cache (list or None): Existing past_key_values, a list of tuples [(key, value), ...].
                                      If None, it is treated as empty.

        Returns:
            list: Updated past_key_values with memory tensors concatenated.
        """
        B = 1  # Batch size, adjust if needed
        T = memory_readout.shape[0]  # Number of memory tokens
        H = self.config.num_key_value_heads  # Number of attention heads
        L = self.config.num_hidden_layers  # Number of layers
        Dh = 64  # Dimension per head, confirm from your model config

        new_cache = []

        # Iterate over all layers
        for i in range(L):
            # Project memory_readout to key and value for the current layer
            mem_key = self.model.memory_key_projs[i](memory_readout).view(B, T, H, Dh)
            mem_key = mem_key.permute(0, 2, 1, 3).contiguous()  # [B, H, T, Dh]
            mem_value = self.model.memory_value_projs[i](memory_readout).view(B, T, H, Dh)
            mem_value = mem_value.permute(0, 2, 1, 3).contiguous()  # [B, H, T, Dh]

            # Handle the case where old_cache is None or empty
            if old_cache is None or len(old_cache) == 0:
                old_key = torch.empty(B, H, 0, Dh, dtype=memory_readout.dtype, device=memory_readout.device)
                old_value = torch.empty_like(old_key)
            else:
                old_key, old_value = old_cache[i]

            # Concatenate memory tensors with the old cache
            new_key = torch.cat([mem_key, old_key], dim=2)  # [B, H, T + old_len, Dh]
            new_value = torch.cat([mem_value, old_value], dim=2)  # [B, H, T + old_len, Dh]

            # Append the updated key-value pair to the new cache
            new_cache.append((new_key, new_value))

        return new_cache

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
