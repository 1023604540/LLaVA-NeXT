import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from llava.model.memory_module.compress_functions import drop_feature, merge_feature, kmeans_feature, weighted_kmeans_feature, k_drop_feature, k_merge_feature, attention_feature
from llava.model.memory_module.segment import segment

class NeuralTuringMachine(nn.Module):
    def __init__(self, input_dim=1152, output_dim=1152, attention_dropout=0.1):
        super(NeuralTuringMachine, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.q_proj = nn.Linear(input_dim, output_dim)
        self.k_proj = nn.Linear(input_dim, output_dim)
        self.v_proj = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(attention_dropout)
        self.out_proj = nn.Linear(output_dim, input_dim)
        self.out_dropout = nn.Dropout(attention_dropout)
        self.out_ln = nn.LayerNorm(input_dim, eps=1e-12)

    def get_weight(self, x, y):
        query = self.q_proj(x)
        key = self.k_proj(y)
        scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(self.output_dim)
        weight = torch.softmax(scores, dim=-1)
        return weight

    def forward(self, x, y):
        query = self.q_proj(x)
        key = self.k_proj(y)
        scores = torch.matmul(query, key.transpose(0, 1)) / math.sqrt(self.output_dim)
        weight = torch.softmax(scores, dim=-1)
        attn = self.dropout(weight)
        value = self.v_proj(y)
        output = torch.matmul(attn, value)
        output = self.out_proj(output)
        output = self.out_dropout(output)
        output = self.out_ln(output.unsqueeze(0)).squeeze(0)
        return output

class MultimodalOpsMixin:
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Replace with GELU activation:
        self.memory_mlp = nn.Sequential(
            nn.Linear(1152, 1152),
            nn.GELU(),          # <-- Use GELU here
            nn.Linear(1152, 1152),
        )
        #self.attention_model = NeuralTuringMachine(input_dim=1152, output_dim=1152, attention_dropout=0.1)
    def attention(self, turing_memory, new_feature, update_ratio=0.2):
        """
        Update the turing_memory using attention between turing_memory and new_feature.
        """
        T1, D1 = turing_memory.shape
        T2, D2 = new_feature.shape
        assert D1 == D2, f"Dimension mismatch: {D1} != {D2}"
        model = self.get_model().attention_model
        weight = model.get_weight(turing_memory, new_feature)
        weight = weight * update_ratio  # Scale weights
        decay = weight.sum(dim=1, keepdim=True)
        turing_memory = turing_memory * (1 - decay) + torch.mm(weight, new_feature)
        return turing_memory
    def attention2(self, turing_memory, new_feature, update_ratio=0.2):  # deprecated
        T1, D1 = turing_memory.shape
        T2, D2 = new_feature.shape
        assert D1 == D2, f"dimmension not match, {D1} != {D2}"
        model = self.get_model().attention_model
        turing_memory = model.forward(turing_memory, new_feature)
        return turing_memory
    def compress_spatial_features(self, image_features, compress_size=1):
        """
        Compresses spatial features using a 2D pooling operation.
        Assumes image_features is of shape (num_frames, num_patches, hidden_dim)
        with num_patches being a square number.
        """
        compress_type = getattr(self.config, "compress_type", 'mean')
        patch_size = round(math.sqrt(image_features.shape[1]))
        assert patch_size * patch_size == image_features.shape[1], (
            f"For ViT feature map, {patch_size}*{patch_size} != {image_features.shape[1]}"
        )
        if patch_size == compress_size:
            return image_features
        elif compress_type is not None:
            if 'mean' in compress_type:
                if compress_size == 1:
                    image_features = image_features.mean(dim=1, keepdim=True)
                else:
                    image_features = image_features.view(-1, patch_size, patch_size, image_features.shape[-1])
                    image_features = image_features.permute(0, 3, 1, 2).contiguous()
                    pooled_features = torch.nn.functional.avg_pool2d(
                        image_features, (patch_size // compress_size, patch_size // compress_size)
                    )
                    pooled_features = pooled_features.permute(0, 2, 3, 1).contiguous()
                    image_features = pooled_features.view(-1, compress_size * compress_size, pooled_features.shape[-1])
            else:
                raise NotImplementedError(f"`compress_type` {self.config.compress_type} is not supported yet.")
        return image_features

    def compress_temporal_features(self, image_features, video_idx_in_batch, all_video = False):
        """
        Compress temporal features from image_features.
        This method depends on configuration parameters such as:
            video_long_memory_length, video_Turing_memory_length, etc.
        """
        video_long_memory_length = getattr(self.config, "video_long_memory_length", 3)
        video_Turing_memory_length = getattr(self.config, "video_Turing_memory_length", 3)
        video_current_memory_length = getattr(self.config, "video_current_memory_length", 1)
        compress_long_memory_size = getattr(self.config, "compress_long_memory_size", 27)
        compress_Turing_memory_size = getattr(self.config, "compress_Turing_memory_size", 27)
        compress_Turing_update_ratio = getattr(self.config, "compress_Turing_update_ratio", 0.2)
        video_sample_type = getattr(self.config, "video_sample_type", "weighted_kmeans")

        # A dictionary mapping sample types to functions (assumed to be imported or defined elsewhere)
        compress_fn_dic = {
            'drop': drop_feature,
            'merge': merge_feature,
            'kmeans': kmeans_feature,
            'weighted_kmeans': weighted_kmeans_feature,
            'kdrop': k_drop_feature,
            'kmerge': k_merge_feature,
            'attention': attention_feature,
        }
        if video_sample_type in compress_fn_dic:
            compress_fn = compress_fn_dic[video_sample_type]
        else:
            raise NotImplementedError(f'video_sample_type {video_sample_type} is not supported.')
        if all_video:
            video_idx_in_batch = list(range(len(image_features)))
        new_image_features = []
        for idx, img_feature in enumerate(image_features):
            # If it is not a video feature, we don't need to process it
            if idx not in video_idx_in_batch:
                new_image_features.append(None)
                continue

            cur_start = min(video_current_memory_length, img_feature.shape[0])
            if cur_start == 0:
                cur_memory = img_feature[:0]
                long_memory = img_feature
                Turing_memory = img_feature
            else:
                cur_memory = img_feature[-cur_start:]
                long_memory = img_feature[:-cur_start]
                Turing_memory = img_feature[:-cur_start]

            if compress_long_memory_size * compress_long_memory_size != long_memory.shape[1]:
                long_memory = self.compress_spatial_features(long_memory, compress_long_memory_size)
            if compress_Turing_memory_size * compress_Turing_memory_size != Turing_memory.shape[1]:
                Turing_memory = self.compress_spatial_features(Turing_memory, compress_Turing_memory_size)

            if video_long_memory_length == 0 or long_memory.shape[0] == 0:
                long_memory_compressed = long_memory[:0]
            else:
                long_memory_compressed, weight, step_long_indices = compress_fn(long_memory, video_long_memory_length)

                sorted_indices = torch.argsort(weight, descending=True)
                key_centroids = long_memory_compressed[sorted_indices]
                key_length = 3
                if key_centroids.shape[0] > key_length:
                    key_centroids = key_centroids[:key_length]
                dists = ((long_memory.unsqueeze(1) - key_centroids.unsqueeze(0)) ** 2).sum(dim=3).sum(dim=2).sqrt()
                min_indices = torch.argmin(dists, dim=0)
                min_indices_sorted, _ = torch.sort(min_indices)
                print(f"Min indices sorted: {min_indices_sorted}")
                key_memory = img_feature[min_indices_sorted]
                cur_memory = torch.cat([key_memory, cur_memory], dim=0)

            if video_Turing_memory_length == 0 or Turing_memory.shape[0] == 0:
                Turing_memory_compressed = Turing_memory[:0]
            else:
                Turing_memory_compressed, _ = attention_feature(
                    Turing_memory, video_Turing_memory_length, self.attention, update_ratio=compress_Turing_update_ratio
                )
            print(f"Memory: Long {long_memory_compressed.shape}, Turing {Turing_memory_compressed.shape}, Cur {cur_memory.shape}")
            if long_memory_compressed.shape[0] < video_long_memory_length:
                long_memory_compressed = long_memory[:0]
            if Turing_memory_compressed.shape[0] < video_Turing_memory_length:
                Turing_memory_compressed = Turing_memory[:0]
            memory_feature = torch.cat([
                Turing_memory_compressed.view(-1, 729, 1152),   #（9，9*9，1152）（1，729，1152）
                long_memory_compressed.view(-1, 729, 1152),
                cur_memory.view(-1, 729, 1152),
            ], dim=0).to(dtype=img_feature.dtype)

            mem_shape = memory_feature.shape  # e.g. (frames, 729, 1152)
            memory_feature = memory_feature.view(-1, mem_shape[-1])  # Flatten (frames*729, 1152)
            memory_feature = self.get_model().memory_mlp(memory_feature)  # MLP with GELU activation
            memory_feature = memory_feature.view(*mem_shape)
            new_image_features.append(memory_feature)
        return new_image_features
