import math
from typing import Optional, List, Tuple
from einops import rearrange

import torch
from torch import nn
from transformers.activations import ACT2FN


class Config:
    mm_hidden_size = 896
    mm_hidden_act = "relu"
    mm_num_attention_heads = 8
    patch_size = 196
    mm_attention_probs_dropout_prob = 0.1
    mm_layer_norm_eps = 1e-12
    mm_hidden_dropout_prob = 0.1
    mm_intermediate_size = 4 * mm_hidden_size
    num_memory_tokens = 8
    depth = 1
    mm_dtype = torch.float16


class Residual(nn.Module):
    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size, dtype=config.mm_dtype)
        self.layernorm = nn.LayerNorm(output_size, eps=config.mm_layer_norm_eps, dtype=config.mm_dtype)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        return self.layernorm(hidden_states + input_tensor)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.mm_hidden_size
        self.num_attention_heads = config.mm_num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, dtype=config.mm_dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, dtype=config.mm_dtype)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, dtype=config.mm_dtype)

        self.residual = Residual(self.hidden_size, self.hidden_size, config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, kv_hidden_states=None):
        query = self.transpose_for_scores(self.q_proj(hidden_states))

        if kv_hidden_states is None:
            key = self.transpose_for_scores(self.k_proj(hidden_states))
            value = self.transpose_for_scores(self.v_proj(hidden_states))
        else:
            key = self.transpose_for_scores(self.k_proj(kv_hidden_states))
            value = self.transpose_for_scores(self.v_proj(kv_hidden_states))

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            scores += attention_mask

        probs = nn.functional.softmax(scores, dim=-1)
        if head_mask is not None:
            probs = probs * head_mask

        context = torch.matmul(probs, value).permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size(0), -1, self.hidden_size)

        return self.residual(context, hidden_states)


class CrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attention = Attention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.mm_intermediate_size, dtype=config.mm_dtype),
            ACT2FN[config.mm_hidden_act],
        )
        self.residual = Residual(config.mm_intermediate_size, config.mm_hidden_size, config)

    def forward(self, memory_states, image_states):
        attention_output = self.cross_attention(memory_states, kv_hidden_states=image_states)
        ffn_output = self.mlp(attention_output)
        return self.residual(ffn_output, attention_output)


class MemoryModule(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        self.memory_fusion_layers = nn.ModuleList([CrossAttentionBlock(self.config) for _ in range(self.config.depth)])
        self.memory_update_attention = Attention(self.config)

        self.num_memory_tokens = self.config.num_memory_tokens
        self.hidden_size = self.config.mm_hidden_size
        self.patch_size = self.config.patch_size

        self.initial_memory = nn.Parameter(torch.empty(self.num_memory_tokens, self.patch_size, self.hidden_size))
        nn.init.xavier_uniform_(self.initial_memory)

        self.memory_cache: List[torch.Tensor] = []

    def _update_memory_with_cache(self, current_memory):
        if not self.memory_cache:
            return current_memory

        past_memory = torch.cat(self.memory_cache, dim=0).unsqueeze(0)
        query = current_memory.unsqueeze(0)
        B, Lq, P, D = query.shape
        query_2d = query.view(B, Lq * P, D)
        keyval_2d = past_memory.view(1, -1, D)

        updated = self.memory_update_attention(query_2d, kv_hidden_states=keyval_2d)
        return updated.view(B, Lq, P, D).squeeze(0)

    def forward(self, image_features: torch.Tensor):
        device = image_features.device
        dtype = image_features.dtype

        if not self.memory_cache:
            memory = self.initial_memory.to(device=device, dtype=dtype)
        else:
            memory = self.memory_cache[-1].to(device=device, dtype=dtype)

        if len(self.memory_cache) > 1:
            memory = self._update_memory_with_cache(memory)

        for layer in self.memory_fusion_layers:
            N, P, D = memory.shape
            M, Q, D_ = image_features.shape
            memory_2d = memory.reshape(1, N * P, D)
            image_2d = image_features.reshape(1, M * Q, D_)
            memory_2d = layer(memory_2d, image_2d)
            memory = memory_2d.view(1, N, P, D).squeeze(0)

        self.memory_cache.append(memory)
        MAX_BACKPROP_STEPS = 10
        if len(self.memory_cache) > MAX_BACKPROP_STEPS:
            self.memory_cache = self.memory_cache[-MAX_BACKPROP_STEPS:]

        return self.memory_cache
