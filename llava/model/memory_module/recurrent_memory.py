import math
from typing import Optional, List, Tuple, Union
from einops import rearrange, repeat, pack, unpack

import torch
from torch import nn
from transformers.activations import ACT2FN


class Config:
    mm_hidden_size = 1152
    mm_hidden_act = "relu"
    mm_num_attention_heads = 8
    patch_size = 729  # Patch size
    mm_attention_probs_dropout_prob = 0.1  # Attention dropout
    mm_layer_norm_eps = 1e-12  # LayerNorm epsilon
    mm_hidden_dropout_prob = 0.1  # Residual dropout
    mm_intermediate_size = 4096  # Feedforward hidden layer size
    num_memory_tokens = 8  # Number of memory tokens
    depth = 1  # Number of Transformer layers
    mm_dtype = torch.float16


class Residual(nn.Module):
    def __init__(self, input_size, output_size, config):
        super().__init__()
        # Define layers with a specified dtype, but do NOT force a device here
        self.dense = nn.Linear(
            input_size, output_size, dtype=config.mm_dtype
        )
        self.layernorm = nn.LayerNorm(
            output_size, eps=config.mm_layer_norm_eps, dtype=config.mm_dtype
        )
        self.dropout = nn.Dropout(config.mm_hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.mm_hidden_size
        self.num_attention_heads = config.mm_num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads

        assert self.hidden_size % self.num_attention_heads == 0

        # Again, no explicit device; just dtype
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, dtype=config.mm_dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, dtype=config.mm_dtype)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, dtype=config.mm_dtype)

        self.dropout = nn.Dropout(config.mm_attention_probs_dropout_prob)
        self.residual = Residual(self.hidden_size, self.hidden_size, config)

    def transpose_for_scores(self, x):
        """
        (B, Lq*P, D) -> (B, H, Lq*P, DH)
        where H = num_attention_heads, DH = attention_head_size
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        kv_hidden_states: Optional[torch.FloatTensor] = None,
        kv_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        If `kv_hidden_states` is None, we do self-attention.
        Otherwise, cross-attention with `kv_hidden_states` as K/V.
        """
        query = self.transpose_for_scores(self.q_proj(hidden_states))  # (B, H, Lq*P, DH)

        if kv_hidden_states is not None:
            # Cross-attention
            if past_key_value is not None:
                key = past_key_value[0]
                value = past_key_value[1]
                attention_mask = kv_attention_mask
            else:
                key = self.transpose_for_scores(self.k_proj(kv_hidden_states))
                value = self.transpose_for_scores(self.v_proj(kv_hidden_states))
                attention_mask = kv_attention_mask

            past_key_value = (key, value)
        else:
            # Self-attention
            if past_key_value is not None:
                key = self.transpose_for_scores(self.k_proj(hidden_states))
                value = self.transpose_for_scores(self.v_proj(hidden_states))
                key = torch.cat([past_key_value[0], key], dim=2)
                value = torch.cat([past_key_value[1], value], dim=2)
            else:
                key = self.transpose_for_scores(self.k_proj(hidden_states))
                value = self.transpose_for_scores(self.v_proj(hidden_states))

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores += attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(new_context_shape)  # (B, Lq*P, D)

        # Residual
        output = self.residual(context, hidden_states)

        outputs = (output, attention_probs) if output_attentions else (output,)
        if past_key_value is not None:
            outputs = outputs + (past_key_value,)

        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attention = Attention(config)

        self.mlp = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.mm_intermediate_size, dtype=config.mm_dtype),
            ACT2FN[config.mm_hidden_act],
        )
        self.residual = Residual(config.mm_intermediate_size, config.mm_hidden_size, config)

    def ffn(self, attention_output):
        intermediate_output = self.mlp(attention_output)
        layer_output = self.residual(intermediate_output, attention_output)
        return layer_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        kv_hidden_states: Optional[torch.FloatTensor] = None,
        kv_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        """
        Standard Transformer block:
        1) Self-Attention
        2) Feed-forward
        """
        if past_key_value is not None:
            self_past_key_value = past_key_value[:2]
        else:
            self_past_key_value = None

        # Self-Attention
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # e.g. attention_probs, (past_key_value)...

        # Feed-forward
        layer_output = self.ffn(attention_output)
        return (layer_output,) + outputs


class TransformerProjector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else Config()

        # Main Transformer layers
        self.layers = nn.ModuleList(
            [TransformerLayer(self.config) for _ in range(self.config.depth)]
        )

        # Cross-attention layer for memory retrieval
        self.memory_retrieval_attention = Attention(self.config)

        self.num_memory_tokens = self.config.num_memory_tokens
        self.hidden_size = self.config.mm_hidden_size
        self.patch_size = self.config.patch_size

        # Define initial memory (uninitialized)
        self.initial_memory = nn.Parameter(
            torch.randn(
                self.num_memory_tokens, self.patch_size, self.hidden_size
            )
        )

        # Will store previous memory tokens across calls
        self.memory_cache: List[torch.Tensor] = []

    def _update_memory_tokens_with_cache(self, current_memory: torch.Tensor) -> torch.Tensor:
        """
        Cross-attend current_memory to entire memory_cache to produce updated memory.
        """
        if len(self.memory_cache) == 0:
            # No previous memory
            return current_memory

        past_memory = torch.cat(self.memory_cache, dim=0).unsqueeze(0)  # (1, N*n, patch, d)
        query_mem = current_memory.unsqueeze(0)                         # (1, n, patch, d)

        B, Lq, P, D = query_mem.shape
        query_2d = query_mem.view(B, Lq * P, D)

        B2, Lk, Pk, D2 = past_memory.shape
        keyval_2d = past_memory.view(B2, Lk * Pk, D2)

        cross_attn_out = self.memory_retrieval_attention(
            hidden_states=query_2d,
            kv_hidden_states=keyval_2d,
            attention_mask=None,
            head_mask=None,
            output_attentions=False
        )
        updated_2d = cross_attn_out[0]  # (1, Lq*P, D)

        updated_4d = updated_2d.view(B, Lq, P, D)
        return updated_4d.squeeze(0)

    def forward(
        self,
        image_features: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
    ):
        """
        image_features: (frames=F, patch=P, dimension=D)
        1) If memory_cache empty, use self.initial_memory. Otherwise, memory_cache[-1].
        2) Possibly cross-attend memory tokens with the entire memory_cache.
        3) Prepend memory to image => shape (F+n, P, D).
        4) Self-attend with `depth` layers.
        5) Split memory vs image outputs; store memory in cache.
        6) Return new memory tokens.
        """
        device = image_features.device
        dtype = image_features.dtype

        # (1) Decide memory
        if len(self.memory_cache) == 0:
            current_memory = self.initial_memory.to(device=device, dtype=dtype)
        else:
            current_memory = self.memory_cache[-1].to(device=device, dtype=dtype)

        # (2) Cross-attention if we have >1 memory blocks
        if len(self.memory_cache) > 1:
            current_memory = self._update_memory_tokens_with_cache(current_memory)

        # (3) Prepend memory
        combined = torch.cat([current_memory, image_features], dim=0)
        combined = combined.unsqueeze(0)  # => (1, F+n, P, D)
        B, L, P_, D_ = combined.shape
        combined_2d = combined.view(B, L * P_, D_)

        # (4) Self-attention through each layer
        hidden_states = combined_2d
        for i, layer in enumerate(self.layers):
            layer_head_mask = head_mask[i] if (head_mask is not None) else None
            layer_outputs = layer(
                hidden_states,
                attention_mask=None,
                head_mask=layer_head_mask,
                kv_hidden_states=None,
                kv_attention_mask=None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

        # (5) Reshape, split memory vs. image
        hidden_4d = hidden_states.view(B, L, P_, D_)
        new_memory_tokens = hidden_4d[:, : self.num_memory_tokens, :, :]  # => (1, n, P, D)
        updated_image_features = hidden_4d[:, self.num_memory_tokens :, :, :]  # => (1, F, P, D)

        # (6) Cache new memory tokens
        self.memory_cache.append(new_memory_tokens.squeeze(0).detach())

        # Return new memory tokens (or updated image if you prefer)
        return new_memory_tokens.squeeze(0)

    #
    # Optional initialization function:
    #
    # def init_weights_(self):
    #     """
    #     Optional: Call this after you've placed the model on a real device
    #     (CPU/GPU) if you want to random-initialize everything.
    #     """
    #     for name, param in self.named_parameters():
    #         # Check if param is still on meta
    #         if param.device.type == "meta":
    #             raise RuntimeError(
    #                 f"Parameter {name} is on meta. Move model to a real device before init."
    #             )
    #
    #         # Example default init
    #         if param.dim() > 1:
    #             nn.init.xavier_normal_(param)
    #         else:
    #             nn.init.zeros_(param)


#
# EXAMPLE USAGE (no explicit "meta" mention here):
#
if __name__ == "__main__":
    # 1) Create config
    config = Config()

    # 2) Instantiate the model (where you place it on "meta" or real device is up to you)
    proj = TransformerProjector(config)

    # Example: move the model to CPU
    proj.to("cpu")

    # 3) Initialize weights on CPU
    proj.init_weights_()

    # 4) Forward pass with dummy data
    dummy_input = torch.randn(10, 729, 1152, dtype=torch.float16, device="cpu")
    out = proj(dummy_input)
    print("Output shape after first call:", out.shape)

    # Another pass
    dummy_input2 = torch.randn(12, 729, 1152, dtype=torch.float16, device="cpu")
    out2 = proj(dummy_input2)
    print("Output shape after second call:", out2.shape)
