import math
from typing import Optional, List, Tuple, Union
from einops import rearrange, repeat, pack, unpack

import torch
from torch import nn
from transformers.activations import ACT2FN


class Config:
    mm_hidden_size = 1152
    mm_hidden_act = 'relu'
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
        self.dense = nn.Linear(input_size, output_size, dtype=config.mm_dtype)
        self.layernorm = nn.LayerNorm(output_size, eps=config.mm_layer_norm_eps, dtype=config.mm_dtype)
        self.dropout = nn.Dropout(config.mm_hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_tensor: torch.Tensor,
    ):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.mm_hidden_size
        self.num_attention_heads = config.mm_num_attention_heads
        self.attention_head_size = config.mm_hidden_size // config.mm_num_attention_heads

        assert config.mm_hidden_size % config.mm_num_attention_heads == 0

        self.k_proj = nn.Linear(config.mm_hidden_size, config.mm_hidden_size, dtype=config.mm_dtype)
        self.v_proj = nn.Linear(config.mm_hidden_size, config.mm_hidden_size, dtype=config.mm_dtype)
        self.q_proj = nn.Linear(config.mm_hidden_size, config.mm_hidden_size, dtype=config.mm_dtype)
        self.dropout = nn.Dropout(config.mm_attention_probs_dropout_prob)

        self.residual = Residual(config.mm_hidden_size, config.mm_hidden_size, config)

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
        # Key.shape (B, H, Lq*P, DH)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # (B, H, Lq*P(Q), Lq*P(K/V))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores += attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)  # (B, H, Lq*P(Q), Lq*P(K/V))
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context = torch.matmul(attention_probs, value)  # (B, H, Lq*P, DH)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)
        context = context.view(new_context_shape)  # (B, Lq*P, D)

        # Residual
        output = self.residual(context, hidden_states)  # (B, Lq*P, D)

        outputs = (output, attention_probs) if output_attentions else (output,)
        if past_key_value is not None:
            outputs = outputs + (past_key_value,)

        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self_attention = Attention(config)
        self.cross_attention = Attention(config)

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
        2) (Optional) Cross-Attention
        3) Feed-forward
        """
        if past_key_value is not None:
            self_past_key_value = past_key_value[:2]
        else:
            self_past_key_value = None

        # (1) Self-Attention
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_past_key_value
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # e.g. attention_probs, (past_key_value)...

        # (2) Optional cross-attention if kv_hidden_states given
        if kv_hidden_states is not None:
            cross_attn_out = self.cross_attention(
                attention_output,
                attention_mask=None,
                head_mask=head_mask,
                kv_hidden_states=kv_hidden_states,
                kv_attention_mask=kv_attention_mask,
                output_attentions=output_attentions,
            )
            attention_output = cross_attn_out[0]
            # You could handle more outputs here if desired.

        # (3) Feed-forward
        layer_output = self.ffn(attention_output)
        return (layer_output,) + outputs



class TransformerProjector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else Config()

        # Main Transformer layers (self-attention for the input + memory tokens)
        self.layers = nn.ModuleList(
            [TransformerLayer(self.config) for _ in range(self.config.depth)]
        )

        # NEW cross-attention layer for memory retrieval
        self.memory_retrieval_attention = Attention(self.config)

        self.num_memory_tokens = self.config.num_memory_tokens
        self.hidden_size = self.config.mm_hidden_size
        self.patch_size = self.config.patch_size

        # The "initial" memory tokens (m0): shape (n, patch, d).
        self.initial_memory = nn.Parameter(
            torch.randn(self.num_memory_tokens, self.patch_size, self.hidden_size)
        )

        # This will store all previous memory tokens across calls
        self.memory_cache: List[torch.Tensor] = []

    def _update_memory_tokens_with_cache(
        self,
        current_memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Given current memory tokens mᵢ of shape (n, patch, d),
        do one cross-attention step with all memory_cache as K/V to get mᵢ₊₁.
        """
        if len(self.memory_cache) == 0:
            # No previous memory: just return the current memory as-is
            return current_memory

        # Combine all previous memory tokens into a single large tensor
        # Suppose each item in memory_cache is shape (n, patch, d).
        # We'll cat them on dim=0 => shape (N*n, patch, d).
        past_memory = torch.cat(self.memory_cache, dim=0).unsqueeze(0)  # (1, N*n, patch, d)
        query_mem = current_memory.unsqueeze(0)  # (1, n, patch, d)

        # Flatten the "patch" dimension into the sequence dimension for standard attention
        B, Lq, P, D = query_mem.shape  # B=1, Lq=n, P=patch, D=hidden
        query_2d = query_mem.view(B, Lq * P, D)

        B2, Lk, Pk, D2 = past_memory.shape  # B2=1, Lk=N*n, Pk=patch
        keyval_2d = past_memory.view(B2, Lk * Pk, D2)

        # Use the dedicated cross-attention layer
        cross_attn_out = self.memory_retrieval_attention(
            hidden_states=query_2d,
            kv_hidden_states=keyval_2d,
            attention_mask=None,
            head_mask=None,
            output_attentions=False
        )
        updated_2d = cross_attn_out[0]  # shape (1, Lq*P, D)

        # Reshape back to (1, n, patch, d)
        updated_4d = updated_2d.view(B, Lq, P, D)
        return updated_4d.squeeze(0)  # => (n, patch, d)

    def forward(
        self,
        image_features: torch.Tensor,
        head_mask: Optional[torch.FloatTensor] = None,
    ):
        """
        image_features: (frames=F, patch=P, dimension=D)

        Steps:
         1) If memory_cache is empty, start with self.initial_memory; else use the last memory_cache entry.
         2) Self-retrieval: update them by cross-attending to the entire memory_cache.
         3) Prepend the updated memory tokens => shape (F+n, P, D).
         4) Run through {self.depth} layers of self-attention (no mask).
         5) Split out the first n tokens => new memory tokens.
         6) Append them to memory_cache, return the rest as “transformed image features.”
        """
        device = image_features.device
        dtype = image_features.dtype

        # (1) Decide what your "current memory" is:
        if len(self.memory_cache) == 0:
            # First call: use initial_memory
            current_memory = self.initial_memory.to(device=device, dtype=dtype)
            if torch.isnan(current_memory).any():
                raise ValueError("NaNs detected in current_memory!")
        else:
            # Later calls: use last updated memory from memory_cache
            current_memory = self.memory_cache[-1].to(device=device, dtype=dtype)
        if len(self.memory_cache) > 1:
            print("Memory cache length:", len(self.memory_cache))
            # (2) Cross-attention update with entire memory_cache
            current_memory = self._update_memory_tokens_with_cache(current_memory)

        # (3) Prepend memory to new image features => shape (F+n, P, D)
        combined = torch.cat([current_memory, image_features], dim=0)

        # Insert a batch dimension (B=1), then flatten patch into sequence for self-attn
        combined = combined.unsqueeze(0)  # => (1, F+n, P, D)
        B, L, P_, D_ = combined.shape
        combined_2d = combined.view(B, L * P_, D_)  # => (1, (F+n)*P, D)

        # (4) Pass through self.depth layers of self-attention
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
            hidden_states = layer_outputs[0]  # (1, (F+n)*P, D)

        if torch.isnan(hidden_states).any():
            raise ValueError("NaNs detected in hidden_states!")

        # (5) Reshape back, split out memory vs. image
        hidden_4d = hidden_states.view(B, L, P_, D_)  # => (1, F+n, P, D)
        new_memory_tokens = hidden_4d[:, :self.num_memory_tokens, :, :]  # => (1, n, P, D)
        updated_image_features = hidden_4d[:, self.num_memory_tokens:, :, :]  # => (1, F, P, D)

        # (6) Save the new memory tokens to memory_cache for next time
        self.memory_cache.append(new_memory_tokens.squeeze(0).detach())

        # Return just the updated image portion
        return new_memory_tokens.squeeze(0)


#
# EXAMPLE USAGE
#
if __name__ == "__main__":
    config = Config()
    proj = TransformerProjector(config)

    # Suppose we have input image_features with shape (F=4, P=729, D=1152)
    dummy_input = torch.randn(10, 729, 1152)

    out = proj(dummy_input)
    print("Output shape after first call:", out.shape)  # => (8, 729, 1152)

    # Now the cache has 1 set of memory tokens.
    # Next time, we use the memory_cache[-1] from the last pass, not the initial_memory.
    dummy_input2 = torch.randn(12, 729, 1152)
    out2 = proj(dummy_input2)
    print("Output shape after second call:", out2.shape)  # => (8, 729, 1152)
