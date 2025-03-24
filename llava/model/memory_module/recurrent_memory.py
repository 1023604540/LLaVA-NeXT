import math
from typing import Optional, List, Tuple, Union
from einops import rearrange, repeat, pack, unpack

import torch
from torch import nn
from transformers.activations import ACT2FN


class Residual(nn.Module):
    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.layernorm = nn.LayerNorm(output_size, eps=config.mm_layer_norm_eps)
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

        self.k_proj = nn.Linear(config.mm_hidden_size, config.mm_hidden_size)
        self.v_proj = nn.Linear(config.mm_hidden_size, config.mm_hidden_size)
        self.q_proj = nn.Linear(config.mm_hidden_size, config.mm_hidden_size)
        self.dropout = nn.Dropout(config.mm_attention_probs_dropout_prob)

        self.residual = Residual(config.mm_hidden_size, config.mm_hidden_size, config)

    def transpose_for_scores(self, x):  # prepare tensor for multi-head attention
        # B, L, D --> B, H, L, DH   H: num_attention_heads, DH: attention_head_size
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ):
        query = self.transpose_for_scores(self.q_proj(hidden_states))
        #print("query", query.shape)

        if encoder_hidden_states is not None:  # use encoder_hidden_states to initialize key and value
            # cross attention
            if past_key_value is not None:
                # use cache
                key = past_key_value[0]
                value = past_key_value[1]
                attention_mask = encoder_attention_mask
            else:
                key = self.transpose_for_scores(self.k_proj(encoder_hidden_states))
                value = self.transpose_for_scores(self.v_proj(encoder_hidden_states))
                attention_mask = encoder_attention_mask
            # cache key & value for crossattention
            past_key_value = (key, value)
        else:
            # self attention
            if past_key_value is not None:
                # use cache
                key = self.transpose_for_scores(self.k_proj(hidden_states))
                value = self.transpose_for_scores(self.v_proj(hidden_states))
                key = torch.cat([past_key_value[0], key], dim=2)
                value = torch.cat([past_key_value[1], value], dim=2)
            else:
                key = self.transpose_for_scores(self.k_proj(hidden_states))
                value = self.transpose_for_scores(self.v_proj(hidden_states))
        #print("key", key.shape)
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # B, H, N, M

        # TODO position encoding

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # print("with memory token, hidden_state = ", hidden_states.shape)
        # print("attention_scores", attention_scores.shape)
        # print("attention mask", attention_mask.shape)
        if attention_mask is not None:
            attention_scores += attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        output = torch.matmul(attention_probs, value)
        output = output.permute(0, 2, 1, 3).contiguous()
        output_shape = output.size()[:-2] + (self.hidden_size,)
        output = output.view(output_shape)

        output = self.residual(output, hidden_states)

        outputs = (output, attention_probs) if output_attentions else (output,)
        if past_key_value is not None:
            outputs = outputs + (past_key_value,)

        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.selfattention = Attention(config)
        self.crossattention = Attention(config)

        self.mlp = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.mm_intermediate_size),
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
            #encoder_hidden_states: Optional[torch.FloatTensor] = None,
            #encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ):
        if past_key_value is not None:
            self_past_key_value = past_key_value[:2]
        else:
            self_past_key_value = None
        self_attention_outputs = self.selfattention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_past_key_value
        )
        attention_output = self_attention_outputs[0]  # extract the attention output
        outputs = self_attention_outputs[1:]  # save the past_key_value

        # if encoder_hidden_states is not None:
        #     outputs = self_attention_outputs[1:-1]
        #     present_key_value = self_attention_outputs[-1]
        #     if past_key_value is not None:
        #         cross_past_key_value = past_key_value[:2]
        #     else:
        #         cross_past_key_value = None
        #     cross_attention_outputs = self.crossattention(
        #         attention_output,
        #         attention_mask,
        #         head_mask,
        #         encoder_hidden_states,
        #         encoder_attention_mask,
        #         cross_past_key_value,
        #         output_attentions,
        #     )
        #     attention_output = cross_attention_outputs[0]
        #
        #     outputs = outputs + cross_attention_outputs[1:-1]
        #     present_key_value = present_key_value + cross_attention_outputs[-1]

        output = self.ffn(attention_output)
        outputs = (output,) + outputs  # include past_key_value in the outputs tuple
        return outputs


class TransformerProjector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            self.config = Config()  # Use default configuration
        self.layers = nn.ModuleList([TransformerLayer(self.config) for _ in range(self.config.depth)])
        self.num_memory_tokens = self.config.num_memory_tokens
        #self.recurrent_memory = nn.Parameter(torch.randn(self.num_memory_tokens, self.config.mm_hidden_size))
        self.batch_size = None


    def forward(
            self,
            hidden_states: torch.Tensor,
            recurrent_memory: Optional[torch.FloatTensor] = None,
            # attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = False,

    ):
        if recurrent_memory is None:
            recurrent_memory = nn.Parameter(torch.randn(self.num_memory_tokens, self.config.mm_hidden_size))
        recurrent_memory = recurrent_memory.to(hidden_states.dtype)

        # use cache
        next_cache = () if use_cache else None

        assert hidden_states.shape[-1] == self.config.mm_hidden_size  # memory token dimension should match hidden state dimension
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        assert hidden_states.ndim == 3

        self.batch_size = hidden_states.shape[0]
        patch_size = hidden_states.shape[-2] + self.config.num_memory_tokens
        attention_mask = torch.zeros((self.batch_size, self.config.mm_num_attention_heads, patch_size, patch_size))  # attention mask for self-attention
        attention_mask[:, :, self.num_memory_tokens:, :self.num_memory_tokens] = float('-inf')  # hidden_states can not attend to memory
        if recurrent_memory.ndim == 2:
            read_memories = repeat(recurrent_memory, 'n d -> b n d', b=self.batch_size)
        else:
            read_memories = recurrent_memory

        device = hidden_states.device
        read_memories = read_memories.to(device)
        attention_mask = attention_mask.to(device)

        hidden_states, ps = pack([read_memories, hidden_states], 'b * d')  # shape: [B, num_memory_tokens + seq_length, D]
        for i, layer in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states,
                attention_mask,
                layer_head_mask,
                past_key_value,
            )

            hidden_states = layer_outputs[0]  # extracts the first element of the tuple, which is the attention output
            if use_cache:
                next_cache += (layer_outputs[-1],)
        read_memories, hidden_states = unpack(hidden_states, ps, 'b * d')
        # print("recurrent_run_success")
        return read_memories, hidden_states





class Config:
    mm_hidden_size = 1024
    mm_hidden_act = 'relu'
    mm_num_attention_heads = 8
    mm_attention_probs_dropout_prob = 0.1  # Attention dropout
    mm_layer_norm_eps = 1e-12  # LayerNorm epsilon (avoid div by zero)
    mm_hidden_dropout_prob = 0.1  # Residual layer dropout
    mm_intermediate_size = 4096  # Feedforward hidden layer size
    num_memory_tokens = 16  # Number of memory tokens
    depth = 1  # Number of Transformer layers

## Example Usage


# Instantiate the model


# Define input tensors

# assert hidden_states.shape[-1] == config.mm_hidden_size
# patch_size = hidden_states.shape[-2] + config.num_memory_tokens
# encoder_hidden_states = None  # [L=50, P=16, D=1024]
# attention_mask = torch.zeros((50, 8, 48, 48))  # attentions head = 8
# attention_mask = torch.zeros((1, config.mm_num_attention_heads, patch_size, patch_size))  # [B, P, L, L]
# attention_mask[:, :, config.num_memory_tokens:, :config.num_memory_tokens] = float('-inf')  # hidden_states can not attend to memory
# encoder_attention_mask = None  # No masking for cross-attention

#
# # Usage
# hidden_states = torch.randn(16, 1024)  # [L=50, P=16, D=1024]
# old_read_memories = torch.randn(1, 16, 1024)  # [B, num_memory_tokens, D]
# model = TransformerProjector()
# output = model(
#     hidden_states=hidden_states,
#     recurrent_memory=old_read_memories,
#     # encoder_hidden_states=encoder_hidden_states,
#     # encoder_attention_mask=encoder_attention_mask,
# )
#
# # Output shapes
# read_memories, hidden_states = output
# print("Read Memories:", read_memories.shape)  # [B, num_memory_tokens, config.mm_hidden_size]
