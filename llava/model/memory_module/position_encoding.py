"""
temporal_positional_encoding.py

Script to add temporal positional encoding to visual feature vectors.

Visual feature tensor shape: (T, N, C) or (B, T, N, C).
"""

import torch
import torch.nn as nn
import math

class TemporalPositionalEncoding(nn.Module):
    def __init__(self, max_frames, embed_dim, learnable=True):
        """
        Args:
            num_frames (int): number of frames (T).
            embed_dim (int): feature dimension (C).
            learnable (bool): if True, use nn.Embedding; else, use fixed sin-cos.
        """
        super().__init__()
        self.max_frames = max_frames
        self.embed_dim = embed_dim
        self.learnable = learnable

        if learnable:
            self.frame_embed = nn.Embedding(max_frames, embed_dim)
        else:
            # Create fixed sin-cos positional encodings and register as buffer
            pe = torch.zeros(max_frames, embed_dim)
            position = torch.arange(0, max_frames).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('frame_embed', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): shape (T, N, C) or (B, T, N, C)
        Returns:
            Tensor: same shape as x, with temporal embeddings added.
        """
        # Determine frame indices
        if x.dim() == 3:
            # x: (T, N, C)
            T, N, C = x.shape
            if T > self.max_frames:
                raise ValueError(f"T={T} exceeds max_frames={self.max_frames} in TemporalPositionalEncoding.")
            t_idxs = torch.arange(T, device=x.device)
            # Retrieve positional encodings
            if self.learnable:
                pe = self.frame_embed(t_idxs)      # (T, C)
            else:
                pe = self.frame_embed[t_idxs]     # (T, C)
            # Add and broadcast over patches
            return x + pe[:, None, :]

        elif x.dim() == 4:
            # x: (B, T, N, C)
            B, T, N, C = x.shape
            t_idxs = torch.arange(T, device=x.device)
            if self.learnable:
                pe = self.frame_embed(t_idxs)      # (T, C)
            else:
                pe = self.frame_embed[t_idxs]     # (T, C)
            # Add and broadcast over batch and patches
            return x + pe[None, :, None, :]

        else:
            raise ValueError(f'Expected 3D or 4D input, got {x.dim()}D.')

if __name__ == "__main__":
    # Example usage
    T, N, C = 8, 196, 512
    features = torch.randn(T, N, C)
    # Create temporal encoder (fixed sin-cos)
    temp_enc = TemporalPositionalEncoding(max_frames=T, embed_dim=C, learnable=False)
    encoded_features = temp_enc(features)
    print("Encoded features shape:", encoded_features.shape)
