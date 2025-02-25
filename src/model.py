# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import hyperparameters from hyperparams.py
from src.hyperparam import (
    num_classes, patch_size, img_size, in_channels, num_heads, dropout,
    hidden_dim, activation, num_encoders, embed_dim, num_patches, device
)

# -------------------------
# Patch Embedding Module
# -------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2)  # Flatten height and width into one dimension
        )
        # Learnable class token: shape (1, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Positional embeddings for (num_patches + 1) tokens
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (B, in_channels, H, W)
        B = x.shape[0]
        x = self.patcher(x)           # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)         # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)          # (B, num_patches+1, embed_dim)
        x = x + self.position_embeddings             # Add positional embeddings
        x = self.dropout(x)
        return x

# -------------------------
# Multi-Head Self-Attention
# -------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = dropout(attn)
        output = torch.matmul(attn, value)
        return output, attn

    def forward(self, q, k, v, mask=None):
        # q, k, v: (B, seq_len, d_model)
        B, seq_len, _ = q.size()
        query = self.w_q(q)  # (B, seq_len, d_model)
        key   = self.w_k(k)
        value = self.w_v(v)
        # Reshape to (B, h, seq_len, d_k)
        query = query.view(B, seq_len, self.h, self.d_k).transpose(1,2)
        key   = key.view(B, seq_len, self.h, self.d_k).transpose(1,2)
        value = value.view(B, seq_len, self.h, self.d_k).transpose(1,2)
        # Scaled dot-product attention
        x, attn = self.attention(query, key, value, mask, self.dropout)
        # Reshape back to (B, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(B, seq_len, self.d_model)
        return self.w_o(x)

# -------------------------
# Feed-Forward Block
# -------------------------
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation="gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# -------------------------
# Encoder Layer
# -------------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout, activation=activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention sublayer with residual connection
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        # Feed-forward sublayer with residual connection
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, h, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# -------------------------
# Vision Transformer (ViT)
# -------------------------
class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes,
                 embed_dim, num_encoders, num_heads, hidden_dim, dropout):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        self.encoder = Encoder(num_encoders, embed_dim, num_heads, hidden_dim, dropout)
        # MLP head on the class token (first token)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x, mask=None):
        # x: (B, in_channels, img_size, img_size)
        x = self.patch_embedding(x)  # (B, num_patches+1, embed_dim)
        x = self.encoder(x, mask)      # (B, num_patches+1, embed_dim)
        cls_token = x[:, 0, :]         # (B, embed_dim)
        logits = self.mlp_head(cls_token)
        return logits