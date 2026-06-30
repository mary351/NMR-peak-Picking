import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")    # swap to "TkAgg" for interactive windows
import matplotlib.pyplot as plt
from scipy.ndimage import label as ndlabel, maximum_filter, gaussian_filter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False
    print("Warning: einops not installed — ViT unavailable.  pip install einops")



def positional_embd_sin_cos(h, w, dim, temp=10_000, dtype=torch.float32):
    """Fixed 2-D sinusoidal positional embedding for an (h × w) patch grid."""
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert dim % 4 == 0, "dim must be divisible by 4"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temp ** omega)
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Multi-head self-attention.
    BUG 1 FIXED: super().__init__() was missing → all sub-modules were plain
                 Python attributes; the optimizer saw zero parameters.
    BUG 2 FIXED: inner_dim = dim * heads → must be dim_head * heads.
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()                      # FIX 1
        inner_dim    = dim_head * heads         # FIX 2
        self.heads   = heads
        self.scale   = dim_head ** -0.5
        self.norm    = nn.LayerNorm(dim)
        self.attend  = nn.Softmax(dim=-1)
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out  = nn.Linear(inner_dim, dim,      bias=False)

    def forward(self, x):
        x       = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
                      (q, k, v))
        attn = self.attend(torch.matmul(q, k.transpose(-1, -2)) * self.scale)
        out  = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm   = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([Attention(dim, heads=heads, dim_head=dim_head),
                           FeedForward(dim, mlp_dim)])
            for _ in range(depth)
        ])
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x)   + x
        return self.norm(x)


class ViT_PP(nn.Module):
    """
    Vision Transformer for 2-D NMR peak picking.

    Input : Tensor (B, 1, H, W)
    Output: Tensor (B, 1, H, W) — per-pixel peak probability in [0, 1]

    How it works
    ------------
    1. Divide the (H, W) spectrum into non-overlapping (ph × pw) patches.
    2. Flatten each patch → linear projection → sequence of tokens.
    3. Add 2-D sinusoidal positional embedding so the model knows where
       each patch came from
    4. Feed through Transformer encoder → one feature vector per token.
    5. Project each token → 1 scalar score.
    6. Reshape token scores to patch grid (gh × gw), then bilinearly
       upsample back to (H, W) → per-pixel probability heatmap.

    """

    def __init__(self, *, spectra_size, patch_size, dim, depth, heads,
                 mlp_dim, channels=1, dim_head=64):
        super().__init__()
        if not HAS_EINOPS:
            raise ImportError("pip install einops")

        sh, sw = spectra_size
        ph, pw = patch_size
        assert sh % ph == 0 and sw % pw == 0, \
            "spectra_size must be exactly divisible by patch_size"

        self.ph, self.pw = ph, pw
        self.gh = sh // ph   # number of patches along ¹H axis
        self.gw = sw // pw   # number of patches along ¹⁵N axis
        patch_dim = channels * ph * pw

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=ph, p2=pw),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.register_buffer(
            "pos_embedding",
            positional_embd_sin_cos(h=self.gh, w=self.gw, dim=dim),
        )

        self.transformer = Transformer(dim=dim, depth=depth, heads=heads,
                                       dim_head=dim_head, mlp_dim=mlp_dim)
        self.patch_head  = nn.Linear(dim, 1)
        self.upsample    = nn.Upsample(scale_factor=(ph, pw),
                                       mode="bilinear", align_corners=False)
        self.sigmoid     = nn.Sigmoid()

    def forward(self, spectra):
        # FIX 4: output shape is (B, 1, H, W), one probability per pixel
        x = self.to_patch_embedding(spectra)          # (B, gh*gw, dim)
        x = x + self.pos_embedding.to(dtype=x.dtype) # add positional info
        x = self.transformer(x)                       # (B, gh*gw, dim)

        scores = self.patch_head(x)                   # (B, gh*gw, 1)
        scores = scores.permute(0, 2, 1)              # (B, 1, gh*gw)
        scores = scores.reshape(-1, 1, self.gh, self.gw)  # (B, 1, gh, gw)
        heatmap = self.upsample(scores)               # (B, 1, H, W)
        return self.sigmoid(heatmap)


# ================================================================
# SECTION 5 — 2-D HEATMAP LABELS, DATASET, TRAINING
# ================================================================

def make_label_heatmap(peak_positions, height, width, sigma_y=8.0, sigma_x=3.0):
    """
    Build a (H, W) float32 Gaussian heatmap from peak (row, col) positions.

    Why Gaussian labels instead of binary point labels?
    ---------------------------------------------------
    A single-pixel label at each peak provides almost no gradient
    signal during training (99.9% of pixels would be 0).  A Gaussian
    blob spreads the target over ~σ² pixels, providing a smooth
    gradient field that trains much more stably.

    sigma_y / sigma_x are chosen to match the expected peak lineshape.
    """
    label = np.zeros((height, width), dtype=np.float32)
    y     = np.arange(height)
    x     = np.arange(width)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    for (cy, cx) in peak_positions:
        blob  = np.exp(
            -((yy - cy) ** 2) / (2 * sigma_y ** 2)
            -((xx - cx) ** 2) / (2 * sigma_x ** 2)
        )
        label = np.maximum(label, blob)
    return label