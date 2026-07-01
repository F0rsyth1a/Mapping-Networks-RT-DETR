"""
Multimodal fusion interface (RGB-D extension).

For RGB-D inputs, the mapping network generates cross-modal fusion
coefficients alpha from the latent vector:

    alpha = g_fuse(z)
    F_fused = alpha * F_rgb + (1 - alpha) * F_depth

The interface defines an abstract base with hot-swappable backbone
support for 2D (ResNet) and 3D (PointNet++ / SparseConv) modalities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BackboneInterface(ABC, nn.Module):
    """Abstract backbone interface supporting 2D and 3D variants."""

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> dict:
        """Return multi-scale feature dict {'S3', 'S4', 'S5'}."""
        ...

    @property
    @abstractmethod
    def out_channels(self) -> dict:
        """Return {'S3': int, 'S4': int, 'S5': int}."""
        ...


class MultimodalFusion(nn.Module):
    """
    Cross-modal fusion coefficient generator.

    Generates per-channel fusion weights alpha from the latent vector,
    enabling adaptive blending of RGB and depth features.

        alpha = sigmoid( W_fuse @ z + b_fuse )
        F_fused = alpha * F_rgb + (1 - alpha) * F_depth
    """

    def __init__(self, d_latent: int, num_channels: int):
        """
        Args:
            d_latent:     dimension of the latent slice for fusion
            num_channels: number of feature channels to fuse
        """
        super().__init__()
        self.num_channels = num_channels

        self.W_fuse = nn.Parameter(torch.empty(num_channels, d_latent))
        nn.init.orthogonal_(self.W_fuse)
        self.W_fuse.requires_grad_(False)

        self.b_fuse = nn.Parameter(torch.zeros(num_channels))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (d_latent,) latent slice for fusion control

        Returns:
            alpha: (num_channels,) per-channel fusion coefficients in [0, 1]
        """
        return torch.sigmoid(self.W_fuse @ z + self.b_fuse)

    @staticmethod
    def fuse(
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply weighted fusion: F = alpha * F_rgb + (1-alpha) * F_depth

        Args:
            rgb_features:   (B, C, H, W)
            depth_features: (B, C, H, W)
            alpha:          (C,) per-channel weights

        Returns:
            fused: (B, C, H, W)
        """
        a = alpha.view(1, -1, 1, 1)
        return a * rgb_features + (1 - a) * depth_features
