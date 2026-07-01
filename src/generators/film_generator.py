"""
FiLM Generator (Feature-wise Linear Modulation).

Operates on CCFF fusion nodes in the encoder. For each fusion node,
generates per-channel affine parameters from the FiLM latent slice:

    gamma = W_gamma_orth @ z_film + b_gamma   ->  (C,)
    beta  = W_beta_orth  @ z_film + b_beta    ->  (C,)

Applied post-fusion as a channel-wise affine transform:

    F'[c, :, :] = gamma[c] * F[c, :, :] + beta[c]

This preserves spatial topology by using per-channel (not scalar) parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from .orthogonal_base import OrthogonalBase as OB


class FiLMLayer(nn.Module):
    """
    Per-fusion-node FiLM parameter generator.

    Uses orthogonal-initialized frozen mapping matrices with
    trainable biases for stable optimization.
    """

    def __init__(self, d_film: int, C: int):
        """
        Args:
            d_film: dimension of the FiLM latent slice
            C:      number of channels in the target feature map
        """
        super().__init__()
        self.C = C

        self.W_gamma_orth = OB.init_orthogonal_fixed(d_film, C)
        self.W_beta_orth = OB.init_orthogonal_fixed(d_film, C)

        self.b_gamma = OB.trainable_bias(C)
        self.b_beta = OB.trainable_bias(C)

    def forward(self, z_film: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_film: (d_film,) L2-normalized latent slice

        Returns:
            gamma: (C,), beta: (C,) per-channel affine parameters
        """
        gamma = self.W_gamma_orth @ z_film + self.b_gamma
        beta = self.W_beta_orth @ z_film + self.b_beta
        return gamma, beta

    @staticmethod
    def apply(feature: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation to a feature map.

        Args:
            feature: (B, C, H, W)
            gamma:   (C,) channel-wise scale
            beta:    (C,) channel-wise shift

        Returns:
            modulated: (B, C, H, W)
        """
        gamma = gamma.view(1, -1, 1, 1)
        beta = beta.view(1, -1, 1, 1)
        return gamma * feature + beta


class FiLMGenerator(nn.Module):
    """
    Manages FiLM generation for all CCFF fusion nodes.

    Configuration: list of dicts with 'name' and 'C' (channels).
    All nodes share z_film but each has its own mapping parameters.
    """

    def __init__(self, d_film: int, node_configs: List[Dict]):
        """
        Args:
            d_film:       FiLM latent slice dimension
            node_configs: list of {'name': str, 'C': int}
        """
        super().__init__()
        self.node_names = []
        nodes = {}
        for cfg in node_configs:
            name = cfg["name"]
            self.node_names.append(name)
            nodes[name] = FiLMLayer(d_film, cfg["C"])
        self.nodes = nn.ModuleDict(nodes)

    def forward(self, z_film: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return {name: node(z_film) for name, node in self.nodes.items()}
