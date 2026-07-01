"""
LoRA Generator (Partial Low-Rank Injection).

For each target layer, generates low-rank matrices U, V from the shared
LoRA latent slice z_lora using the additive modulation equation:

    U(z_lora) = sigma( gamma_u * W_u_orth @ z_lora + beta_u )
    V(z_lora) = sigma( gamma_v * W_v_orth @ z_lora + beta_v )

The effective weight at injection time is:
    W_adaptive = W_pretrained + U @ V^T

This reduces parameter generation cost from O(m*n) to O(r*(m+n)).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from .orthogonal_base import OrthogonalBase as OB


class LoRALayer(nn.Module):
    """
    Per-layer LoRA generation node.

    Given a fixed z_lora, produces (U, V) pair for one specific layer.
    Each layer gets independent W_orth, gamma, beta to preserve capacity.

    Shape convention:
        U: (m, r)   -- output-side low-rank factor
        V: (n, r)   -- input-side low-rank factor
        W_effective: (m, n) where m = out_features, n = in_features_flat
    """

    def __init__(self, d_lora: int, m: int, n: int, r: int):
        """
        Args:
            d_lora: dimension of the LoRA latent slice
            m:      output dimension of the target weight
            n:      flattened input dimension of the target weight
            r:      LoRA rank
        """
        super().__init__()
        self.m = m
        self.n = n
        self.r = r

        self.W_u_orth = OB.init_orthogonal_fixed(d_lora, m * r)
        self.W_v_orth = OB.init_orthogonal_fixed(d_lora, n * r)

        self.gamma_u = OB.trainable_scalar(1.0)
        self.beta_u = OB.trainable_bias(m * r)
        self.gamma_v = OB.trainable_scalar(1.0)
        self.beta_v = OB.trainable_bias(n * r)

    def forward(self, z_lora: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_lora: (d_lora,) L2-normalized latent slice

        Returns:
            U: (m, r), V: (n, r)
        """
        u_flat = self.gamma_u * (self.W_u_orth @ z_lora) + self.beta_u
        U = torch.sigmoid(u_flat).view(self.m, self.r)

        v_flat = self.gamma_v * (self.W_v_orth @ z_lora) + self.beta_v
        V = torch.sigmoid(v_flat).view(self.n, self.r)

        return U, V


class LoRAGenerator(nn.Module):
    """
    Manages LoRA injection for all registered layers.

    Configuration is a list of dicts specifying each target layer:
        {'name': str, 'm': int, 'n': int}

    All layers share the same z_lora slice but have independent
    mapping parameters, enforcing orthogonal control.

    Usage:
        gen = LoRAGenerator(d_lora=256, layer_configs=[...], rank=16)
        z_lora = latent_space.get_z_lora()
        lora_map = gen(z_lora)
        # lora_map['backbone.layer3.0.conv1'] -> (U, V)
    """

    def __init__(
        self, d_lora: int, layer_configs: List[Dict], rank: int = 16
    ):
        super().__init__()
        self.layer_names = []
        layers = {}
        for cfg in layer_configs:
            name = cfg["name"]
            safe_name = name.replace(".", "_")
            self.layer_names.append(name)
            layers[safe_name] = LoRALayer(d_lora, cfg["m"], cfg["n"], rank)
        self.layers = nn.ModuleDict(layers)
        self._name_map = {n: n.replace(".", "_") for n in self.layer_names}

    def forward(self, z_lora: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return {name: self.layers[self._name_map[name]](z_lora) for name in self.layer_names}
