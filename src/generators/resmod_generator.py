"""
Residual Modulated Query Generator.

Generates a global manifold prior for the decoder query initialization.

    delta_Q = sigma( W_q_orth @ z_query + b_q )

This residual is added to the uncertainty-minimal selected encoder
features and then layer-normalized:

    Q_final = LayerNorm( Q_init + delta_Q )

This preserves the quality of the uncertainty-minimal query selection
while injecting manifold-guided prior information.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .orthogonal_base import OrthogonalBase as OB


class ResModGenerator(nn.Module):
    """
    Generates residual modulation delta_Q for decoder query initialization.

    The orthogonal mapping ensures the generated prior lives on a
    well-conditioned submanifold of the query space.
    """

    def __init__(self, d_query: int, num_queries: int, hidden_dim: int):
        """
        Args:
            d_query:     query latent slice dimension
            num_queries: number of object queries (default: 300)
            hidden_dim:  decoder hidden dimension (default: 256)
        """
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        out_dim = num_queries * hidden_dim

        self.W_q_orth = OB.init_orthogonal_fixed(d_query, out_dim)
        self.b_q = OB.trainable_bias(out_dim)

    def forward(self, z_query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_query: (d_query,) L2-normalized latent slice

        Returns:
            delta_Q: (num_queries, hidden_dim) residual modulation
        """
        flat = torch.sigmoid(self.W_q_orth @ z_query + self.b_q)
        return flat.view(self.num_queries, self.hidden_dim)
