"""
Smoothness Loss (L_smooth).

Uses Hutchinson trace estimator (JVP-based) to compute the Frobenius
norm of the generator Jacobians without instantiating the full Jacobian
matrix. This enforces C^2 continuity of the mapping network.

    L_smooth ≈ Σ_k E_v[ ||∇_{z_k} (v^T g_k(z_k))||^2 ]

Imported as a convenience wrapper around hutchinson_jvp.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Callable, Dict, Optional

from .hutchinson_jvp import compute_smoothness_hutchinson


class SmoothnessLoss(nn.Module):
    """
    Smoothness regularization via Hutchinson JVP estimation.

    Applies the trace estimator to each independent control path
    (LoRA, FiLM, Query) and sums the results.
    """

    def __init__(self, num_samples: int = 1):
        super().__init__()
        self.num_samples = num_samples

    def forward(
        self,
        generators: Dict[str, Callable],
        z_slices: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            generators: {'lora': fn, 'film': fn, 'query': fn}
                        Each fn maps z_k -> output dict/tuple.
            z_slices:   {'lora': z_lora, 'film': z_film, 'query': z_query}

        Returns:
            L_smooth: scalar
        """
        total, _ = compute_smoothness_hutchinson(
            generators, z_slices, self.num_samples
        )
        return total
