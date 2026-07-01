"""
L2 normalization utility for latent space projection.
"""

import torch
import torch.nn.functional as F


def l2_normalize(z: torch.Tensor, dim: int = 0, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize along specified dimension."""
    return F.normalize(z, p=2, dim=dim, eps=eps)
