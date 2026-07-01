"""
Orthogonal Latent Space: manifold-structured latent vector with
orthogonal slicing and L2 hypersphere constraints.

z = [z_lora, z_film, z_query]

Each slice is independently L2-normalized to the unit hypersphere,
preventing gradient entanglement across control paths and ensuring
that perturbations in one path do not cause scale explosions in others.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class OrthogonalLatentSpace(nn.Module):
    """
    Global latent vector z with orthogonal subspace decomposition.

    Implements the orthogonal-sliced constraint:
        z_k = z_k / ||z_k||_2   for k in {lora, film, query}

    This L2 normalization projects each control slice onto the unit
    hypersphere, preventing scale drift and gradient coupling between
    independent control paths.
    """

    def __init__(self, d_lora: int, d_film: int, d_query: int):
        """
        Args:
            d_lora:  latent dimension for LoRA control path
            d_film:  latent dimension for FiLM control path
            d_query: latent dimension for Query residual control path
        """
        super().__init__()
        self.d_lora = d_lora
        self.d_film = d_film
        self.d_query = d_query
        total = d_lora + d_film + d_query

        self.z = nn.Parameter(torch.randn(total) * 0.02)

        lo = 0
        hi = d_lora
        self.register_buffer("_lora_idx", torch.arange(lo, hi))
        lo = hi
        hi = lo + d_film
        self.register_buffer("_film_idx", torch.arange(lo, hi))
        lo = hi
        hi = lo + d_query
        self.register_buffer("_query_idx", torch.arange(lo, hi))

    def _slice_and_norm(self, idx: torch.Tensor) -> torch.Tensor:
        z_slice = self.z[idx]
        return F.normalize(z_slice, p=2, dim=0)

    def get_z_lora(self) -> torch.Tensor:
        """L2-normalized LoRA control slice (d_lora,)."""
        return self._slice_and_norm(self._lora_idx)

    def get_z_film(self) -> torch.Tensor:
        """L2-normalized FiLM control slice (d_film,)."""
        return self._slice_and_norm(self._film_idx)

    def get_z_query(self) -> torch.Tensor:
        """L2-normalized Query control slice (d_query,)."""
        return self._slice_and_norm(self._query_idx)

    def get_all_slices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z_lora, z_film, z_query) all L2-normalized."""
        return self.get_z_lora(), self.get_z_film(), self.get_z_query()

    def get_full_z(self) -> torch.Tensor:
        """Raw (pre-normalization) full latent vector (total_dim,)."""
        return self.z
