"""
Alignment Loss (L_align).

    L_align = Sigma_k (1 - cos(z_k, W_mean_k))

Per-control-path cosine similarity between latent slices and mean
orthogonal matrix directions. Numerical protections against zero-norm
W_mean vectors (common for large orthonormal matrices).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class AlignmentLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        z_lora: torch.Tensor,
        z_film: torch.Tensor,
        z_query: torch.Tensor,
        orth_lora: List[torch.Tensor],
        orth_film: List[torch.Tensor],
        orth_query: List[torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.zeros(1, device=z_lora.device, requires_grad=True).squeeze()

        for z, orth_list in [(z_lora, orth_lora), (z_film, orth_film), (z_query, orth_query)]:
            if len(orth_list) == 0:
                continue
            w_means = []
            for W in orth_list:
                w_mean = W.mean(dim=0)
                w_norm = w_mean.norm(p=2)
                if w_norm > 1e-8:
                    w_means.append(w_mean / w_norm)
                else:
                    w_means.append(torch.zeros_like(w_mean))
            W_mean = torch.stack(w_means).mean(dim=0)
            W_norm = W_mean.norm(p=2)
            if W_norm > 1e-8:
                W_mean = W_mean / W_norm
            else:
                W_mean = torch.zeros_like(W_mean)
            z_norm = z / (z.norm(p=2) + 1e-8)
            loss = loss + 1.0 - torch.nan_to_num(torch.dot(z_norm, W_mean), nan=0.0)

        return loss
