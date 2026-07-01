"""
Stability Loss (L_stab).

Punishes output deviation under latent space perturbation:

    L_stab = E[ || f(z + ε) - f(z) ||_2^2 ]

where ε ~ N(0, σ^2 I) is injected Gaussian noise.

This enforces local Lipschitz continuity of the mapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StabilityLoss(nn.Module):
    """
    Stability regularization via latent perturbation.

    Measures MSE between outputs of the full model with and without
    noise injected into the latent vector z.
    """

    def __init__(self, noise_sigma: float = 0.01):
        super().__init__()
        self.noise_sigma = noise_sigma

    def forward(
        self,
        output_clean: dict,
        output_noisy: dict,
    ) -> torch.Tensor:
        """
        Args:
            output_clean: model output dict for f(z)
            output_noisy: model output dict for f(z + ε)

        Returns:
            L_stab: scalar
        """
        loss = torch.tensor(0.0, device=output_clean["pred_logits"].device)

        for key in output_clean:
            if key in output_noisy:
                loss = loss + ((output_clean[key] - output_noisy[key]) ** 2).mean()

        return loss

    def perturb_and_forward(
        self,
        model_fn,
        x: torch.Tensor,
        z_full: torch.Tensor,
        **kwargs,
    ) -> dict:
        """
        Convenience: compute f(z + ε) given model forward function.

        Args:
            model_fn: callable (x, z_full, **kwargs) -> output_dict
            x:        input tensor
            z_full:   full latent vector
            **kwargs: passed to model_fn

        Returns:
            output dict for perturbed forward
        """
        noise = torch.randn_like(z_full) * self.noise_sigma
        return model_fn(x, z_full=z_full + noise, **kwargs)
