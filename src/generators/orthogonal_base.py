"""
Orthogonal initialization base for all mapping generators.
Provides shared utility for creating orthogonal-initialized,
gradient-frozen mapping matrices that constrain the solution space
as required by the Weight-Manifold Hypothesis.
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class OrthogonalBase:
    """
    Mixin-style base providing orthogonal initialization utilities.

    All mapping matrices W_orth must be initialized orthogonally and
    kept frozen (requires_grad=False) to satisfy the Mapping Theorem's
    requirement of a constrained solution space.
    """

    @staticmethod
    def init_orthogonal_fixed(
        in_features: int, out_features: int, gain: float = 1.0
    ) -> nn.Parameter:
        """
        Create an orthogonal matrix mapping from latent dim to output dim.

        Args:
            in_features:  input dimension  (latent dimension d)
            out_features: output dimension (target parameter count p)

        Returns:
            nn.Parameter with requires_grad=False, initialized orthogonally.
        """
        weight = nn.Parameter(torch.empty(out_features, in_features))
        init.orthogonal_(weight, gain=gain)
        weight.requires_grad_(False)
        return weight

    @staticmethod
    def trainable_scalar(init_val: float = 1.0) -> nn.Parameter:
        """Create a learnable scalar modulation factor gamma."""
        return nn.Parameter(torch.tensor(init_val, dtype=torch.float32))

    @staticmethod
    def trainable_bias(dim: int) -> nn.Parameter:
        """Create a learnable bias vector for additive modulation."""
        return nn.Parameter(torch.zeros(dim))
