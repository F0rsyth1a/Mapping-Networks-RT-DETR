"""
RT-DETR Encoder: AIFI + CCFF with LoRA and FiLM injection.

AIFI (Attention-based Intra-scale Feature Interaction):
    Single-scale Transformer encoder applied to S5 features.
    QKV projections accept LoRA injection.

CCFF (CNN-based Cross-scale Feature Fusion):
    Multi-scale fusion blocks with FiLM modulation after each
    fusion node:  F' = gamma * F + beta.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

from .backbone import FrozenConv2d, FrozenBatchNorm2d


# ---------------------------------------------------------------------------
# LoRA-injectable Linear layer
# ---------------------------------------------------------------------------


class FrozenLinear(nn.Module):
    """Linear layer with frozen weight and bias. Optional init_gain to control output scale."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_gain: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, mode="fan_in", nonlinearity="relu")
        self.weight.data.mul_(init_gain)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class LoRALinear(nn.Module):
    """Linear layer with LoRA delta: W_eff = W_frozen + U @ V^T."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_frozen = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight_frozen, mode="fan_in", nonlinearity="relu")
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(
        self, x: torch.Tensor, U: Optional[torch.Tensor] = None, V: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        weight = self.weight_frozen
        if U is not None and V is not None:
            weight = weight + (U @ V.T)
        return F.linear(x, weight, self.bias)


# ---------------------------------------------------------------------------
# AIFI: Single-scale Transformer Encoder
# ---------------------------------------------------------------------------


class AIFI(nn.Module):
    """
    Attention-based Intra-scale Feature Interaction.

    Applies a single Transformer encoder layer to the highest-level
    feature map (S5). The Q, K, V projections support LoRA injection.

    Architecture:
        Self-Attention (Multi-Head) -> FFN (2-layer MLP)
        with residual connections and pre-norm (LayerNorm).
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 8, use_lora: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        LinearCls = LoRALinear if use_lora else FrozenLinear

        self.q_proj = LinearCls(hidden_dim, hidden_dim)
        self.k_proj = LinearCls(hidden_dim, hidden_dim)
        self.v_proj = LinearCls(hidden_dim, hidden_dim)
        self.out_proj = FrozenLinear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            FrozenLinear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            FrozenLinear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        lora: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        prefix: str = "",
    ) -> torch.Tensor:
        B, N, C = x.shape

        def _l(suffix):
            if lora is None:
                return None, None
            return lora.get(f"{prefix}{suffix}", (None, None))

        residual = x
        x_norm = self.norm1(x)

        u_q, v_q = _l("q_proj")
        q = self.q_proj(x_norm, u_q, v_q).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        u_k, v_k = _l("k_proj")
        k = self.k_proj(x_norm, u_k, v_k).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        u_v, v_v = _l("v_proj")
        v_val = self.v_proj(x_norm, u_v, v_v).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v_val).transpose(1, 2).contiguous().view(B, N, C)
        attn_out = self.out_proj(attn_out)
        x = residual + attn_out

        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# CCFF: Fusion Blocks with FiLM
# ---------------------------------------------------------------------------


class RepBlock(nn.Module):
    """Convolution block: Conv-BN-SiLU used in CCFF fusion nodes."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = FrozenConv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = FrozenBatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.bn(self.conv(x)))


class CCFFFusionBlock(nn.Module):
    """
    One CCFF fusion block. Takes two scale features and produces
    fused output. FiLM modulation is applied after the fusion.
    """

    def __init__(self, in_ch: int, out_ch: int, direction: str = "top_down"):
        super().__init__()
        self.direction = direction
        self.adjust = FrozenConv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.fuse = RepBlock(out_ch * 2, out_ch)

    def forward(
        self,
        current: torch.Tensor,
        neighbor: torch.Tensor,
        film_gamma: Optional[torch.Tensor] = None,
        film_beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current = self.adjust(current)

        if self.direction == "top_down":
            neighbor = F.interpolate(neighbor, size=current.shape[-2:], mode="bilinear", align_corners=False)
        else:
            neighbor = F.interpolate(neighbor, scale_factor=0.5, mode="bilinear", align_corners=False)

        fused = self.fuse(torch.cat([current, neighbor], dim=1))

        if film_gamma is not None and film_beta is not None:
            fused = film_gamma.view(1, -1, 1, 1) * fused + film_beta.view(1, -1, 1, 1)

        return fused


class CCFF(nn.Module):
    """
    CNN-based Cross-scale Feature Fusion.

    Takes 3-scale features (S3, S4, S5) and applies alternating
    top-down and bottom-up fusion with FiLM modulation.
    """

    def __init__(self, in_channels: list, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj_s3 = FrozenConv2d(in_channels[0], hidden_dim, kernel_size=1)
        self.input_proj_s4 = FrozenConv2d(in_channels[1], hidden_dim, kernel_size=1)
        self.input_proj_s5 = FrozenConv2d(in_channels[2], hidden_dim, kernel_size=1)

        self.fuse_s5_to_s4 = CCFFFusionBlock(hidden_dim, hidden_dim, "top_down")
        self.fuse_s4_to_s3 = CCFFFusionBlock(hidden_dim, hidden_dim, "top_down")
        self.fuse_s3_to_s4 = CCFFFusionBlock(hidden_dim, hidden_dim, "bottom_up")
        self.fuse_s4_to_s5 = CCFFFusionBlock(hidden_dim, hidden_dim, "bottom_up")

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        film_params: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        s3 = self.input_proj_s3(features["S3"])
        s4 = self.input_proj_s4(features["S4"])
        s5 = self.input_proj_s5(features["S5"])
        return self.forward_projected({"S3": s3, "S4": s4, "S5": s5}, film_params)

    def forward_projected(
        self,
        projected: Dict[str, torch.Tensor],
        film_params: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply fusion on already-projected features (all hidden_dim channels).
        Used when projection is done externally (e.g., after AIFI).
        """
        s3, s4, s5 = projected["S3"], projected["S4"], projected["S5"]

        def _film(name):
            if film_params is None:
                return None, None
            return film_params.get(name, (None, None))

        g, b = _film("fuse_s5_to_s4")
        s4_f = self.fuse_s5_to_s4(s4, s5, g, b)

        g, b = _film("fuse_s4_to_s3")
        s3_f = self.fuse_s4_to_s3(s3, s4_f, g, b)

        g, b = _film("fuse_s3_to_s4")
        s4_o = self.fuse_s3_to_s4(s4_f, s3_f, g, b)

        g, b = _film("fuse_s4_to_s5")
        s5_o = self.fuse_s4_to_s5(s5, s4_o, g, b)

        return {"S3": s3_f, "S4": s4_o, "S5": s5_o}


# ---------------------------------------------------------------------------
# Full Encoder
# ---------------------------------------------------------------------------


class HybridEncoder(nn.Module):
    """
    RT-DETR Efficient Hybrid Encoder.

    Pipeline:
        1. Project multi-scale backbone features to hidden_dim
        2. AIFI on S5 features (flatten -> attention -> unflatten)
        3. CCFF fusion on all three scales (with FiLM modulation)
    """

    def __init__(
        self,
        in_channels: list,
        hidden_dim: int = 256,
        num_heads: int = 8,
        use_lora_aifi: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aifi = AIFI(hidden_dim, num_heads, use_lora=use_lora_aifi)
        self.ccff = CCFF(in_channels, hidden_dim)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        lora: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        film_params: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        # Step 1: project all scales to hidden_dim
        s3 = self.ccff.input_proj_s3(features["S3"])
        s4 = self.ccff.input_proj_s4(features["S4"])
        s5 = self.ccff.input_proj_s5(features["S5"])

        # Step 2: AIFI on S5
        B, C, H, W = s5.shape
        s5_flat = s5.flatten(2).transpose(1, 2)
        s5_flat = self.aifi(s5_flat, lora, prefix="encoder.aifi.")
        s5 = s5_flat.transpose(1, 2).view(B, C, H, W)

        # Step 3: CCFF fusion (on already-projected features)
        projected = {"S3": s3, "S4": s4, "S5": s5}
        return self.ccff.forward_projected(projected, film_params)
