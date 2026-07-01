"""
RT-DETR Decoder with Residual Modulated Query Selection.

Implements uncertainty-minimal query selection from encoder features,
with manifold-guided residual modulation:

    Q_final = LayerNorm( Q_init + sigma(W_q_orth @ z_query + b_q) )

Decoder layers use self-attention (no LoRA) and cross-attention (LoRA on QKV).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple

from .encoder import FrozenLinear, LoRALinear
from .backbone import FrozenConv2d, FrozenBatchNorm2d


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    """
    One RT-DETR decoder layer.

    Architecture (pre-norm):
        Self-Attention -> Cross-Attention -> FFN
        All with residual connections.
        Cross-attention QKV supports LoRA injection.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        use_lora_cross_attn: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Self-attention
        self.self_attn_qkv = FrozenLinear(hidden_dim, hidden_dim * 3)
        self.self_attn_out = FrozenLinear(hidden_dim, hidden_dim)
        self.norm_sa = nn.LayerNorm(hidden_dim)

        # Cross-attention (LoRA on QKV)
        LinearCls = LoRALinear if use_lora_cross_attn else FrozenLinear
        self.cross_attn_q_proj = LinearCls(hidden_dim, hidden_dim)
        self.cross_attn_k_proj = LinearCls(hidden_dim, hidden_dim)
        self.cross_attn_v_proj = LinearCls(hidden_dim, hidden_dim)
        self.cross_attn_out = FrozenLinear(hidden_dim, hidden_dim)
        self.norm_ca = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            FrozenLinear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            FrozenLinear(hidden_dim * 4, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        lora: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        prefix: str = "",
    ) -> torch.Tensor:
        """
        Args:
            query:  (B, N_q, hidden_dim) object queries
            memory: (B, H*W, hidden_dim) flattened encoder features
            mask:   (B, H*W) optional mask for cross-attention
            lora:   LoRA params for cross-attention QKV
            prefix: layer name prefix for lora key lookup

        Returns:
            query: (B, N_q, hidden_dim) updated queries
        """
        B, N_q, C = query.shape

        def _l(suffix):
            if lora is None:
                return None, None
            return lora.get(f"{prefix}{suffix}", (None, None))

        # --- Self-Attention ---
        residual = query
        q_norm = self.norm_sa(query)
        qkv = self.self_attn_qkv(q_norm).view(B, N_q, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        sa_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N_q, C)
        sa_out = self.self_attn_out(sa_out)
        query = residual + sa_out

        # --- Cross-Attention ---
        residual = query
        q_norm = self.norm_ca(query)

        u_q, v_q = _l("cross_attn_q_proj")
        q = self.cross_attn_q_proj(q_norm, u_q, v_q).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)

        u_k, v_k = _l("cross_attn_k_proj")
        k = self.cross_attn_k_proj(memory, u_k, v_k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        u_v, v_v = _l("cross_attn_v_proj")
        v_val = self.cross_attn_v_proj(memory, u_v, v_v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        ca_out = torch.matmul(attn, v_val).transpose(1, 2).contiguous().view(B, N_q, C)
        ca_out = self.cross_attn_out(ca_out)
        query = residual + ca_out

        # --- FFN ---
        residual = query
        query = residual + self.ffn(self.norm_ffn(query))

        return query


# ---------------------------------------------------------------------------
# Query Selection: Uncertainty-Minimal
# ---------------------------------------------------------------------------


class QuerySelector(nn.Module):
    """
    Uncertainty-minimal query selection.

    Selects top-K encoder features that have:
        - High classification confidence (confident about class)
        - High localization uncertainty (less certain about bbox)

    Combined metric: cls_score - alpha * bbox_uncertainty

    The selected features form Q_init, which is then modulated by
    the ResModGenerator's delta_Q.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_classes: int = 80,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        self.cls_proj = FrozenLinear(hidden_dim, num_classes, init_gain=0.1)
        self.bbox_proj = nn.Sequential(
            FrozenLinear(hidden_dim, hidden_dim, init_gain=0.1),
            nn.ReLU(),
            FrozenLinear(hidden_dim, hidden_dim, init_gain=0.1),
            nn.ReLU(),
            FrozenLinear(hidden_dim, 4, init_gain=0.1),
        )

    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_features: (B, C, H, W) S5 encoder output

        Returns:
            Q_init: (B, num_queries, hidden_dim) selected query features
        """
        B, C, H, W = encoder_features.shape
        N = H * W
        x_flat = encoder_features.flatten(2).transpose(1, 2)  # (B, N, C)

        cls_logits = self.cls_proj(x_flat)      # (B, N, num_classes)
        bbox_pred = self.bbox_proj(x_flat)       # (B, N, 4)

        cls_score = cls_logits.max(dim=-1).values  # (B, N)

        bbox_uncertainty = bbox_pred.abs().sum(dim=-1)  # (B, N)

        uncertainty_score = cls_score - 0.3 * bbox_uncertainty

        _, topk_idx = torch.topk(uncertainty_score, self.num_queries, dim=1)

        batch_idx = torch.arange(B, device=encoder_features.device).unsqueeze(1).expand(-1, self.num_queries)
        Q_init = x_flat[batch_idx, topk_idx]  # (B, num_queries, hidden_dim)

        return Q_init


# ---------------------------------------------------------------------------
# Full Decoder
# ---------------------------------------------------------------------------


class RTDETRDecoder(nn.Module):
    """
    RT-DETR Decoder with residual modulated query selection.

    1. Uncertainty-minimal query selection from S5 encoder features
    2. Residual modulation: Q = LayerNorm(Q_init + delta_Q)
    3. 6 decoder layers with self/cross attention
    4. Classification + BBox prediction heads
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_classes: int = 80,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        use_lora_cross_attn: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        self.query_selector = QuerySelector(hidden_dim, num_queries, num_classes)
        self.query_norm = nn.LayerNorm(hidden_dim)

        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim, num_heads, use_lora_cross_attn)
            for _ in range(num_decoder_layers)
        ])

        self.cls_head = nn.Sequential(
            FrozenLinear(hidden_dim, hidden_dim, init_gain=0.1),
            nn.ReLU(),
            FrozenLinear(hidden_dim, num_classes, init_gain=0.1),
        )
        self.bbox_head = nn.Sequential(
            FrozenLinear(hidden_dim, hidden_dim, init_gain=0.1),
            nn.ReLU(),
            FrozenLinear(hidden_dim, hidden_dim, init_gain=0.1),
            nn.ReLU(),
            FrozenLinear(hidden_dim, 4, init_gain=0.1),
        )

    def forward(
        self,
        encoder_features: Dict[str, torch.Tensor],
        delta_Q: torch.Tensor,
        lora: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_features: {'S3', 'S4', 'S5'} from HybridEncoder
            delta_Q: (num_queries, hidden_dim) from ResModGenerator
            lora:    LoRA params for cross-attention QKV

        Returns:
            dict with 'pred_logits' and 'pred_boxes'
        """
        s5 = encoder_features["S5"]

        Q_init = self.query_selector(s5)  # (B, num_queries, hidden_dim)

        delta_Q_expanded = delta_Q.unsqueeze(0).expand(Q_init.shape[0], -1, -1)
        query = self.query_norm(Q_init + delta_Q_expanded)

        # Multi-scale memory: concatenate S3, S4, S5 flattened features
        memory_parts = []
        for key in ["S3", "S4", "S5"]:
            feat = encoder_features[key]
            B, C, H, W = feat.shape
            memory_parts.append(feat.flatten(2).transpose(1, 2))
        memory = torch.cat(memory_parts, dim=1)  # (B, sum(H_i*W_i), hidden_dim)

        for i, layer in enumerate(self.layers):
            query = layer(query, memory, lora=lora, prefix=f"decoder.layers.{i}.")

        pred_logits = self.cls_head(query)
        pred_boxes = self.bbox_head(query).sigmoid()

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
