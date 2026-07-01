"""
RT-DETR with Mapping Network Injection.

Assembles the full target network f(x; z) where:
- x is the input image
- z = [z_lora, z_film, z_query] is the orthogonally-sliced latent vector

All pretrained weights are frozen. Adaptation comes from:
    LoRA injection:  W_adaptive = W_pretrained + U(z_lora) @ V(z_lora)^T
    FiLM modulation: F' = gamma(z_film) * F + beta(z_film)
    Query residual:  Q_final = LayerNorm(Q_init + delta_Q(z_query))
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .backbone import build_backbone
from .encoder import HybridEncoder
from .decoder import RTDETRDecoder


class RTDETRWithMapping(nn.Module):
    """
    Complete RT-DETR model controlled by mapping network outputs.

    Forward pass signature:
        f(x, lora_params, film_params, delta_Q) -> predictions

    Three injection sites:
        1. Backbone (LoRA on layer3/layer4 conv)
        2. Encoder (LoRA on AIFI QKV + FiLM on CCFF fusion)
        3. Decoder (LoRA on cross-attn QKV + ResMod on queries)
    """

    def __init__(
        self,
        num_classes: int = 80,
        hidden_dim: int = 256,
        num_queries: int = 300,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        use_lora_backbone_stage4: bool = True,
        use_lora_backbone_stage5: bool = True,
        use_lora_aifi: bool = True,
        use_lora_cross_attn: bool = True,
        backbone_pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        self.backbone = build_backbone(
            lora_layer3=use_lora_backbone_stage4,
            lora_layer4=use_lora_backbone_stage5,
            pretrained=backbone_pretrained,
        )

        in_channels = [
            self.backbone.out_channels["S3"],  # 512
            self.backbone.out_channels["S4"],  # 1024
            self.backbone.out_channels["S5"],  # 2048
        ]

        self.encoder = HybridEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            use_lora_aifi=use_lora_aifi,
        )

        self.decoder = RTDETRDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            use_lora_cross_attn=use_lora_cross_attn,
        )

    def forward(
        self,
        x: torch.Tensor,
        lora_backbone: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        lora_encoder: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        lora_decoder: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        film_params: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        delta_Q: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with mapping network injection.

        Args:
            x:             (B, 3, H, W) input images
            lora_backbone: LoRA (U,V) for backbone conv layers
            lora_encoder:  LoRA (U,V) for encoder AIFI QKV
            lora_decoder:  LoRA (U,V) for decoder cross-attention QKV
            film_params:   FiLM (gamma, beta) for CCFF fusion nodes
            delta_Q:       (num_queries, hidden_dim) query residual

        Returns:
            dict with keys: pred_logits, pred_boxes
        """
        backbone_features = self.backbone(x, lora_backbone)

        encoder_features = self.encoder(
            backbone_features,
            lora=lora_encoder,
            film_params=film_params,
        )

        outputs = self.decoder(
            encoder_features,
            delta_Q=delta_Q if delta_Q is not None else torch.zeros(
                self.num_queries, self.hidden_dim, device=x.device
            ),
            lora=lora_decoder,
        )

        return outputs


def extract_layer_configs(model: RTDETRWithMapping, lora_rank: int = 16) -> dict:
    """
    Extract layer configurations for LoRA/FiLM generator setup.

    Returns a dict with keys:
        'lora_configs':    list of {'name', 'm', 'n'} for LoRA generator
        'film_node_configs': list of {'name', 'C'} for FiLM generator
    """
    lora_cfgs = []

    # Backbone LoRA: only for layers that actually use LoRAConv2d
    from .backbone import LoRAConv2d
    for stage_name, stage in [("backbone.layer3", model.backbone.layer3),
                               ("backbone.layer4", model.backbone.layer4)]:
        for i, block in enumerate(stage):
            for j, conv in enumerate([block.conv1, block.conv2, block.conv3]):
                if not isinstance(conv, LoRAConv2d):
                    continue
                name = f"{stage_name}.{i}.conv{j+1}"
                m = conv.out_channels
                n = conv.in_channels * conv.kernel_size[0] * conv.kernel_size[1]
                lora_cfgs.append({"name": name, "m": m, "n": n})

    # Encoder AIFI LoRA: only if AIFI uses LoRA
    if hasattr(model.encoder.aifi.q_proj, 'weight_frozen'):
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            lora_cfgs.append({
                "name": f"encoder.aifi.{proj_name}",
                "m": model.hidden_dim, "n": model.hidden_dim,
            })

    # Decoder cross-attention LoRA: only if decoder uses LoRA
    first_layer = model.decoder.layers[0]
    if hasattr(first_layer.cross_attn_q_proj, 'weight_frozen'):
        for i in range(len(model.decoder.layers)):
            for proj_name in ["cross_attn_q_proj", "cross_attn_k_proj", "cross_attn_v_proj"]:
                lora_cfgs.append({
                    "name": f"decoder.layers.{i}.{proj_name}",
                    "m": model.hidden_dim, "n": model.hidden_dim,
                })

    # FiLM nodes for CCFF fusion blocks
    film_cfgs = [
        {"name": "fuse_s5_to_s4", "C": model.hidden_dim},
        {"name": "fuse_s4_to_s3", "C": model.hidden_dim},
        {"name": "fuse_s3_to_s4", "C": model.hidden_dim},
        {"name": "fuse_s4_to_s5", "C": model.hidden_dim},
    ]

    return {"lora_configs": lora_cfgs, "film_node_configs": film_cfgs}
