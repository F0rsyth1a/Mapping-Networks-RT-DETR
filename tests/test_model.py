"""
Test RT-DETR model forward pass with mapping injection.
"""

import torch
import sys
sys.path.insert(0, ".")

from src.models import RTDETRWithMapping
from src.models.rt_detr import extract_layer_configs
from src.generators import LoRAGenerator, FiLMGenerator, ResModGenerator
from src.latent import OrthogonalLatentSpace


def test_model_forward_no_injection():
    """Model should work without any mapping injection (identity behavior)."""
    model = RTDETRWithMapping(
        num_classes=80,
        hidden_dim=256,
        num_queries=10,  # small for fast test
        num_decoder_layers=2,
        num_heads=4,
        use_lora_backbone_stage4=False,
        use_lora_backbone_stage5=False,
        use_lora_aifi=False,
        use_lora_cross_attn=False,
        backbone_pretrained=False,
    )
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert "pred_logits" in out
    assert "pred_boxes" in out
    assert out["pred_logits"].shape == (1, 10, 80)
    assert out["pred_boxes"].shape == (1, 10, 4)


def test_model_forward_with_lora_backbone():
    """Model forward with LoRA injection on backbone only."""
    model = RTDETRWithMapping(
        num_classes=80, hidden_dim=256, num_queries=10,
        num_decoder_layers=2, num_heads=4,
        use_lora_backbone_stage4=True,
        use_lora_aifi=False,
        use_lora_cross_attn=False,
        backbone_pretrained=False,
    )
    model.eval()

    info = extract_layer_configs(model, lora_rank=4)

    d_lora = 64
    lora_gen = LoRAGenerator(d_lora, info["lora_configs"], rank=4)
    z_lora = torch.randn(d_lora)

    lora_all = lora_gen(z_lora)
    lora_bb = {k: v for k, v in lora_all.items() if k.startswith("backbone.")}

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x, lora_backbone=lora_bb)

    assert out["pred_logits"].shape == (1, 10, 80)
    assert out["pred_boxes"].shape == (1, 10, 4)


def test_full_pipeline_forward():
    """End-to-end forward pass with all injection points active."""
    model = RTDETRWithMapping(
        num_classes=80, hidden_dim=256, num_queries=10,
        num_decoder_layers=2, num_heads=4,
        use_lora_backbone_stage4=True,
        use_lora_aifi=True,
        use_lora_cross_attn=True,
        backbone_pretrained=False,
    )
    model.eval()

    info = extract_layer_configs(model, lora_rank=4)

    latent = OrthogonalLatentSpace(d_lora=64, d_film=32, d_query=32)

    lora_gen = LoRAGenerator(64, info["lora_configs"], rank=4)
    film_gen = FiLMGenerator(32, info["film_node_configs"])
    resmod_gen = ResModGenerator(32, 10, 256)

    z_lora = latent.get_z_lora()
    z_film = latent.get_z_film()
    z_query = latent.get_z_query()

    lora_all = lora_gen(z_lora)
    film_all = film_gen(z_film)
    delta_q = resmod_gen(z_query)

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(
            x,
            lora_backbone={k: v for k, v in lora_all.items() if k.startswith("backbone.")},
            lora_encoder={k: v for k, v in lora_all.items() if k.startswith("encoder.")},
            lora_decoder={k: v for k, v in lora_all.items() if k.startswith("decoder.")},
            film_params=film_all,
            delta_Q=delta_q,
        )

    assert out["pred_logits"].shape == (1, 10, 80)
    assert out["pred_boxes"].shape == (1, 10, 4)


def test_backbone_output_shapes():
    """Verify backbone outputs correct feature shapes."""
    from src.models.backbone import build_backbone

    bb = build_backbone(lora_layer3=False, lora_layer4=False)
    bb.eval()

    x = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        feats = bb(x)

    B = 2
    assert feats["S3"].shape == (B, 512, 80, 80), f"Got {feats['S3'].shape}"   # stride 8
    assert feats["S4"].shape == (B, 1024, 40, 40), f"Got {feats['S4'].shape}"  # stride 16
    assert feats["S5"].shape == (B, 2048, 20, 20), f"Got {feats['S5'].shape}"  # stride 32


if __name__ == "__main__":
    test_model_forward_no_injection()
    print("  [PASS] forward no injection")

    test_model_forward_with_lora_backbone()
    print("  [PASS] forward with LoRA backbone")

    test_full_pipeline_forward()
    print("  [PASS] full pipeline forward")

    test_backbone_output_shapes()
    print("  [PASS] backbone output shapes")

    print("All model tests passed.")
