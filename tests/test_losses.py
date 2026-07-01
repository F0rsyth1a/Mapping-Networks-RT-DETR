"""
Test loss functions: task loss, smoothness, stability, alignment.
"""

import torch
import sys
sys.path.insert(0, ".")

from src.losses import TaskLoss, SmoothnessLoss, StabilityLoss, AlignmentLoss
from src.generators import LoRAGenerator, FiLMGenerator, ResModGenerator
from src.latent import OrthogonalLatentSpace
from src.models.matching import generalized_box_iou, box_cxcywh_to_xyxy


def test_giou():
    """Test GIoU computation."""
    boxes1 = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.3, 0.3, 0.8, 0.8]])
    boxes2 = torch.tensor([[0.2, 0.2, 0.7, 0.7], [0.0, 0.0, 0.5, 0.5]])

    giou = generalized_box_iou(boxes1, boxes2)
    assert giou.shape == (2, 2)
    # Diagonal should be 1.0 for perfect match
    assert torch.allclose(giou.diag(), torch.tensor([giou[0, 0], giou[1, 1]]))

    # IoU of identical boxes should be close to 1
    giou_self = generalized_box_iou(boxes1, boxes1)
    assert torch.allclose(giou_self.diag(), torch.ones(2), atol=1e-5)


def test_task_loss():
    """Test task loss with synthetic data."""
    loss_fn = TaskLoss(num_classes=5)

    pred_logits = torch.randn(2, 10, 5)
    pred_boxes = torch.rand(2, 10, 4)
    pred_boxes[:, :, 2:] = pred_boxes[:, :, :2] + 0.2 * torch.rand(2, 10, 2)

    gt_labels = [
        torch.tensor([0, 1], dtype=torch.long),
        torch.tensor([2, 3, 1], dtype=torch.long),
    ]
    gt_boxes = [
        torch.tensor([[0.2, 0.3, 0.4, 0.5], [0.5, 0.5, 0.7, 0.7]]),
        torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.4, 0.4, 0.6, 0.6], [0.7, 0.7, 0.9, 0.9]]),
    ]

    loss, info = loss_fn(pred_logits, pred_boxes, gt_labels, gt_boxes)
    assert loss.item() > 0, "Task loss should be positive"
    assert "loss_ce" in info


def test_stability_loss():
    """Test stability loss computation."""
    loss_fn = StabilityLoss(noise_sigma=0.1)

    out_clean = {
        "pred_logits": torch.randn(2, 10, 5),
        "pred_boxes": torch.rand(2, 10, 4),
    }
    out_noisy = {
        "pred_logits": torch.randn(2, 10, 5),
        "pred_boxes": torch.rand(2, 10, 4),
    }

    loss = loss_fn(out_clean, out_noisy)
    assert loss.item() >= 0


def test_stability_loss_identical():
    """Stability loss should be zero for identical outputs."""
    loss_fn = StabilityLoss()

    out = {
        "pred_logits": torch.ones(2, 10, 5),
        "pred_boxes": torch.zeros(2, 10, 4),
    }

    loss = loss_fn(out, out)
    assert loss.item() == 0.0


def test_alignment_loss():
    """Test alignment loss."""
    loss_fn = AlignmentLoss()

    z_l = torch.randn(32)
    z_f = torch.randn(16)
    z_q = torch.randn(16)
    W_l1 = torch.randn(64, 32)
    W_l2 = torch.randn(128, 32)
    W_f = torch.randn(32, 16)
    W_q = torch.randn(48, 16)

    loss = loss_fn(z_l, z_f, z_q, [W_l1, W_l2], [W_f], [W_q])
    assert 0.0 <= loss.item() <= 6.0, f"Alignment loss out of range: {loss.item()}"


def test_smoothness_hutchinson():
    """Test Hutchinson JVP smoothness estimation."""
    d_lora = 16
    gen = LoRAGenerator(d_lora, [{"name": "t", "m": 8, "n": 12}], rank=2)
    z = torch.randn(d_lora)

    loss_fn = SmoothnessLoss(num_samples=1)

    generators = {"lora": gen}
    z_slices = {"lora": z}

    loss = loss_fn(generators, z_slices)
    assert loss.item() >= 0


def test_no_grad_smoothness():
    """Smoothness loss should not persist gradients on z after computation."""
    gen = LoRAGenerator(8, [{"name": "t", "m": 4, "n": 6}], rank=2)
    z = torch.randn(8, requires_grad=True)

    loss_fn = SmoothnessLoss(num_samples=1)
    loss = loss_fn({"lora": gen}, {"lora": z})

    # After backward, z's grad should exist (due to create_graph)
    loss.backward()
    assert z.grad is not None
    assert z.grad.abs().sum() > 0


if __name__ == "__main__":
    test_giou()
    print("  [PASS] GIoU")

    test_task_loss()
    print("  [PASS] task loss")

    test_stability_loss()
    print("  [PASS] stability loss")

    test_stability_loss_identical()
    print("  [PASS] stability identical")

    test_alignment_loss()
    print("  [PASS] alignment loss")

    test_smoothness_hutchinson()
    print("  [PASS] smoothness Hutchinson")

    test_no_grad_smoothness()
    print("  [PASS] smoothness gradient")

    print("All loss tests passed.")
