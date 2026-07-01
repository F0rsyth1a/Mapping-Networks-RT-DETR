"""
Hungarian Matching for RT-DETR.

Computes the optimal bipartite matching between predicted objects
and ground truth objects using the Hungarian algorithm (scipy), based on
classification cost, L1 bounding box cost, and GIoU cost.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute generalized IoU between two sets of boxes.

    Args:
        boxes1: (N, 4) [x1, y1, x2, y2]
        boxes2: (M, 4) [x1, y1, x2, y2]

    Returns:
        giou: (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-7)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]

    giou = iou - (area_c - union) / (area_c + 1e-7)
    return giou


def hungarian_match(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_labels: List[torch.Tensor],
    gt_boxes: List[torch.Tensor],
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compute Hungarian matching between predictions and ground truth
    using scipy.optimize.linear_sum_assignment.

    Args:
        pred_logits: (B, N, num_classes) predicted class logits
        pred_boxes:  (B, N, 4) predicted boxes [cx, cy, w, h] in [0,1]
        gt_labels:   list of Tensors, each (n_i,) class labels
        gt_boxes:    list of Tensors, each (n_i, 4) ground truth boxes [cx, cy, w, h] in [0,1]

    Returns:
        matched_indices: list of (pred_idx, gt_idx) tensors per image
    """
    B = pred_logits.shape[0]
    matched_indices = []

    for b in range(B):
        n_pred = pred_logits.shape[1]
        n_gt = gt_labels[b].shape[0]

        if n_gt == 0:
            matched_indices.append((
                torch.empty(0, dtype=torch.long, device=pred_logits.device),
                torch.empty(0, dtype=torch.long, device=pred_logits.device),
            ))
            continue

        cls_prob = F.softmax(pred_logits[b], dim=-1)
        cost_cls = -cls_prob[:, gt_labels[b]]
        cost_cls = torch.nan_to_num(cost_cls, nan=0.0, posinf=1e6, neginf=-1e6)

        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])
        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes[b])

        cost_l1 = torch.cdist(pred_boxes[b], gt_boxes[b], p=1)
        cost_l1 = torch.nan_to_num(cost_l1, nan=0.0, posinf=1e6, neginf=0.0)

        cost_giou_val = -generalized_box_iou(pred_xyxy, gt_xyxy)
        cost_giou_val = torch.nan_to_num(cost_giou_val, nan=0.0, posinf=1e6, neginf=-1e6)

        C = (cost_class * cost_cls + cost_bbox * cost_l1 + cost_giou * cost_giou_val)
        C = C.detach().cpu().numpy()

        pred_idx, gt_idx = linear_sum_assignment(C)
        matched_indices.append((
            torch.as_tensor(pred_idx, dtype=torch.long, device=pred_logits.device),
            torch.as_tensor(gt_idx, dtype=torch.long, device=pred_logits.device),
        ))

    return matched_indices
