"""
Task Loss for RT-DETR: classification cross-entropy + bounding box regression.

Uses Hungarian matching to assign predictions to ground truth,
then computes:
    L_cls:  cross-entropy loss on matched pairs
    L_l1:   L1 loss on bounding box coordinates
    L_giou: generalized IoU loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from ..models.matching import (
    hungarian_match,
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)


class TaskLoss(nn.Module):
    """
    RT-DETR task loss = L_cls + L_l1 + L_giou.

    Uses Hungarian matching internally to align predictions with
    ground truth annotations.
    """

    def __init__(
        self,
        num_classes: int = 80,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        loss_ce_weight: float = 1.0,
        loss_bbox_weight: float = 5.0,
        loss_giou_weight: float = 2.0,
        empty_weight: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.loss_ce_weight = loss_ce_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_giou_weight = loss_giou_weight
        self.empty_weight = empty_weight

    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_labels: List[torch.Tensor],
        gt_boxes: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred_logits: (B, N, num_classes)
            pred_boxes:  (B, N, 4) [cx, cy, w, h] normalized to [0,1]
            gt_labels:   list of (n_i,) class labels per image
            gt_boxes:    list of (n_i, 4) boxes per image [cx, cy, w, h] normalized

        Returns:
            total_loss: scalar
            loss_dict:  {'loss_ce', 'loss_l1', 'loss_giou'}
        """
        B, N = pred_logits.shape[:2]

        matched = hungarian_match(
            pred_logits, pred_boxes, gt_labels, gt_boxes,
            self.cost_class, self.cost_bbox, self.cost_giou,
        )

        # Build target class labels
        device = pred_logits.device
        tgt_labels = torch.full((B, N), self.num_classes, dtype=torch.long, device=device)
        tgt_boxes = torch.zeros(B, N, 4, device=device)
        valid_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        for b, (pred_idx, gt_idx) in enumerate(matched):
            if pred_idx.numel() == 0:
                continue
            tgt_labels[b, pred_idx] = gt_labels[b][gt_idx]
            tgt_boxes[b, pred_idx] = gt_boxes[b][gt_idx]
            valid_mask[b, pred_idx] = True

        # Classification loss: ignore unmatched (label == num_classes)
        loss_ce = F.cross_entropy(
            pred_logits.view(-1, self.num_classes),
            tgt_labels.view(-1),
            ignore_index=self.num_classes,
            reduction="mean",
        )

        # Bounding box losses (only on matched pairs)
        if valid_mask.any():
            pred_boxes_matched = pred_boxes[valid_mask]
            tgt_boxes_matched = tgt_boxes[valid_mask]

            loss_l1 = F.l1_loss(pred_boxes_matched, tgt_boxes_matched, reduction="mean")

            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes_matched)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes_matched)
            giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
            loss_giou = (1.0 - giou.diag()).mean()
        else:
            loss_l1 = torch.tensor(0.0, device=device)
            loss_giou = torch.tensor(0.0, device=device)

        total = (
            self.loss_ce_weight * loss_ce
            + self.loss_bbox_weight * loss_l1
            + self.loss_giou_weight * loss_giou
        )

        info = {"loss_ce": loss_ce.item(), "loss_l1": loss_l1.item(), "loss_giou": loss_giou.item()}
        return total, info
