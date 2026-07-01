"""
Entry point for Mapping Networks + RT-DETR training.

Single-GPU:
    python main.py --config configs/default.yaml --coco-path /data/coco

Multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=4 main.py --config configs/default.yaml --coco-path /data/coco

Resume from checkpoint:
    python main.py --config configs/default.yaml --coco-path /data/coco --resume output/ckpt_stage2_epoch020.pt

Dry-run for architecture verification:
    python main.py --config configs/default.yaml --dry-run
"""

import argparse
import os
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CocoDetection
from torchvision import transforms as T

from src.training import MappingTrainer


def get_coco_dataset(root: str, image_set: str = "train2017"):
    """
    Load COCO detection dataset with proper transforms.

    Args:
        root:      COCO root directory (contains annotations/ and train2017/)
        image_set: 'train2017' or 'val2017'

    Returns:
        CocoDetection dataset with images as tensors and targets as dicts.
    """
    img_dir = os.path.join(root, image_set)
    ann_file = os.path.join(root, "annotations", f"instances_{image_set}.json")

    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return CocoDetectionExt(root=img_dir, annFile=ann_file, transform=transform)


class CocoDetectionExt(CocoDetection):
    """
    Extended CocoDetection that returns target dicts with 'labels' and 'boxes'.
    Boxes are in [cx, cy, w, h] format, normalized to [0, 1].
    """

    def __getitem__(self, idx):
        img, annotations = super().__getitem__(idx)

        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img_w, img_h = img_info["width"], img_info["height"]

        labels = []
        boxes = []
        for ann in annotations:
            if "bbox" not in ann or "category_id" not in ann:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue

            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            w_n = w / img_w
            h_n = h / img_h

            labels.append(ann["category_id"] - 1)
            boxes.append([cx, cy, w_n, h_n])

        if len(labels) == 0:
            labels = torch.zeros(0, dtype=torch.long)
            boxes = torch.zeros(0, 4)
        else:
            labels = torch.tensor(labels, dtype=torch.long)
            boxes = torch.tensor(boxes, dtype=torch.float32)

        return img, {"labels": labels, "boxes": boxes}


class DummyDetectionDataset(Dataset):
    """Synthetic dataset for architecture verification."""

    def __init__(self, num_samples: int = 100, image_size: int = 640, num_classes: int = 80):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.image_size, self.image_size)
        num_objects = torch.randint(1, 5, (1,)).item()
        labels = torch.randint(0, self.num_classes, (num_objects,))
        boxes = torch.rand(num_objects, 4)
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:] * 0.3
        boxes = boxes.clamp(0, 1)
        return image, {"labels": labels, "boxes": boxes}


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def setup_ddp():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Mapping Networks + RT-DETR")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--coco-path", type=str, default=None, help="COCO dataset root directory")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_ddp()
    is_main = (rank == 0)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size

    if args.dry_run:
        if is_main:
            print("[DRY-RUN] Using synthetic data for architecture verification.")
        cfg["training"]["stage1"]["epochs"] = 1
        cfg["training"]["stage2"]["epochs"] = 1
        cfg["training"]["stage3"]["epochs"] = 1

    if is_main:
        print(f"[INIT] Config: {args.config}")
        print(f"[INIT] Device: {local_rank}, World size: {world_size}")
        print(f"[INIT] Output: {args.output_dir}")

    trainer = MappingTrainer(args.config, output_dir=args.output_dir)
    trainer.build()

    if args.dry_run:
        dataset = DummyDetectionDataset(num_samples=16, image_size=640)
    elif args.coco_path and os.path.isdir(args.coco_path):
        if is_main:
            print(f"[DATA] Loading COCO from {args.coco_path}")
        dataset = get_coco_dataset(args.coco_path, "train2017")
    else:
        if is_main:
            print("[DATA] No COCO path provided. Using synthetic data.")
        dataset = DummyDetectionDataset(num_samples=32, image_size=640)

    sampler = DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if is_main:
        print(f"[TRAIN] Starting three-stage training on {len(dataset)} samples, "
              f"{len(dataloader)} batches/epoch")

    trainer.train(dataloader, resume_from=args.resume)

    if is_main:
        print("[DONE] Training complete.")

    cleanup_ddp()


if __name__ == "__main__":
    main()
