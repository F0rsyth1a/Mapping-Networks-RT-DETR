"""
Three-stage progressive training pipeline.

Stage 1 (Manifold Discovery):
    - Frozen: all pretrained weights, all generator modulation params
    - Trainable: only z (OrthogonalLatentSpace.z)
    - Loss: L_task only

Stage 2 (Joint Modulation):
    - Frozen: all pretrained weights
    - Trainable: z + all generator gamma, beta, bias
    - Loss: L_task only

Stage 3 (Manifold Solidification):
    - Frozen: all pretrained weights
    - Trainable: z + all generator modulation
    - Loss: L_task + lambda1*L_stab + lambda2*L_smooth + lambda3*L_align
    - AMP bfloat16, gradient checkpointing, LR exponential decay
"""

from __future__ import annotations

import os
import time
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from typing import Dict, Any, List, Optional, Callable
import yaml

from ..latent import OrthogonalLatentSpace
from ..generators import LoRAGenerator, FiLMGenerator, ResModGenerator
from ..models import RTDETRWithMapping
from ..models.rt_detr import extract_layer_configs
from ..losses import TaskLoss, SmoothnessLoss, StabilityLoss, AlignmentLoss
from ..utils import gradient_isolation_check, MemoryTracker

try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast


class MappingTrainer:
    """
    Master trainer orchestrating the three-stage training pipeline.

    Usage:
        trainer = MappingTrainer(config_path, output_dir='./output')
        trainer.build()
        trainer.train(dataloader)
    """

    def __init__(self, config_path: str, output_dir: str = "./output"):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.device = torch.device(self.cfg["hardware"].get("device", "cuda"))
        self.output_dir = output_dir
        self._built = False
        self._global_step = 0
        self._current_stage = None

        os.makedirs(output_dir, exist_ok=True)

    def build(self):
        """Instantiate all components: latent space, generators, model, losses."""
        cfg = self.cfg

        self.latent = OrthogonalLatentSpace(
            d_lora=cfg["latent"]["d_lora"],
            d_film=cfg["latent"]["d_film"],
            d_query=cfg["latent"]["d_query"],
        ).to(self.device)

        self.model = RTDETRWithMapping(
            num_classes=cfg["model"]["num_classes"],
            hidden_dim=cfg["model"]["hidden_dim"],
            num_queries=cfg["model"]["num_queries"],
            num_decoder_layers=cfg["model"]["num_decoder_layers"],
            num_heads=cfg["model"]["num_heads"],
            use_lora_backbone_stage4=cfg["lora"]["inject_backbone_stage4"],
            use_lora_backbone_stage5=cfg["lora"]["inject_backbone_stage5"],
            use_lora_aifi=cfg["lora"]["inject_aifi_qkv"],
            use_lora_cross_attn=cfg["lora"]["inject_decoder_cross_attn"],
            backbone_pretrained=cfg["model"].get("backbone_pretrained", True),
        ).to(self.device)

        layer_info = extract_layer_configs(self.model, cfg["lora"]["rank"])

        self.lora_gen = LoRAGenerator(
            d_lora=cfg["latent"]["d_lora"],
            layer_configs=layer_info["lora_configs"],
            rank=cfg["lora"]["rank"],
        ).to(self.device)

        self.film_gen = FiLMGenerator(
            d_film=cfg["latent"]["d_film"],
            node_configs=layer_info["film_node_configs"],
        ).to(self.device)

        self.resmod_gen = ResModGenerator(
            d_query=cfg["latent"]["d_query"],
            num_queries=cfg["model"]["num_queries"],
            hidden_dim=cfg["model"]["hidden_dim"],
        ).to(self.device)

        self.task_loss = TaskLoss(num_classes=cfg["model"]["num_classes"])
        self.stab_loss = StabilityLoss(noise_sigma=cfg["loss"]["noise_sigma"])
        self.smooth_loss = SmoothnessLoss(num_samples=cfg["loss"]["hutchinson_samples"])
        self.align_loss = AlignmentLoss()
        self.mem_tracker = MemoryTracker(self.device)

        self._built = True
        print(f"[BUILD] Model built. "
              f"LoRA layers={len(layer_info['lora_configs'])}, "
              f"FiLM nodes={len(layer_info['film_node_configs'])}")

    # ------------------------------------------------------------------
    # Stage configuration
    # ------------------------------------------------------------------

    def _configure_stage(self, stage_name: str):
        sc = self.cfg["training"][stage_name]
        train_z = sc.get("train_z", True)
        train_mod = sc.get("train_modulation", False)

        self.latent.z.requires_grad_(train_z)

        for p in self.model.parameters():
            p.requires_grad_(False)

        for gen in [self.lora_gen, self.film_gen, self.resmod_gen]:
            for name, p in gen.named_parameters():
                if "orth" in name:
                    p.requires_grad_(False)
                else:
                    p.requires_grad_(train_mod)

        params = []
        if train_z:
            params.append(self.latent.z)
        if train_mod:
            for gen in [self.lora_gen, self.film_gen, self.resmod_gen]:
                params.extend([p for p in gen.parameters() if p.requires_grad])

        return params

    def _forward_pass(
        self, x: torch.Tensor, _z_override: tuple = None
    ) -> Dict[str, torch.Tensor]:
        if _z_override is not None:
            z_lora, z_film, z_query = _z_override
        else:
            z_lora = self.latent.get_z_lora()
            z_film = self.latent.get_z_film()
            z_query = self.latent.get_z_query()

        lora_all = self.lora_gen(z_lora) if self.lora_gen.layer_names else {}
        film_all = self.film_gen(z_film) if self.film_gen.node_names else {}
        delta_q = self.resmod_gen(z_query)

        return self.model(
            x,
            lora_backbone={k: v for k, v in lora_all.items() if k.startswith("backbone.")},
            lora_encoder={k: v for k, v in lora_all.items() if k.startswith("encoder.")},
            lora_decoder={k: v for k, v in lora_all.items() if k.startswith("decoder.")},
            film_params=film_all,
            delta_Q=delta_q,
        )

    # ------------------------------------------------------------------
    # Checkpoint save / resume
    # ------------------------------------------------------------------

    def _save_checkpoint(self, stage_name: str, epoch: int):
        path = os.path.join(self.output_dir, f"ckpt_{stage_name}_epoch{epoch:03d}.pt")
        torch.save({
            "stage": stage_name,
            "epoch": epoch,
            "global_step": self._global_step,
            "latent_z": self.latent.z.data.clone(),
            "lora_gen": self.lora_gen.state_dict(),
            "film_gen": self.film_gen.state_dict(),
            "resmod_gen": self.resmod_gen.state_dict(),
            "config": self.cfg,
        }, path)
        print(f"[CKPT] Saved to {path}")

    def _load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.latent.z.data.copy_(ckpt["latent_z"])
        self.lora_gen.load_state_dict(ckpt["lora_gen"])
        self.film_gen.load_state_dict(ckpt["film_gen"])
        self.resmod_gen.load_state_dict(ckpt["resmod_gen"])
        self._global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[CKPT] Resumed from {path} (epoch {start_epoch}, step {self._global_step})")
        return start_epoch

    # ------------------------------------------------------------------
    # Stage runner
    # ------------------------------------------------------------------

    def _run_stage(
        self,
        stage_name: str,
        dataloader,
        active_losses: List[str],
        resume_from: Optional[str] = None,
    ):
        cfg = self.cfg["training"][stage_name]
        epochs = cfg["epochs"]
        lr = cfg["lr"]
        gamma = cfg.get("lr_decay_gamma", 1.0)
        use_amp = cfg.get("amp", False)
        use_gc = cfg.get("gradient_checkpointing", False)
        save_every = cfg.get("save_every_epoch", 0)

        params = self._configure_stage(stage_name)
        if not params:
            print(f"[{stage_name}] No trainable parameters. Skipping.")
            return

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        is_stage3 = stage_name == "stage3"
        if is_stage3 and gamma < 1.0:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        else:
            scheduler = None

        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            start_epoch = self._load_checkpoint(resume_from)

        generator_map = {"lora": self.lora_gen, "film": self.film_gen, "query": self.resmod_gen}
        gradient_isolation_check(self.latent, generator_map, verbose=True)

        if is_stage3 and gamma < 1.0:
            lr_sched = f"ExponentialDecay(gamma={gamma})"
        else:
            lr_sched = "Constant"

        print(f"\n{'='*60}")
        print(f"  STAGE: {cfg.get('name', stage_name)}")
        print(f"  Epochs: {epochs}, LR: {lr}, Scheduler: {lr_sched}, Losses: {active_losses}")
        print(f"  AMP: {use_amp}, GC: {use_gc}")
        print(f"  Trainable params: {sum(p.numel() for p in params):,}")
        print(f"  GPU memory: {self.mem_tracker.snapshot()}")
        print(f"{'='*60}\n")

        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for epoch in range(start_epoch, epochs):
            epoch_loss = 0.0
            epoch_info = {}
            t_start = time.time()

            for batch_idx, batch in enumerate(dataloader):
                images, targets = batch
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                gt_labels = [t["labels"] for t in targets]
                gt_boxes = [t["boxes"] for t in targets]

                # --- Forward ---
                if use_gc and use_amp:
                    with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        output = torch_checkpoint(self._forward_pass, images, use_reentrant=False)
                elif use_gc:
                    output = torch_checkpoint(self._forward_pass, images, use_reentrant=False)
                elif use_amp:
                    with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                        output = self._forward_pass(images)
                else:
                    output = self._forward_pass(images)

                output["pred_logits"] = torch.nan_to_num(
                    output["pred_logits"], nan=0.0, posinf=1e4, neginf=-1e4
                )
                output["pred_boxes"] = torch.nan_to_num(
                    output["pred_boxes"], nan=0.5, posinf=1.0, neginf=0.0
                )

                # --- Task Loss ---
                loss_total = torch.tensor(0.0, device=self.device)
                if "task" in active_losses:
                    l_task, info = self.task_loss(
                        output["pred_logits"].float(),
                        output["pred_boxes"].float(),
                        gt_labels, gt_boxes,
                    )
                    loss_total = loss_total + l_task
                    for k, v in info.items():
                        epoch_info[k] = epoch_info.get(k, 0) + v

                # --- Stability Loss ---
                if "stab" in active_losses:
                    sigma = self.cfg["loss"]["noise_sigma"]
                    z_l, z_f, z_q = self.latent.get_all_slices()
                    noisy_slices = (
                        z_l + torch.randn_like(z_l) * sigma,
                        z_f + torch.randn_like(z_f) * sigma,
                        z_q + torch.randn_like(z_q) * sigma,
                    )
                    if use_amp:
                        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                            output_noisy = self._forward_pass(images, _z_override=noisy_slices)
                    else:
                        output_noisy = self._forward_pass(images, _z_override=noisy_slices)

                    output_noisy["pred_logits"] = torch.nan_to_num(
                        output_noisy["pred_logits"], nan=0.0, posinf=1e4, neginf=-1e4
                    )
                    output_noisy["pred_boxes"] = torch.nan_to_num(
                        output_noisy["pred_boxes"], nan=0.5, posinf=1.0, neginf=0.0
                    )

                    l_stab = self.stab_loss(output, output_noisy)
                    loss_total = loss_total + self.cfg["loss"]["lambda_stab"] * l_stab
                    epoch_info["loss_stab"] = epoch_info.get("loss_stab", 0) + l_stab.item()

                # --- Smoothness Loss ---
                if "smooth" in active_losses:
                    generators = {"lora": self.lora_gen, "film": self.film_gen, "query": self.resmod_gen}
                    z_slices = {
                        "lora": self.latent.get_z_lora(),
                        "film": self.latent.get_z_film(),
                        "query": self.latent.get_z_query(),
                    }
                    l_smooth, sm_info = self.compute_smoothness(generators, z_slices)
                    loss_total = loss_total + self.cfg["loss"]["lambda_smooth"] * l_smooth
                    for k, v in sm_info.items():
                        epoch_info[k] = epoch_info.get(k, 0) + v

                # --- Alignment Loss ---
                if "align" in active_losses:
                    orth_l, orth_f, orth_q = self._collect_orth_by_generator()
                    l_align = self.align_loss(
                        self.latent.get_z_lora(),
                        self.latent.get_z_film(),
                        self.latent.get_z_query(),
                        orth_l, orth_f, orth_q,
                    )
                    loss_total = loss_total + self.cfg["loss"]["lambda_align"] * l_align
                    epoch_info["loss_align"] = epoch_info.get("loss_align", 0) + l_align.item()

                optimizer.zero_grad()

                if use_amp:
                    scaler.scale(loss_total).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_total.backward()
                    optimizer.step()

                self._global_step += 1
                epoch_loss += loss_total.item()

                if batch_idx % 10 == 0:
                    info_str = " | ".join(
                        f"{k}={v / (batch_idx + 1):.4f}" for k, v in epoch_info.items()
                    )
                    print(f"  Epoch {epoch+1}/{epochs} Batch {batch_idx} "
                          f"loss={loss_total.item():.4f} | {info_str}")

            avg_loss = epoch_loss / max(len(dataloader), 1)
            elapsed = time.time() - t_start
            lr_now = scheduler.get_last_lr()[0] if scheduler is not None else lr
            print(f"  Epoch {epoch+1} avg_loss={avg_loss:.4f} "
                  f"time={elapsed:.1f}s LR={lr_now:.2e}")

            if scheduler is not None:
                scheduler.step()

            if save_every > 0 and (epoch + 1) % save_every == 0:
                self._save_checkpoint(stage_name, epoch + 1)

        self._save_checkpoint(stage_name, epochs)

    # ------------------------------------------------------------------
    # Smoothness helper
    # ------------------------------------------------------------------

    def compute_smoothness(self, generators, z_slices):
        from ..losses.hutchinson_jvp import compute_smoothness_hutchinson
        return compute_smoothness_hutchinson(
            generators, z_slices, self.cfg["loss"]["hutchinson_samples"]
        )

    def _collect_orth_by_generator(self):
        mats_l, mats_f, mats_q = [], [], []
        for name, param in self.lora_gen.named_parameters():
            if "orth" in name:
                mats_l.append(param)
        for name, param in self.film_gen.named_parameters():
            if "orth" in name:
                mats_f.append(param)
        for name, param in self.resmod_gen.named_parameters():
            if "orth" in name:
                mats_q.append(param)
        return mats_l, mats_f, mats_q

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, dataloader, resume_from: Optional[str] = None):
        if not self._built:
            raise RuntimeError("Call build() before train().")

        stages = ["stage1", "stage2", "stage3"]

        for stage_name in stages:
            sc = self.cfg["training"].get(stage_name)
            if sc is None:
                print(f"[SKIP] Stage '{stage_name}' not configured.")
                continue

            self._run_stage(
                stage_name=stage_name,
                dataloader=dataloader,
                active_losses=sc.get("losses", ["task"]),
                resume_from=resume_from if stage_name == stages[0] else None,
            )

        print("\n[DONE] Three-stage training complete.")
