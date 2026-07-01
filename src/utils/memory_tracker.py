"""
Memory tracker for GPU VRAM monitoring.

Provides utilities to log and track memory usage during training,
critical for ensuring the 90GB VRAM constraint is respected.
"""

from __future__ import annotations

import torch


class MemoryTracker:
    """Simple GPU memory tracking context manager / utility."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._start_allocated = 0
        self._start_reserved = 0

    def snapshot(self) -> dict:
        """Return current memory stats."""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
                "reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
            }
        return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}

    def log(self, tag: str = ""):
        """Print current memory usage."""
        stats = self.snapshot()
        print(
            f"[MEM {tag}] "
            f"allocated: {stats['allocated_gb']:.2f}GB, "
            f"reserved: {stats['reserved_gb']:.2f}GB, "
            f"peak: {stats['max_allocated_gb']:.2f}GB"
        )

    def reset_peak(self):
        """Reset peak memory tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def __enter__(self):
        if torch.cuda.is_available():
            self._start_allocated = torch.cuda.memory_allocated(self.device)
            self._start_reserved = torch.cuda.memory_reserved(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            end_allocated = torch.cuda.memory_allocated(self.device)
            end_reserved = torch.cuda.memory_reserved(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
            print(
                f"[MEM] Delta allocated: {(end_allocated - self._start_allocated) / 1e9:.2f}GB, "
                f"peak: {peak / 1e9:.2f}GB"
            )
