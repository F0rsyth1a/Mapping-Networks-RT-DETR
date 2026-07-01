"""
Gradient isolation check utility.

Verifies that gradients do not leak between orthogonal control paths
by checking the leaf tensor latent.z.grad after independent backprops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict


def gradient_isolation_check(
    latent: nn.Module,
    generators: Dict[str, nn.Module],
    verbose: bool = False,
) -> bool:
    """
    Verify gradient paths are isolated between control slices.

    Runs independent forward+backward through each generator and checks
    that only the corresponding slice of latent.z receives gradients.
    """
    passed = True

    for key, gen in generators.items():
        zero_grad_all(latent, gen)

        if key == "lora":
            z = latent.get_z_lora()
            idx_range = (0, latent.d_lora)
        elif key == "film":
            z = latent.get_z_film()
            idx_range = (latent.d_lora, latent.d_lora + latent.d_film)
        elif key == "query":
            z = latent.get_z_query()
            idx_range = (latent.d_lora + latent.d_film, latent.d_lora + latent.d_film + latent.d_query)
        else:
            continue

        output = gen(z)
        loss = flatten_and_sum(output)
        loss.backward()

        grad = latent.z.grad
        if grad is None:
            if verbose:
                print(f"[WARN] No gradient at all for z_{key}")
            passed = False
            continue

        lo, hi = idx_range
        grad_own = grad[lo:hi].abs().sum().item()
        grad_other = (grad[:lo].abs().sum().item() + grad[hi:].abs().sum().item())

        if grad_own == 0:
            if verbose:
                print(f"[WARN] No gradient to z_{key} slice")
            passed = False

        if grad_other > 1e-10:
            if verbose:
                print(f"[FAIL] Gradient leak: {key} -> other slices ({grad_other:.2e})")
            passed = False

    if verbose and passed:
        print("[PASS] Gradient isolation verified for all control paths")

    zero_grad_all(latent, *generators.values())
    return passed


def flatten_and_sum(output):
    if isinstance(output, dict):
        s = torch.tensor(0.0, device=_device_of(output))
        for v in output.values():
            s = s + flatten_and_sum(v)
        return s
    elif isinstance(output, tuple):
        return sum(t.sum() for t in output)
    elif isinstance(output, torch.Tensor):
        return output.sum()
    return torch.tensor(0.0)


def _device_of(output):
    if isinstance(output, dict):
        for v in output.values():
            return _device_of(v)
    elif isinstance(output, tuple):
        return output[0].device if len(output) > 0 else torch.device("cpu")
    elif isinstance(output, torch.Tensor):
        return output.device
    return torch.device("cpu")


def zero_grad_all(latent, *modules):
    if latent.z.grad is not None:
        latent.z.grad = None
    for m in modules:
        m.zero_grad()
