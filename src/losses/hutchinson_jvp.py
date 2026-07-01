"""
Hutchinson Trace Estimator for Jacobian-via-VJP.

    ||J||_F^2 = E_v[ ||J^T v||^2 ]

For generators with layers (LoRA), VJP is computed per-layer via
torch.func.vjp to keep each computation graph small (~50K elements),
preserving gradient flow to z. For small generators (FiLM, ResMod),
single VJP is used. Fallback to leaf autograd.grad if torch.func
unavailable.
"""

from __future__ import annotations

import torch
from typing import Callable, Dict, Any, Tuple


def _flatten(t: Any) -> torch.Tensor:
    if isinstance(t, tuple):
        return torch.cat([_flatten(x) for x in t])
    elif isinstance(t, torch.Tensor):
        return t.reshape(-1)
    elif isinstance(t, dict):
        parts = [_flatten(v) for v in t.values()]
        return torch.cat(parts) if parts else torch.tensor(0.0)
    else:
        raise TypeError(type(t))


# ------------------------------------------------------------------
# torch.func.vjp path — gradient-preserving
# ------------------------------------------------------------------

def _smoothness_vjp_single(fn: Callable, z: torch.Tensor) -> torch.Tensor:
    """||J(z)^T v||^2 via torch.func.vjp. Full output, gradient flowing."""
    from torch.func import vjp

    def func(zz):
        return _flatten(fn(zz))

    flat_out, vjp_fn = vjp(func, z)
    v = torch.randn_like(flat_out)
    vjp_val, = vjp_fn(v)
    return (vjp_val ** 2).sum()


def _smoothness_vjp_per_layer(gen, z: torch.Tensor) -> torch.Tensor:
    """
    Per-layer VJP for LoRAGenerator. Each layer's output (~50K elements)
    is small enough for torch.func.vjp to handle stably. Gradient flows
    through each layer's VJP back to the shared z.
    """
    from torch.func import vjp

    total = torch.tensor(0.0, device=z.device)

    for name in getattr(gen, "layer_names", []):
        def layer_fn(zz, n=name):
            return _flatten(gen(zz)[n])

        flat_out, vjp_fn = vjp(layer_fn, z)
        v = torch.randn_like(flat_out)
        vjp_val, = vjp_fn(v)
        total = total + (vjp_val ** 2).sum()

    return total


# ------------------------------------------------------------------
# Fallback — leaf autograd.grad, metric only
# ------------------------------------------------------------------

def _smoothness_fallback(fn: Callable, z: torch.Tensor) -> torch.Tensor:
    """Leaf autograd.grad. Detached — metric only, no gradient to z."""
    zg = z.detach().requires_grad_(True)
    flat_out = _flatten(fn(zg))
    v = torch.randn_like(flat_out)
    scalar = torch.dot(v, flat_out)
    grad_z = torch.autograd.grad(scalar, zg)[0]
    val = (grad_z ** 2).sum()
    return torch.nan_to_num(val.detach(), nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------

_HAS_FUNC = None


def _has_torch_func():
    global _HAS_FUNC
    if _HAS_FUNC is None:
        try:
            from torch.func import vjp as _
            _HAS_FUNC = True
        except ImportError:
            _HAS_FUNC = False
    return _HAS_FUNC


def hutchinson_jvp(
    fn: Callable[[torch.Tensor], Any],
    z: torch.Tensor,
    num_samples: int = 1,
    _is_lora: bool = False,
) -> torch.Tensor:
    total = torch.tensor(0.0, device=z.device)

    if _has_torch_func():
        jvp_impl = _smoothness_vjp_per_layer if _is_lora else _smoothness_vjp_single
    else:
        jvp_impl = _smoothness_fallback

    for _ in range(num_samples):
        val = jvp_impl(fn, z)
        total = total + torch.nan_to_num(val, nan=0.0, posinf=1e6, neginf=0.0)

    return total / num_samples


def compute_smoothness_hutchinson(
    generators: Dict[str, Any],
    z_slices: Dict[str, torch.Tensor],
    num_samples: int = 1,
) -> Tuple[torch.Tensor, dict]:
    total = torch.tensor(0.0, device=next(iter(z_slices.values())).device)
    info = {}

    for key in ["lora", "film", "query"]:
        if key not in generators or key not in z_slices:
            continue
        is_lora = (key == "lora")
        val = hutchinson_jvp(generators[key], z_slices[key], num_samples, _is_lora=is_lora)
        total = total + val
        info[f"smooth_{key}"] = val.item()

    return total, info
