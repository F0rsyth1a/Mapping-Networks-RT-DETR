"""
Test gradient isolation between orthogonal control paths.
"""

import torch
import sys
sys.path.insert(0, ".")

from src.latent import OrthogonalLatentSpace
from src.generators import LoRAGenerator, FiLMGenerator, ResModGenerator
from src.utils import gradient_isolation_check


def test_gradient_isolation():
    """Verify that gradient paths are decoupled between control slices."""

    latent = OrthogonalLatentSpace(d_lora=32, d_film=16, d_query=16)

    lora_gen = LoRAGenerator(32, [
        {"name": "test1", "m": 8, "n": 16},
        {"name": "test2", "m": 16, "n": 32},
    ], rank=4)

    film_gen = FiLMGenerator(16, [
        {"name": "node1", "C": 64},
        {"name": "node2", "C": 128},
    ])

    resmod_gen = ResModGenerator(16, num_queries=10, hidden_dim=256)

    generators = {
        "lora": lora_gen,
        "film": film_gen,
        "query": resmod_gen,
    }

    result = gradient_isolation_check(latent, generators, verbose=False)
    assert result, "Gradient isolation check failed"


def test_lora_only_affects_lora_slice():
    """Verify that backprop through LoRA only touches z_lora slice."""
    latent = OrthogonalLatentSpace(d_lora=16, d_film=8, d_query=8)
    gen = LoRAGenerator(16, [{"name": "t", "m": 4, "n": 8}], rank=2)

    z_lora = latent.get_z_lora()
    U, V = gen(z_lora)["t"]
    loss = U.sum() + V.sum()
    loss.backward()

    grad = latent.z.grad
    assert grad[:16].abs().sum() > 0, "LoRA slice should have gradient"
    assert grad[16:].abs().sum() == 0, "Non-LoRA slices should be zero"


def test_film_only_affects_film_slice():
    """Verify that backprop through FiLM only touches z_film slice."""
    latent = OrthogonalLatentSpace(d_lora=16, d_film=8, d_query=8)
    gen = FiLMGenerator(8, [{"name": "n", "C": 32}])

    z_film = latent.get_z_film()
    gamma, beta = gen(z_film)["n"]
    loss = gamma.sum() + beta.sum()
    loss.backward()

    grad = latent.z.grad
    assert grad[16:24].abs().sum() > 0, "FiLM slice should have gradient"
    assert grad[:16].abs().sum() == 0, "LoRA slice should be zero"
    assert grad[24:].abs().sum() == 0, "Query slice should be zero"


def test_query_only_affects_query_slice():
    """Verify that backprop through ResMod only touches z_query slice."""
    latent = OrthogonalLatentSpace(d_lora=16, d_film=8, d_query=8)
    gen = ResModGenerator(8, num_queries=10, hidden_dim=256)

    z_query = latent.get_z_query()
    delta = gen(z_query)
    loss = delta.sum()
    loss.backward()

    grad = latent.z.grad
    assert grad[24:].abs().sum() > 0, "Query slice should have gradient"
    assert grad[:24].abs().sum() == 0, "Non-query slices should be zero"


if __name__ == "__main__":
    test_gradient_isolation()
    print("  [PASS] gradient isolation check")

    test_lora_only_affects_lora_slice()
    print("  [PASS] LoRA gradient isolation")

    test_film_only_affects_film_slice()
    print("  [PASS] FiLM gradient isolation")

    test_query_only_affects_query_slice()
    print("  [PASS] Query gradient isolation")

    print("All gradient isolation tests passed.")
