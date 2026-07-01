"""
Test generators: LoRA, FiLM, ResMod.
"""

import torch
import sys
sys.path.insert(0, ".")

from src.generators import LoRAGenerator, FiLMGenerator, ResModGenerator


def test_lora_generator_shapes():
    """Verify LoRA generator output shapes."""
    d_lora = 64
    r = 4
    layer_configs = [
        {"name": "layer_a", "m": 256, "n": 64 * 9},   # 3x3 conv, 64 in_ch
        {"name": "layer_b", "m": 512, "n": 256},        # linear
        {"name": "layer_c", "m": 128, "n": 128 * 9},   # 3x3 conv, 128 in_ch
    ]

    gen = LoRAGenerator(d_lora, layer_configs, rank=r)
    z_lora = torch.randn(d_lora)

    results = gen(z_lora)

    assert len(results) == 3
    U_a, V_a = results["layer_a"]
    assert U_a.shape == (256, r), f"Expected (256, {r}), got {U_a.shape}"
    assert V_a.shape == (64 * 9, r), f"Expected (576, {r}), got {V_a.shape}"

    U_b, V_b = results["layer_b"]
    assert U_b.shape == (512, r)
    assert V_b.shape == (256, r)

    U_c, V_c = results["layer_c"]
    assert U_c.shape == (128, r)
    assert V_c.shape == (128 * 9, r)


def test_lora_output_range():
    """Verify LoRA outputs are in (0, 1) range due to sigmoid."""
    d_lora = 32
    r = 2
    configs = [{"name": "test", "m": 10, "n": 20}]
    gen = LoRAGenerator(d_lora, configs, rank=r)

    z = torch.randn(d_lora)
    U, V = gen(z)["test"]

    assert (U >= 0).all() and (U <= 1).all(), "U should be in [0,1]"
    assert (V >= 0).all() and (V <= 1).all(), "V should be in [0,1]"


def test_film_generator_shapes():
    """Verify FiLM generator output shapes."""
    d_film = 32
    node_configs = [
        {"name": "fuse1", "C": 256},
        {"name": "fuse2", "C": 512},
    ]

    gen = FiLMGenerator(d_film, node_configs)
    z_film = torch.randn(d_film)

    results = gen(z_film)
    assert len(results) == 2

    gamma1, beta1 = results["fuse1"]
    assert gamma1.shape == (256,)
    assert beta1.shape == (256,)

    gamma2, beta2 = results["fuse2"]
    assert gamma2.shape == (512,)
    assert beta2.shape == (512,)


def test_resmod_generator_shape():
    """Verify ResMod generator output shape."""
    d_query = 32
    num_queries = 300
    hidden_dim = 256

    gen = ResModGenerator(d_query, num_queries, hidden_dim)
    z_query = torch.randn(d_query)

    delta_q = gen(z_query)
    assert delta_q.shape == (num_queries, hidden_dim)
    assert (delta_q >= 0).all() and (delta_q <= 1).all()


def test_orth_weights_frozen():
    """Verify orthogonal weights are not trainable."""
    gen = LoRAGenerator(32, [{"name": "t", "m": 10, "n": 20}], rank=2)

    for name, param in gen.named_parameters():
        if "orth" in name:
            assert not param.requires_grad, f"{name} should be frozen"
        else:
            assert param.requires_grad, f"{name} should be trainable"


def test_deterministic_with_fixed_z():
    """Verify generators are deterministic given the same input."""
    d_lora = 16
    gen = LoRAGenerator(d_lora, [{"name": "t", "m": 8, "n": 12}], rank=2)
    gen.eval()

    z = torch.ones(d_lora)
    out1 = gen(z)["t"]
    out2 = gen(z)["t"]

    assert torch.allclose(out1[0], out2[0])
    assert torch.allclose(out1[1], out2[1])


if __name__ == "__main__":
    test_lora_generator_shapes()
    test_lora_output_range()
    test_film_generator_shapes()
    test_resmod_generator_shape()
    test_orth_weights_frozen()
    test_deterministic_with_fixed_z()
    print("All generator tests passed.")
