"""
Test OrthogonalLatentSpace: slicing and L2 normalization.
"""

import torch
import sys
sys.path.insert(0, ".")

from src.latent import OrthogonalLatentSpace


def test_latent_dimensions():
    """Verify correct slicing dimensions."""
    ls = OrthogonalLatentSpace(d_lora=256, d_film=128, d_query=128)

    z_lora = ls.get_z_lora()
    z_film = ls.get_z_film()
    z_query = ls.get_z_query()
    z_full = ls.get_full_z()

    assert z_lora.shape == (256,), f"Expected (256,), got {z_lora.shape}"
    assert z_film.shape == (128,), f"Expected (128,), got {z_film.shape}"
    assert z_query.shape == (128,), f"Expected (128,), got {z_query.shape}"
    assert z_full.shape == (512,), f"Expected (512,), got {z_full.shape}"


def test_l2_normalization():
    """Verify all slices are on the unit hypersphere."""
    ls = OrthogonalLatentSpace(d_lora=64, d_film=32, d_query=32)

    for _ in range(10):
        # Randomize z
        ls.z.data = torch.randn_like(ls.z)

        z_lora = ls.get_z_lora()
        z_film = ls.get_z_film()
        z_query = ls.get_z_query()

        assert torch.allclose(z_lora.norm(p=2), torch.tensor(1.0), atol=1e-6), \
            f"LoRA norm: {z_lora.norm(p=2)}"
        assert torch.allclose(z_film.norm(p=2), torch.tensor(1.0), atol=1e-6), \
            f"FiLM norm: {z_film.norm(p=2)}"
        assert torch.allclose(z_query.norm(p=2), torch.tensor(1.0), atol=1e-6), \
            f"Query norm: {z_query.norm(p=2)}"


def test_no_overlap():
    """Verify slices do not overlap."""
    ls = OrthogonalLatentSpace(d_lora=64, d_film=32, d_query=32)

    z_full = ls.get_full_z()
    z_lora = ls.get_z_lora()
    z_film = ls.get_z_film()
    z_query = ls.get_z_query()

    # Concatenated slices should reconstruct (up to L2 scaling) the full vector
    cats = torch.cat([z_lora * z_full[:64].norm(), 
                       z_film * z_full[64:96].norm(),
                       z_query * z_full[96:].norm()])
    # Direction should match (ignoring per-slice scaling)
    assert torch.allclose(
        torch.nn.functional.normalize(cats, dim=0),
        torch.nn.functional.normalize(z_full, dim=0),
        atol=1e-5
    ), "Sliced and normalized direction should match original"


def test_gradient_flow():
    """Verify gradients flow through normalization."""
    ls = OrthogonalLatentSpace(d_lora=16, d_film=8, d_query=8)

    z_lora = ls.get_z_lora()
    loss = z_lora.sum()
    loss.backward()

    assert ls.z.grad is not None, "Gradient should flow to z"
    # Only lora portion should receive gradients
    grad = ls.z.grad
    assert grad[:16].abs().sum() > 0, "LoRA portion should receive gradient"
    assert grad[16:24].abs().sum() == 0, "FiLM portion should not receive gradient"
    assert grad[24:].abs().sum() == 0, "Query portion should not receive gradient"


if __name__ == "__main__":
    test_latent_dimensions()
    test_l2_normalization()
    test_no_overlap()
    test_gradient_flow()
    print("All latent space tests passed.")
