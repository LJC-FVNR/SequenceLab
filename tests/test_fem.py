# tests/test_fem.py
# -----------------------------------------------------------------------------
# Smoke tests for FreeEnergyMixer (FEM).
# - Exercise different priors that do not require optional CUDA backends
# - Check both branches: with/without the temperature/LSE path
# - Validate output shapes and autograd on CPU
# -----------------------------------------------------------------------------

import pytest
import torch

from sequencelab.build import build_attention
from sequencelab.config import FEMConfig


@pytest.mark.parametrize("prior_type", ["softmax", "linear", "gla", "rnn_softmax"])
@pytest.mark.parametrize("use_temp_lse", [True, False])
def test_fem_forward_autograd(prior_type, use_temp_lse):
    torch.manual_seed(0)
    B, T, D, H = 2, 24, 64, 4
    x = torch.randn(B, T, D, requires_grad=True)

    # Optional attention mask: True=keep, False=pad
    # Keep most tokens, pad a few at the tail
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -3:] = torch.tensor([1, 0, 1], dtype=torch.bool)  # small variation

    cfg = FEMConfig(
        n_embd=D,
        n_head=H,
        prior_type=prior_type,
        dropout=0.0,
        causal=True,
        bias=True,

        # Paper-aligned parameter budgets:
        # fem_ratio=0.5 -> fem_dim ~ D/2; p_t_to_fem_ratio=4.0 -> |Q|+|K| ~= 2D
        fem_ratio=0.5,
        p_t_to_fem_ratio=4.0,

        # Toggle the free-energy branch
        use_temperature=use_temp_lse,
        use_lse=use_temp_lse,
        use_outer_gate=True,

        # Keep extras on; they are CPU-safe in this implementation
        use_rope=True,
        use_conv=True,
        conv_hidden=32,
        conv_norm_first=True,
        conv_bidirectional=False,
    )

    layer = build_attention(cfg)
    layer.eval()

    # Forward
    y = layer(x, attention_mask=mask)
    assert y.shape == (B, T, D)

    # Backprop
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_fem_softmax_minimal_no_branches():
    """Sanity: FEM with softmax prior and all free-energy branches disabled."""
    torch.manual_seed(0)
    B, T, D, H = 1, 8, 32, 4
    x = torch.randn(B, T, D, requires_grad=True)

    cfg = FEMConfig(
        n_embd=D, n_head=H,
        prior_type="softmax",
        dropout=0.0,
        causal=True,
        bias=True,
        fem_ratio=0.5, p_t_to_fem_ratio=4.0,
        use_temperature=False, use_lse=False, use_outer_gate=False,
        use_rope=True, use_conv=False,
    )
    layer = build_attention(cfg).eval()
    y = layer(x)
    assert y.shape == (B, T, D)
    y.sum().backward()
    assert x.grad is not None
