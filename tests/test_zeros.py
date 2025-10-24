# tests/test_zeros.py
# -----------------------------------------------------------------------------
# Smoke tests for ZeroSAttention.
# - Build both associative-scan path and fallback path
# - Forward on small random input
# - Check output shape and autograd
# -----------------------------------------------------------------------------

import pytest
import torch

from sequencelab.build import build_attention
from sequencelab.config import ZeroSConfig


@pytest.mark.parametrize("use_associative", [True, False])
def test_zeros_forward_and_backward(use_associative):
    torch.manual_seed(0)
    B, T, D, H = 2, 32, 64, 8
    x = torch.randn(B, T, D, requires_grad=True)

    cfg = ZeroSConfig(
        n_embd=D,
        n_head=H,
        dropout=0.0,
        block_size=64,      # rotary embedding upper bound
        is_causal=True,
        init_params=True,   # exercise init path
        init_n_layers=2,
        use_norm=True,
        use_associative=use_associative,
    )
    layer = build_attention(cfg)
    layer.eval()

    y = layer(x)
    assert y.shape == (B, T, D)

    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
