# tests/test_wave.py
# -----------------------------------------------------------------------------
# Smoke tests for WAVE/ARMA attention variants.
# - Build a layer via the factory for different variants
# - Forward on small random input
# - Check output shape and autograd
# -----------------------------------------------------------------------------

import pytest
import torch

from sequencelab.build import build_attention
from sequencelab.config import WaveConfig


@pytest.mark.parametrize(
    "variant",
    [
        "softmax_arma",
        "linear_arma",
        "gla_arma",
        "aft_arma",
    ],
)
def test_wave_forward_autograd(variant):
    torch.manual_seed(0)
    B, T, D, H = 2, 16, 32, 4
    x = torch.randn(B, T, D, requires_grad=True)

    cfg = WaveConfig(
        variant=variant,
        n_embd=D,
        n_head=H,
        dropout=0.0,
        ma_dropout=0.0,
        block_size=64,  # for softmax path
        bias=True,
        decay=True,     # allow on for GLA; ignored otherwise
    )
    layer = build_attention(cfg)
    layer.train(False)

    y = layer(x)
    assert y.shape == (B, T, D)

    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()