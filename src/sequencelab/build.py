# src/attnlab/build.py
# -----------------------------------------------------------------------------
# Single factory entry-point for creating attention layers from small configs.
# This version is resilient to constructor styles:
# - if a layer exposes a single 'config/cfg' parameter, pass a SimpleNamespace
# - else we pass filtered **kwargs that match the ctor signature
# -----------------------------------------------------------------------------

from __future__ import annotations
from types import SimpleNamespace
from typing import Any
import inspect

import torch.nn as nn

from .config import WaveConfig, ZeroSConfig, FEMConfig, AnyAttentionConfig

# WAVE / ARMA family
from .layers.wave import (
    CausalSelfAttentionARMA,       # softmax (ARMA)
    LinearAttentionARMA,           # linear (ARMA)
    GatedLinearAttentionARMA,      # GLA (ARMA)
    TwoStageSelfgatingRNNARMA,     # AFT-like (ARMA)
)

# ZeroS
from .layers.zeros import ZeroSAttention

# FEM
from .layers.fem import FreeEnergyMixer


# ---- helpers ----------------------------------------------------------------

def _ns(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def _instantiate_layer(cls: type[nn.Module], base_kwargs: dict[str, Any], **extra_kwargs: Any) -> nn.Module:
    """
    Signature-aware instantiation:
    - If the ctor looks like __init__(self, config/cfg), pass a SimpleNamespace.
    - Else, pass **kwargs but filter keys to parameters in the signature
      (unless the ctor has **kwargs).
    """
    # Merge kwargs
    all_kwargs = dict(base_kwargs)
    all_kwargs.update(extra_kwargs)

    # Inspect the signature
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.values())[1:]  # skip 'self'

    # Detect **kwargs on the ctor
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    # Collect accepted keyword names (pos-or-kw + kw-only)
    accepted = [
        p.name for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]

    # Config-style ctor? (single config/cfg argument)
    if len(accepted) == 1 and accepted[0] in {"config", "cfg"} and not has_var_kw:
        return cls(_ns(**all_kwargs))

    # Otherwise: kwargs-style. Filter if no **kwargs.
    if not has_var_kw:
        all_kwargs = {k: v for k, v in all_kwargs.items() if k in accepted}

    return cls(**all_kwargs)


def _build_wave(cfg: WaveConfig) -> nn.Module:
    """
    Create a WAVE family attention layer based on cfg.variant.
    Supports both ctor styles (kwargs or single config object).
    """
    # Resolve lengths (some variants may ignore these)
    block_size = cfg.block_size or 1024
    max_len = cfg.max_len or block_size

    # Minimal set of knobs shared by most implementations
    base = dict(
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
        bias=cfg.bias,
        block_size=block_size,   # accepted only by classes that declare it
        max_len=max_len,         # ditto
        ma_dropout=cfg.ma_dropout,
        decay=cfg.decay,
    )

    v = cfg.variant.lower()

    # ---- Softmax (ARMA)
    if v == "softmax_arma":
        return _instantiate_layer(CausalSelfAttentionARMA, base)

    # ---- Linear (ARMA)
    if v == "linear_arma":
        return _instantiate_layer(LinearAttentionARMA, base)

    # ---- Gated Linear Attention (ARMA)
    if v == "gla_arma":
        return _instantiate_layer(GatedLinearAttentionARMA, base)

    # ---- AFT-like (element-wise linear, ARMA)
    if v == "aft_arma":
        return _instantiate_layer(TwoStageSelfgatingRNNARMA, base)

    raise ValueError(f"[attnlab] Unknown WaveConfig.variant: {cfg.variant!r}")


def _build_zeros(cfg: ZeroSConfig) -> nn.Module:
    """
    Create a ZeroSAttention layer. Also support both ctor styles.
    """
    base = dict(
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        bias=cfg.bias,
        dropout=cfg.dropout,
        block_size=cfg.block_size,
        is_causal=cfg.is_causal,
        init_params=cfg.init_params,
        init_n_layers=cfg.init_n_layers,
        use_norm=cfg.use_norm,
        use_associative=cfg.use_associative,
    )
    return _instantiate_layer(ZeroSAttention, base)


def _build_fem(cfg: FEMConfig) -> nn.Module:
    """
    Create a FreeEnergyMixer (FEM) module directly from its dataclass config.
    The FEM implementation expects its own dataclass; pass-through is correct.
    """
    return FreeEnergyMixer(cfg)


# ---- public factory ----------------------------------------------------------

def build_attention(cfg: AnyAttentionConfig) -> nn.Module:
    """
    Unified attention builder.

    Args:
        cfg: One of {WaveConfig, ZeroSConfig, FEMConfig}.

    Returns:
        nn.Module: an attention layer with forward(x: [B,T,D], mask?) -> [B,T,D]
    """
    if isinstance(cfg, WaveConfig):
        return _build_wave(cfg)
    if isinstance(cfg, ZeroSConfig):
        return _build_zeros(cfg)
    if isinstance(cfg, FEMConfig):
        return _build_fem(cfg)

    raise TypeError(
        "[attnlab] Unsupported config type. "
        "Expected WaveConfig, ZeroSConfig, or FEMConfig."
    )


__all__ = ["build_attention"]
