# src/attnlab/config.py
# -----------------------------------------------------------------------------
# Tiny, task-agnostic config dataclasses for the attention layers in attnlab.
# These configs are intentionally small and easy to instantiate from your own
# training scripts. The build factory (build_attention) accepts any of these.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal, Union

# NOTE: We re-use the FEM config defined near the FEM implementation to avoid
# duplication / drift. It lives with the layer for clarity and to keep knobs
# co-located with their implementation.
from .layers.fem import FreeEnergyMixerConfig as FEMConfig

# ---- WAVE (AR / ARMA) -------------------------------------------------------

WaveVariant = Literal[
    # Softmax attention
    "softmax", "softmax_arma",
    # Linear (kernelizable) attention
    "linear", "linear_arma",
    # Gated linear attention (GLA)
    "gla", "gla_arma",
    # Element-wise linear (AFT-like)
    "aft", "aft_arma",
    # Fixed causal linear (a.k.a. masked causal linear layer)
    "fixed", "fixed_arma",
]

@dataclass
class WaveConfig:
    """
    Config for WAVE/ARMA attention family (softmax / linear / GLA / AFT / fixed).

    The variant names follow a simple convention:
      - '<kind>'        → pure AR version
      - '<kind>_arma'   → ARMA version with the MA branch enabled

    Required:
      n_embd, n_head

    Common:
      dropout        → residual/value dropout inside the layer
      ma_dropout     → dropout on the MA/indirect weights branch (used by *_arma variants)
      block_size     → max causal length for softmax path (used to pre-allocate masks)
      max_len        → max length for 'fixed' attention (causal masked linear)
      bias           → include bias in linear projections
      decay          → enable/disable an extra decay gate in GLA
      generative_dependency → enable dependency cumsum in 'fixed' variants
    """
    variant: WaveVariant = "softmax_arma"

    n_embd: int = 512
    n_head: int = 8

    dropout: float = 0.0
    ma_dropout: float = 0.0

    block_size: Optional[int] = 1024  # used by softmax path
    max_len: Optional[int] = None     # used by fixed attention; defaults to block_size if None

    bias: bool = True
    decay: bool = False               # used by GLA
    generative_dependency: bool = False  # used by fixed/fixed_arma


# ---- ZeroS (Zero-Sum linear attention) --------------------------------------

@dataclass
class ZeroSConfig:
    """
    Config for ZeroSAttention.

    Required:
      n_embd, n_head

    Common:
      dropout        → residual/value dropout in the output projection
      block_size     → rotary embedding table length (upper bound)
      bias           → include bias in linear projections
      is_causal      → apply causal behavior
      init_params    → enable GPT-style init scaling (optional)
      use_norm       → apply LayerNorm/RMS-like post attention normalization inside head
      use_associative→ enable associative (scan) path vs. fallback pairwise path
      init_n_layers  → number of layers to estimate init scaling (when init_params=True)
    """
    n_embd: int = 512
    n_head: int = 8

    dropout: float = 0.0
    block_size: int = 1024

    bias: bool = True
    is_causal: bool = True

    init_params: bool = False
    init_n_layers: int = 12

    use_norm: bool = True
    use_associative: bool = True


# ---- Union of supported configs for the build() factory ---------------------

AnyAttentionConfig = Union[WaveConfig, ZeroSConfig, FEMConfig]

__all__ = [
    "WaveConfig",
    "ZeroSConfig",
    "FEMConfig",
    "AnyAttentionConfig",
    "WaveVariant",
]
