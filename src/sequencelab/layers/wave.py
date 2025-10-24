# -----------------------------------------------------------------------------
# WAVE (ARMA-Attention) core layers
# -----------------------------------------------------------------------------
# This file implements the core, task-agnostic attention modules used by WAVE:
#   - Softmax (quadratic) attention + ARMA residual branch
#   - Linear attention  + ARMA residual branch
#   - Gated linear attention (GLA) + ARMA residual branch
#   - Element-wise linear (AFT-like) + ARMA residual branch
#
# The AR (autoregressive) term is the base attention; the MA (moving-average)
# term is applied to the one-step-ahead residuals with indirect MA weight
# generation (linear-time for linear/GLA/AFT bases).
#
# Notes:
# - No full Transformer block is provided here; only attention modules.
# - These layers accept (B, T, D) and return (B, T, D).
# - PyTorch >= 2.0 is recommended for scaled_dot_product_attention in softmax.
# -----------------------------------------------------------------------------

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# MA activations (indirect MA weight generation)
# -----------------------------------------------------------------------------
def ma_q_activation(x: torch.Tensor) -> torch.Tensor:
    """Query activation for the MA branch (negative LeakyReLU on scaled inputs)."""
    x = -x / math.sqrt(x.shape[-1])
    x = -F.leaky_relu(x, negative_slope=0.02)
    return x


def ma_k_activation(x: torch.Tensor, k: float = 0.02) -> torch.Tensor:
    """Key activation for the MA branch (sigmoid on scaled inputs)."""
    x = x / math.sqrt(x.shape[-1])
    return 1.0 / (1.0 + torch.exp(-x * k))


# -----------------------------------------------------------------------------
# A tiny MA scaled dot-product attention used by softmax-ARMA
#   - We apply ma_q_activation on Q and ma_k_activation (+dropout) on K
#   - We enforce causality via tril
#   - When return_weight=True, we return the (causal) attention matrix (no @V)
# -----------------------------------------------------------------------------
def ma_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: Optional[torch.Tensor],
    dropout_p: float = 0.0,
    is_causal: bool = True,
    return_weight: bool = False,
) -> torch.Tensor:
    """
    query, key: (B, H, T, d)
    value:     (B, H, T, d) or None if return_weight=True
    """
    # MA activations
    q = ma_q_activation(query)
    k = ma_k_activation(F.dropout(key, p=dropout_p, training=True))

    # Score and apply causal mask by lower-triangular keep
    attn = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, T)
    if is_causal:
        attn = attn.tril(diagonal=0)

    if return_weight:
        return attn

    assert value is not None, "value must be provided when return_weight=False"
    out = torch.matmul(attn, value)  # (B, H, T, d)
    return out


# -----------------------------------------------------------------------------
# Weighted cumulative-sum helpers (for GLA)
# -----------------------------------------------------------------------------
def calculate_weighted_cumsum_weight(weights: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute stable cumulative product of 'weights' along dim via log-sum-exp trick."""
    w = torch.clamp(weights, min=1e-6)
    log_w = torch.log(w)
    log_cumprod = torch.cumsum(log_w, dim=dim)
    log_cumprod = torch.clamp(log_cumprod, max=30.0, min=-30.0)
    return torch.exp(log_cumprod) + 1e-6


def apply_weighted_cumsum_weight(x: torch.Tensor, cumprod_weights: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply invertible weighted cumsum: cumsum(x / cumprod) * cumprod."""
    return torch.cumsum(x / cumprod_weights, dim=dim) * cumprod_weights


# -----------------------------------------------------------------------------
# Softmax (quadratic) attention + ARMA
# -----------------------------------------------------------------------------
class CausalSelfAttentionARMA(nn.Module):
    """
    Quadratic softmax attention as AR term, plus an MA branch computed on the
    one-step residuals (y_t - v_t+1) with indirect MA weight generation.

    Shapes:
      x: (B, T, D) → out: (B, T, D)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        ma_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.dropout = dropout
        self.ma_dropout = ma_dropout

        # Q,K for AR; V taken as identity (x)
        self.c_attn = nn.Linear(n_embd, 2 * n_embd, bias=bias)
        # K2 for MA
        self.k2 = nn.Linear(n_embd, n_embd, bias=bias)
        # Shared output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.resid_dropout = nn.Dropout(dropout)
        self.ma_value_dropout = nn.Dropout(ma_dropout)

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # AR path (softmax attention)
        q, k = self.c_attn(x).split(self.n_embd, dim=-1)
        q = self._shape_heads(q)
        k = self._shape_heads(k)
        v = self._shape_heads(x)  # identity value

        # PyTorch 2.0+ SDPA; causal + dropout when training
        y_ar = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )  # (B,H,T,d)

        # MA path on one-step residuals: e_t = v_{t} - y_{t-1}
        e = v[:, :, 1:, :] - y_ar[:, :, :-1, :]                            # (B,H,T-1,d)
        q2 = q[:, :, 1:, :]                                                # (B,H,T-1,d)
        k2 = self._shape_heads(self.k2(x[:, :-1]))                         # (B,H,T-1,d)

        y_ma = ma_scaled_dot_product_attention(
            q2, k2, e, dropout_p=self.ma_dropout, is_causal=True
        )                                                                   # (B,H,T-1,d)
        # Align time by padding a zero at the start
        y_ma = torch.cat([torch.zeros_like(y_ma[:, :, :1, :]), y_ma], dim=2)  # (B,H,T,d)

        # Output projection + branch-wise dropout (shared proj as in the original impl)
        y_ar = y_ar.transpose(1, 2).contiguous().view(B, T, D)
        y_ma = y_ma.transpose(1, 2).contiguous().view(B, T, D)

        out = self.resid_dropout(self.c_proj(y_ar)) + self.ma_value_dropout(self.c_proj(y_ma))
        return out


# -----------------------------------------------------------------------------
# Linear attention + ARMA (O(Nd^2))
# -----------------------------------------------------------------------------
class LinearAttentionARMA(nn.Module):
    """
    Linear attention as AR term (kernelized accumulation of K^T V),
    plus an MA branch on one-step residuals with indirect weights.

    Shapes:
      x: (B, T, D) → out: (B, T, D)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        ma_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.q1 = nn.Linear(n_embd, n_embd, bias=bias)
        self.k1 = nn.Linear(n_embd, n_embd, bias=bias)
        self.v1 = nn.Linear(n_embd, n_embd, bias=bias)

        self.k2 = nn.Linear(n_embd, n_embd, bias=bias)  # MA keys
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.ma_dropout = nn.Dropout(ma_dropout)

    def _shape_heads(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        return x.reshape(B, L, self.n_head, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # ----- AR (linear attention) -----
        Q = self._shape_heads(self.q1(x), B, L)                              # (B,L,H,d)
        K = self._shape_heads(self.k1(self.dropout(x)), B, L).unsqueeze(-1)  # (B,L,H,d,1)
        V = self._shape_heads(self.v1(x), B, L).unsqueeze(-1).permute(0, 1, 2, 4, 3)  # (B,L,H,1,d)

        # State accumulation W_t = sum_{i<=t} K_i^T V_i  (d x d per head)
        W = torch.einsum("blhdk,blhke->blhde", K, V)  # (B,L,H,d,d)
        W = torch.cumsum(W, dim=1)
        O1 = torch.einsum("blhd,blhde->blhe", Q, W)   # (B,L,H,d)  -- AR output

        # ----- MA (indirect weights on residuals) -----
        # Residuals between value (shifted) and AR output (shifted)
        Y = O1.unsqueeze(-2)                               # (B,L,H,1,d)
        V_raw = self._shape_heads(x, B, L).unsqueeze(-2)   # (B,L,H,1,d)  (identity v-path)
        E = V_raw[:, 1:] - Y[:, :-1]                       # (B,L-1,H,1,d)

        # Apply MA activations/weights
        k2 = self.ma_dropout(ma_k_activation(self.k2(x[:, :-1]))).reshape(B, L - 1, self.n_head, -1).unsqueeze(-1)  # (B,L-1,H,d,1)
        q2 = ma_q_activation(Q[:, :-1])                    # (B,L-1,H,d)

        WE = torch.einsum("blhdk,blhke->blhde", k2, E)     # (B,L-1,H,d,d)
        WE = torch.cumsum(WE, dim=1)
        O2 = torch.einsum("blhd,blhde->blhe", q2, WE)      # (B,L-1,H,d)
        O2 = torch.cat([torch.zeros_like(O2[:, :1]), O2], dim=1)  # (B,L,H,d)

        O1 = self.dropout(O1).reshape(B, L, D)
        O2 = self.ma_dropout(O2).reshape(B, L, D)
        out = self.c_proj(O1 + O2)
        return out


# -----------------------------------------------------------------------------
# Gated Linear Attention (GLA) + ARMA
# -----------------------------------------------------------------------------
class GatedLinearAttentionARMA(nn.Module):
    """
    GLA as AR term (with forget/decay gate), plus an MA branch on residuals.

    Shapes:
      x: (B, T, D) → out: (B, T, D)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float = 0.0,
        ma_dropout: float = 0.0,
        bias: bool = True,
        decay: bool = False,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.decay = decay

        self.q1 = nn.Linear(n_embd, n_embd, bias=bias)
        self.k1 = nn.Linear(n_embd, n_embd, bias=bias)
        self.k2 = nn.Linear(n_embd, n_embd, bias=bias)  # MA keys

        self.gw = nn.Linear(self.head_dim, 1, bias=bias)  # gating vector -> scalar gate per head
        if self.decay:
            self.sw = nn.Linear(self.head_dim, 1, bias=bias)  # optional state decay

        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.ma_dropout = nn.Dropout(ma_dropout)

    def _shape_heads(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        return x.reshape(B, L, self.n_head, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # ----- AR (GLA) -----
        Q = self._shape_heads(self.q1(x), B, L)                              # (B,L,H,d)
        K = self._shape_heads(self.k1(self.dropout(x)), B, L).unsqueeze(-1)  # (B,L,H,d,1)
        V = self._shape_heads(x, B, L).unsqueeze(-1).permute(0, 1, 2, 4, 3)  # (B,L,H,1,d)

        # State: W_t = Σ g_prod(i) * K_i^T V_i   with learnable gate cumulative product
        W = torch.einsum("blhdk,blhke->blhde", K, V)                          # (B,L,H,d,d)
        G = torch.sigmoid(self.gw(x.view(B, L, self.n_head, -1)))             # (B,L,H,1)
        G = calculate_weighted_cumsum_weight(G, dim=1).unsqueeze(-1).expand(-1, -1, -1, self.head_dim, self.head_dim)  # (B,L,H,d,d)

        if self.decay:
            # Optional additional per-step decay
            R = F.silu(self.sw(K[:, :, :, :, 0])).unsqueeze(-1).expand(-1, -1, -1, self.head_dim, self.head_dim)
        else:
            R = 1.0

        W = apply_weighted_cumsum_weight(W * R, G, dim=1)
        O1 = torch.einsum("blhd,blhde->blhe", Q, W)                            # (B,L,H,d)

        # ----- MA (residuals) -----
        E = V[:, 1:, :] - O1.unsqueeze(-2)[:, :-1, :]                          # (B,L-1,H,1,d)
        q2 = ma_q_activation(Q[:, :-1])                                        # (B,L-1,H,d)
        k2 = self.ma_dropout(ma_k_activation(self.k2(x[:, :-1]))).reshape(B, L - 1, self.n_head, -1).unsqueeze(-1)  # (B,L-1,H,d,1)

        W_ma = torch.einsum("blhdk,blhke->blhde", k2, E)                       # (B,L-1,H,d,d)
        W_ma = torch.cumsum(W_ma, dim=1)
        O2 = torch.einsum("blhd,blhde->blhe", q2, W_ma)                         # (B,L-1,H,d)
        O2 = torch.cat([torch.zeros_like(O2[:, :1]), O2], dim=1)               # (B,L,H,d)

        O1 = self.dropout(O1).reshape(B, L, D)
        O2 = self.ma_dropout(O2).reshape(B, L, D)
        out = self.c_proj(O1 + O2)
        return out


# -----------------------------------------------------------------------------
# Element-wise linear (AFT-like) + ARMA
# -----------------------------------------------------------------------------
class TwoStageSelfgatingRNNARMA(nn.Module):
    """
    Element-wise linear attention (AFT-like) with exponential keys as AR term,
    plus an MA branch with indirect weights via (q,k)-style activations.

    Shapes:
      x: (B, T, D) → out: (B, T, D)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,              # kept for interface parity; not used internally
        dropout: float = 0.0,
        ma_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.D = n_embd

        # AR (AFT-like): O = σ(Q) * [cumsum(exp(K) * V) / cumsum(exp(K))]
        self.q1 = nn.Linear(self.D, self.D, bias=bias)
        self.k1 = nn.Linear(self.D, self.D, bias=bias)

        # MA branch (only K2; we reuse Q from AR branch with ma_q_activation)
        self.k2 = nn.Linear(self.D, self.D, bias=bias)

        self.c_proj = nn.Linear(self.D, self.D, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.ma_dropout = nn.Dropout(ma_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # ----- AR (AFT-like) -----
        V = x
        K = self.k1(x)
        K = self.dropout(torch.exp(K))
        K_cum = K.cumsum(dim=1) + 1e-6
        H = K * V
        S = H.cumsum(dim=1) / K_cum
        Q = torch.sigmoid(self.q1(x))
        O_ar = S * Q  # (B,L,D)

        # ----- MA (residuals) -----
        E = torch.cat([torch.zeros_like(O_ar[:, :1, :]), V[:, 1:, :] - O_ar[:, :-1, :]], dim=1)  # (B,L,D)

        # Use previous input (shifted by 1) to generate K2
        qk2_in = torch.cat([torch.zeros_like(x[:, :1, :]), x[:, :-1, :]], dim=1)
        K2 = self.ma_dropout(ma_k_activation(self.k2(qk2_in)))  # (B,L,D)

        H2 = K2 * E
        S2 = H2.cumsum(dim=1)

        Q2 = ma_q_activation(Q)
        Q2 = torch.cat([torch.zeros_like(Q2[:, :1, :]), Q2[:, :-1, :]], dim=1)

        O_ma = S2 * Q2  # (B,L,D)

        out = self.c_proj(self.dropout(O_ar) + self.ma_dropout(O_ma))
        return out


# -----------------------------------------------------------------------------
# Unified WAVE wrapper
# -----------------------------------------------------------------------------
class Wave(nn.Module):
    """
    Unified WAVE attention that selects a base AR mechanism:
      base = {"softmax", "linear", "gla", "aft"}
    and adds the MA branch accordingly.

    Example:
        layer = Wave(n_embd=512, n_head=8, base="linear", dropout=0.0, ma_dropout=0.0)
        y = layer(x)  # (B,T,D)
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        base: str = "linear",
        dropout: float = 0.0,
        ma_dropout: float = 0.0,
        bias: bool = True,
        decay: bool = False,  # only used by GLA
    ):
        super().__init__()
        base = base.lower()
        if base == "softmax":
            self.impl = CausalSelfAttentionARMA(
                n_embd=n_embd, n_head=n_head, dropout=dropout, ma_dropout=ma_dropout, bias=bias
            )
        elif base == "linear":
            self.impl = LinearAttentionARMA(
                n_embd=n_embd, n_head=n_head, dropout=dropout, ma_dropout=ma_dropout, bias=bias
            )
        elif base == "gla":
            self.impl = GatedLinearAttentionARMA(
                n_embd=n_embd, n_head=n_head, dropout=dropout, ma_dropout=ma_dropout, bias=bias, decay=decay
            )
        elif base == "aft":
            self.impl = TwoStageSelfgatingRNNARMA(
                n_embd=n_embd, n_head=n_head, dropout=dropout, ma_dropout=ma_dropout, bias=bias
            )
        else:
            raise ValueError(f"Unknown WAVE base: {base}. Choose from 'softmax' | 'linear' | 'gla' | 'aft'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)


__all__ = [
    "Wave",
    "CausalSelfAttentionARMA",
    "LinearAttentionARMA",
    "GatedLinearAttentionARMA",
    "TwoStageSelfgatingRNNARMA",
    "ma_q_activation",
    "ma_k_activation",
    "ma_scaled_dot_product_attention",
]
