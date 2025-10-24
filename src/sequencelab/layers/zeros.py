# -----------------------------------------------------------------------------
# ZeroS (Zero‑Sum Linear Attention) — core attention module
# -----------------------------------------------------------------------------
# This file provides a minimal, task‑agnostic PyTorch implementation of ZeroS:
#   - Reweighted zero‑sum softmax residuals (0th order removed) with gates
#   - Linear‑time associative scan (prefix sums) for efficiency
#   - Fallback non‑associative path that explicitly forms per‑step weights
#   - Optional RoPE and light per‑head LayerNorm on the output
#
# Shapes:
#   input  x: (B, T, D)
#   output y: (B, T, D)
#
# The API matches the original reference implementation closely. You can pass
# a SimpleNamespace-like config with the following fields (defaults shown below):
#
#   n_embd: int
#   n_head: int
#   bias: bool = True
#   dropout: float = 0.0
#   block_size: int  # for RoPE tables & causal mask
#   is_causal: bool = True
#   init_params: bool = False
#   init_n_layers: int = 12         # used only when init_params=True
#   use_norm: bool = True           # LayerNorm(d) applied per head on output
#   use_associative: bool = True    # linear-time scan path if True
#   is_first_layer: bool = False    # optionally restore 0th-order term
#
# -----------------------------------------------------------------------------

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Rotary helpers
# -----------------------------------------------------------------------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    # x, cos, sin shapes are broadcastable (B,T,H,d) with (1,T,1,d)
    return x * cos + rotate_half(x) * sin if not transpose else x * cos - rotate_half(x) * sin


def get_rotary_embedding(L: int, D: int, base: float = 10000.0, device: str | torch.device | None = None):
    """
    Build RoPE cos/sin tables of shape (1, L, 1, D).
    D must be even.
    """
    assert D % 2 == 0, "RoPE dimension must be even"
    dev = torch.device(device) if device is not None else None
    inv = 1.0 / (base ** (torch.arange(0, D, 2, device=dev).float() / D))
    pos = torch.arange(L, device=dev, dtype=torch.float32)
    s = pos[:, None] * inv[None]
    # Repeat interleave along feature for easy broadcasting
    cos = torch.cos(s).repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2)  # (1, L, 1, D)
    sin = torch.sin(s).repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2)  # (1, L, 1, D)
    return cos, sin


# -----------------------------------------------------------------------------
# Core compute: associative (linear-time scan) and fallback
# -----------------------------------------------------------------------------
@torch.no_grad()
def _make_causal_mask(L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(1, 1, L, L) boolean lower-triangular mask."""
    bias = torch.tril(torch.ones(L, L, device=device, dtype=dtype))
    return bias.view(1, 1, L, L).bool()


def compute_o(
    q: torch.Tensor,              # (B, L, H, d), unit-norm
    s_i: torch.Tensor,            # (B, L, H, 1), per-index radial logits
    k: torch.Tensor,              # (B, L, H, d), unit-norm
    v: torch.Tensor,              # (B, L, H, d)
    gate: torch.Tensor,           # (B, L, H, 3) -> (σ^1, σ^h, optionally σ^0)
    mask: Optional[torch.Tensor] = None,  # (1,1,L,L) boolean causal mask
    causal: bool = True,
    associative: bool = True,
    is_first_layer: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    ZeroS output (B, L, H, d).

    Two paths:
      - associative=True: linear-time prefix-sum scan; efficient and parallel
      - associative=False: fallback explicit weight formation (O(T^2) memory)
    """
    B, L, H, d = q.shape

    if associative:
        # -------------------- Linear-time prefix-sum path --------------------
        # Row-wise stabilizer for s_i; safe under large contexts
        s_i_max = s_i.max(dim=1, keepdim=True)[0].detach()       # (B, 1, H, 1)
        s_i_stable = s_i - s_i_max
        exp_s_i = torch.exp(s_i_stable)                          # (B, L, H, 1)

        # Time index 't' for first-order terms (broadcasted)
        t = torch.arange(1, L + 1, device=q.device, dtype=q.dtype).view(1, L, 1, 1) if causal else L

        # Prefix sums for zero-sum softmax decomposition
        E_t = exp_s_i.cumsum(dim=1) if causal else exp_s_i.sum(dim=1, keepdim=True).expand(-1, L, -1, -1)  # ∑ e^{s_i}
        P_t = s_i.cumsum(dim=1) if causal else s_i.sum(dim=1, keepdim=True).expand(-1, L, -1, -1)          # ∑ s_i

        # Build per-step kv states (d×d per head)
        # kv[i] = k_i^T v_i  →   (B,L,H,d,d)
        kv = torch.einsum("blhd,blhe->blhde", k, v)

        F_t = (exp_s_i.unsqueeze(-1) * kv).cumsum(dim=1) if causal else (exp_s_i.unsqueeze(-1) * kv).sum(dim=1, keepdim=True).expand(-1, L, -1, -1, -1)
        G_t = (s_i.unsqueeze(-1) * kv).cumsum(dim=1) if causal else (s_i.unsqueeze(-1) * kv).sum(dim=1, keepdim=True).expand(-1, L, -1, -1, -1)
        H_t = kv.cumsum(dim=1) if causal else kv.sum(dim=1, keepdim=True).expand(-1, L, -1, -1, -1)

        # Gates σ^1 (first-order) and σ^h (higher orders)
        sigma_t_1 = torch.sigmoid(gate[..., [0]])  # (B, L, H, 1)
        sigma_t_h = torch.sigmoid(gate[..., [1]])  # (B, L, H, 1)

        # α, β, γ coefficients for reweighted zero-sum softmax pieces
        alpha_t = sigma_t_h / (E_t + eps)                           # Full / E_t
        beta_t = (sigma_t_1 - sigma_t_h) / t                        # 1st order term / t
        gamma_t = -((beta_t * P_t + sigma_t_h) / t)                 # remainder to keep zero-sum

        # Final read: q_t · [ α·F_t + β·G_t + γ·H_t ]
        core = alpha_t.unsqueeze(-1) * F_t + beta_t.unsqueeze(-1) * G_t + gamma_t.unsqueeze(-1) * H_t
        out = torch.einsum("blhd,blhde->blhe", q, core)             # (B, L, H, d)

        if is_first_layer:
            # Optional restore of 0th-order baseline on first layer only
            sigma_t_0 = torch.tanh(gate[..., [2]])                  # (B, L, H, 1)
            zero_order = torch.einsum("blhd,blhde->blhe", q / t, H_t)
            out = out + sigma_t_0 * zero_order

        return out

    # -------------------- Fallback (explicit weights) --------------------
    if causal:
        if mask is None:
            mask = _make_causal_mask(L=L, device=q.device, dtype=torch.float32)

    # Angular component (cos θ): dot(q_t, k_i), shape (B, H, L, L)
    cos_theta = torch.einsum("blhd,bihd->bhli", q, k)
    cos_theta = cos_theta.masked_fill(~mask, 0) if (mask is not None and causal) else cos_theta

    # Expand s_i to (B, H, L, L), apply masked softmax along last dim
    s_i_expand = s_i.permute(0, 2, 3, 1).expand(-1, -1, L, -1)     # (B, H, L, L)
    if causal and mask is not None:
        s_i_for_exp = s_i_expand.masked_fill(~mask, float("-inf"))
    else:
        s_i_for_exp = s_i_expand
    s_soft = F.softmax(s_i_for_exp, dim=-1)                         # "Full" softmax

    # Lower-triangular keep for 1st-order pieces
    s_tril = s_soft.masked_fill(~mask, 0) if (causal and mask is not None) else s_soft
    s_tril_sum = s_tril.sum(dim=-1, keepdim=True)

    # Zero-order, first-order and higher-order residual components
    t = torch.arange(1, L + 1, device=q.device, dtype=q.dtype).view(1, 1, L, 1) if causal else int(L)
    factor_zero_order = (1.0 / t)
    factor_first_order = (s_tril / t) - (s_tril_sum / (t ** 2))
    factor_remaining_orders = s_soft - factor_zero_order - factor_first_order

    if is_first_layer:
        r_t_i = (
            torch.sigmoid(gate[..., [0]]).transpose(1, 2) * factor_first_order +
            torch.sigmoid(gate[..., [1]]).transpose(1, 2) * factor_remaining_orders +
            torch.tanh(gate[..., [2]]).transpose(1, 2) * factor_zero_order
        )
    else:
        r_t_i = (
            torch.sigmoid(gate[..., [0]]).transpose(1, 2) * factor_first_order +
            torch.sigmoid(gate[..., [1]]).transpose(1, 2) * factor_remaining_orders
        )

    # Apply angular term and read values
    weight = (r_t_i * cos_theta).masked_fill(~mask, 0) if (mask is not None and causal) else (r_t_i * cos_theta)
    out = torch.einsum("bhli,bihd->blhd", weight, v)                # (B, L, H, d)
    return out


# -----------------------------------------------------------------------------
# ZeroS Attention layer
# -----------------------------------------------------------------------------
class ZeroSAttention(nn.Module):
    """
    Zero‑Sum Linear Attention (per-head) with:
      - learned per-step radial logits s_i (deviation-based by default),
      - optional RoPE on (Q, K),
      - associative (linear-time) or explicit (fallback) mixing,
      - per-head LayerNorm on output (optional),
      - standard output projection.

    Expected config (SimpleNamespace-like):
      n_embd, n_head, bias=True, dropout=0.0,
      block_size, is_causal=True,
      init_params=False, init_n_layers=12,
      use_norm=True, use_associative=True, is_first_layer=False
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.D = config.n_embd
        self.H = config.n_head
        self.d = self.D // self.H

        # Projections
        self.q = nn.Linear(self.D, self.D, bias=config.bias)
        self.k = nn.Linear(self.D, self.D, bias=config.bias)
        self.v = nn.Linear(self.D, self.D, bias=config.bias)
        self.u = nn.Linear(self.D, self.D, bias=config.bias)      # for s_i logits
        self.gate = nn.Linear(self.D, 3 * self.H, bias=config.bias)  # σ^1, σ^h, σ^0(optional)
        self.out_proj = nn.Linear(self.D, self.D, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

        # RoPE and causal mask buffers
        cos, sin = get_rotary_embedding(config.block_size, self.d)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        bias = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", bias.view(1, 1, config.block_size, config.block_size).bool())

        # Per-head LayerNorm on output (optional)
        self.norm = nn.LayerNorm(self.d, eps=1e-5, elementwise_affine=False)

        # Prior params for deviation logits
        self.prior_mu = nn.Parameter(torch.zeros(1, 1, self.H, self.d))
        self.prior_log_tau = nn.Parameter(torch.zeros(1, 1, self.H, 1))

        # Flags
        self.is_first_layer = getattr(config, "is_first_layer", False)
        self.is_causal = getattr(config, "is_causal", True)
        self.init_params = getattr(config, "init_params", False)
        self.use_norm = getattr(config, "use_norm", True)
        self.use_associative = getattr(config, "use_associative", True)

        # Optional GPT‑2 style init
        if self.init_params:
            self.apply(self._init_weights)
            self.n_layers = getattr(config, "init_n_layers", 12)
            for pn, p in self.named_parameters():
                if pn.endswith("out_proj.weight"):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

    # -- init helpers ---------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- forward --------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Project Q, K, V, and U (for s_i)
        q = self.q(x).view(B, L, self.H, self.d)
        k = self.k(x).view(B, L, self.H, self.d)
        v = self.v(x).view(B, L, self.H, self.d)
        u = self.u(x).view(B, L, self.H, self.d)

        # Per-step gates (σ^1, σ^h, σ^0)
        gate = self.gate(x).view(B, L, self.H, 3)

        # Radial logits s_i (deviation-based by default)
        s_i = self.calculate_logit(u, logit_type="deviation")  # (B, L, H, 1)

        # RoPE on (Q, K)
        q = apply_rotary_pos_emb(q, self.cos[:, :L], self.sin[:, :L])
        k = apply_rotary_pos_emb(k, self.cos[:, :L], self.sin[:, :L])

        # Unit-norm directions for angular term
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Core mixer
        out = compute_o(
            q, s_i, k, v, gate=gate,
            mask=self.bias[:, :, :L, :L],
            causal=self.is_causal,
            associative=self.use_associative,
            is_first_layer=self.is_first_layer,
        )  # (B, L, H, d)

        # Optional per-head LayerNorm
        if self.use_norm:
            out = self.norm(out)

        # Merge heads and project out
        out = out.reshape(B, L, D)
        out = self.dropout(self.out_proj(out))
        return out

    # -- logits ---------------------------------------------------------------
    def calculate_logit(self, u: torch.Tensor, logit_type: str = "deviation") -> torch.Tensor:
        """
        Produce per-index radial logits s_i: (B, L, H, 1).
        Options:
          - "deviation" (default): negative inner product against smoothed running mean
          - "quadratic": ||u||^2 / d
          - "mean":  <u, 1>/d
          - "distance_1": L1 distance to running mean / d
          - "distance_rms": RMS distance to running mean
        """
        _, L, _, _ = u.shape
        t = torch.arange(1, L + 1, device=u.device, dtype=u.dtype).view(1, L, 1, 1)

        if logit_type == "deviation":
            # Running mean with a learned smoothing prior (μ, τ)
            tau = torch.exp(self.prior_log_tau.clip(-50, 30))     # (1,1,H,1), positive
            u_sum = u.cumsum(dim=1) if self.is_causal else u.sum(dim=1, keepdim=True)
            bar_u_i = (tau * self.prior_mu + u_sum) / (tau + t)   # (B,L,H,d)
            s_i = - (u * bar_u_i).sum(dim=-1, keepdim=True) / math.sqrt(self.d)

        elif logit_type == "quadratic":
            s_i = (u * u).sum(dim=-1, keepdim=True) / self.d

        elif logit_type == "mean":
            s_i = u.sum(dim=-1, keepdim=True) / self.d

        elif logit_type == "distance_1":
            u_mean = u.cumsum(dim=1) / t
            s_i = torch.abs(u - u_mean).sum(dim=-1, keepdim=True) / self.d

        elif logit_type == "distance_rms":
            u_mean = u.cumsum(dim=1) / t
            s_i = torch.sqrt(torch.square(u - u_mean).sum(dim=-1, keepdim=True) / self.d + 1e-8)

        else:
            raise ValueError(f"Unknown logit_type: {logit_type}")

        return s_i


__all__ = ["ZeroSAttention", "compute_o", "get_rotary_embedding", "apply_rotary_pos_emb"]
