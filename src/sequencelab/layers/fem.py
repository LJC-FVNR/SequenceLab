# -----------------------------------------------------------------------------
# Free Energy Mixer (FEM): value-aware, per-channel mixing at the cost of
# the underlying prior (softmax / linear / GLA / AFT-like RNN / Mamba SSM).
#
# This is a compact, task-agnostic implementation that mirrors the original
# reference you provided, with:
#   - Fixed-parameter budget presets via FreeEnergyMixerConfig
#   - Backends auto-detection (FlashAttention / FLA kernels / Mamba SSM)
#   - Stable log-sum-exp branch with exact log restore (no numeric leakage)
#   - Linear-time scans for kernelizable priors
#   - Optional RoPE and a lightweight time-decay conditioning (TDC) module
#
# All paths return a layer with signature:
#   forward(x: Tensor[B, T, D], attention_mask: Optional[Tensor[B, T]]) -> Tensor[B, T, D]
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings
import math
from math import gcd

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Backend auto-detection (best-effort; safe fallbacks are provided)
# -----------------------------------------------------------------------------
_HAS_FLASH_ATTN = False
try:
    # flash-attn expects (B, T, H, D) and requires v_dim == qk_dim
    from flash_attn import flash_attn_func  # type: ignore
    _HAS_FLASH_ATTN = True
except Exception:
    _HAS_FLASH_ATTN = False

_HAS_FLA = False
_HAS_FLA_GLA = False
try:
    # FLA kernels accept v_dim != qk_dim
    from fla.ops.simple_gla import chunk_simple_gla  # type: ignore
    _HAS_FLA = True
    try:
        from fla.ops.gla import chunk_gla  # type: ignore
        _HAS_FLA_GLA = True
    except Exception:
        chunk_gla = None  # type: ignore
        _HAS_FLA_GLA = False
except Exception:
    chunk_simple_gla = None  # type: ignore
    chunk_gla = None  # type: ignore
    _HAS_FLA = False
    _HAS_FLA_GLA = False

_HAS_MAMBA = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # type: ignore
    _HAS_MAMBA = True
except Exception:
    _HAS_MAMBA = False


# -----------------------------------------------------------------------------
# Config (paper-aligned knobs, but kept minimal)
# -----------------------------------------------------------------------------
@dataclass
class FreeEnergyMixerConfig:
    # Core dims
    n_embd: int
    n_head: int
    fem_dim: Optional[int] = None  # defaults to n_embd * fem_ratio if None
    fem_ratio: float = 0.5

    # Per-token Q/K dim (if None, derived from p_t_to_fem_ratio * fem_dim)
    qk_dim: Optional[int] = None
    p_t_to_fem_ratio: float = 4.0  # total (|Q|+|K|) relative to fem_dim

    # Priors: softmax | linear | gla | rnn_softmax | mamba
    prior_type: str = "softmax"

    # Softmax backend choice
    softmax_backend: str = "auto"  # "auto" | "flash" | "naive"

    # Runtime
    dropout: float = 0.0
    causal: bool = True
    bias: bool = True

    # FEM knobs
    beta_max_init: float = 1.8
    value_logexp_cap: float = 40.0  # constrain |β·v| via pre-clip
    eps: float = 1e-6

    # Feature flags
    use_temperature: bool = True  # enable β and exp branch
    use_lse: bool = True          # keep LSE (β=1) when no temperature
    use_outer_gate: bool = True   # multiplicative outer gate
    use_rope: bool = True         # RoPE on Q/K
    rope_theta: float = 10000.0

    # Lightweight conv (time-decay conditioned) to modulate params
    use_conv: bool = True
    conv_hidden: int = 64
    conv_norm_first: bool = True
    conv_bidirectional: bool = False

    # Mamba SSM
    ssm_rank: int = 16
    mamba_normalize: bool = True


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _split_heads(x: torch.Tensor, n_head: int) -> torch.Tensor:
    B, T, D = x.shape
    assert D % n_head == 0, "hidden size must be divisible by n_head"
    d = D // n_head
    return x.view(B, T, n_head, d)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, T, H, d = x.shape
    return x.reshape(B, T, H * d)


def _expand_per_channel(vec: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """vec: (E,) -> (1,1,H,d_v) to match 'like' with shape (B,T,H,d_v)."""
    _, _, H, d = like.shape
    assert vec.numel() == H * d
    return vec.view(1, 1, H, d)


def _phi(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Nonnegative feature map for linear attention."""
    return F.relu(x) + eps


def _softplus_pos(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.softplus(x) + eps


def _lcm(a: int, b: int) -> int:
    a = abs(int(a)); b = abs(int(b))
    if a == 0 or b == 0:
        return 0
    return a // gcd(a, b) * b


def _round_channels(
    x: int,
    multiple: int,
    n_head: int,
    per_head_unit: int = 2,
    mode: str = "nearest",   # "nearest" | "up" | "down"
) -> int:
    """
    Round x to a value that:
      1) is divisible by `multiple` (if multiple<=0, treated as 1)
      2) is divisible by `n_head * per_head_unit`
    """
    n_head = max(int(n_head), 1)
    m = 1 if multiple is None or multiple <= 0 else int(multiple)
    base_req = n_head * max(int(per_head_unit), 1)
    base = _lcm(m, base_req)
    if base <= 0:
        base = base_req

    if mode == "up":
        k = max(1, (x + base - 1) // base)
        return k * base
    elif mode == "down":
        k = max(1, x // base)
        return k * base
    else:  # nearest
        q = int(round(x / base))
        q = max(q, 1)
        cand = q * base
        low = (q - 1) * base if q > 1 else base
        high = (q + 1) * base
        if abs(cand - x) < abs(low - x) and abs(cand - x) <= abs(high - x):
            return cand
        return high if abs(high - x) <= abs(low - x) else low


# -----------------------------------------------------------------------------
# RoPE (simple, tensor-only; no external deps)
# -----------------------------------------------------------------------------
class SimpleRoPE(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k: (B,T,H,d)
        B, T, H, d = q.shape
        dev = q.device
        t = torch.arange(T, device=dev, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,f->tf", t, self.inv_freq)  # (T, d/2)
        emb = torch.cat((freqs, freqs), dim=-1)            # (T, d)

        cos = emb.cos()[None, :, None, :]  # (1, T, 1, d)
        sin = emb.sin()[None, :, None, :]

        def rot(x):
            x1, x2 = x[..., ::2], x[..., 1::2]
            xr = torch.stack((-x2, x1), dim=-1).reshape_as(x)
            return xr

        q_out = q * cos + rot(q) * sin
        k_out = k * cos + rot(k) * sin
        return q_out, k_out


# -----------------------------------------------------------------------------
# TDC (Time-Decay Conditioning): fully-parallel lightweight conv-like feature
# -----------------------------------------------------------------------------
class TimeDecayConditionLayer(nn.Module):
    """
    Closed-form time-decay accumulation (parallel):
      a = -softplus(W_in(x)), c = cumsum(a), D = exp(c)
      out = D * cumsum( (x_f / D) )
    The result gates a shortcut and is linearly projected.
    """
    def __init__(self, config: FreeEnergyMixerConfig, hidden_dim=64, norm_first=True, output_dim=None):
        super().__init__()
        self.norm_first = norm_first
        self.norm = nn.LayerNorm(config.n_embd, bias=config.bias) if norm_first else nn.Identity()
        self.in_f = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.x_f = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.shortcut = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        out_dim = config.n_embd if output_dim is None else output_dim
        self.out_proj = nn.Parameter(torch.zeros(hidden_dim, out_dim))
        self.ln = nn.LayerNorm(hidden_dim, bias=config.bias)

    def forward(self, x: torch.Tensor, gated_output: bool = True) -> torch.Tensor:
        x_ln = self.norm(x)
        sc = self.shortcut(x_ln)
        sc = F.softplus(sc) + 1e-8
        sc = F.normalize(sc, dim=-1, eps=1e-8) * math.sqrt(sc.size(-1))

        xf = self.x_f(x_ln)
        a = -F.softplus(self.in_f(x_ln))
        c = a.cumsum(dim=1)
        D = torch.exp(c)
        y = (xf / (D + 1e-12)).cumsum(dim=1) * D
        y = sc * self.ln(y)
        y = y @ self.out_proj
        return x * y if gated_output else y


class TimeDecayCondition(nn.Module):
    def __init__(self, config: FreeEnergyMixerConfig, hidden_dim=64, norm_first=True, bidirectional=False, output_dim=None):
        super().__init__()
        self.bidirectional = bidirectional
        if bidirectional:
            self.fwd = TimeDecayConditionLayer(config, hidden_dim // 2, norm_first, output_dim)
            self.bwd = TimeDecayConditionLayer(config, hidden_dim // 2, norm_first, output_dim)
        else:
            self.fwd = TimeDecayConditionLayer(config, hidden_dim, norm_first, output_dim)
            self.bwd = None

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.bidirectional:
            return self.fwd(x, *args, **kwargs)
        return self.fwd(x, *args, **kwargs) + self.bwd(x, *args, **kwargs)


# -----------------------------------------------------------------------------
# Stable exp branch utilities (exact log-restore; no M leakage)
# -----------------------------------------------------------------------------
def stable_log_restore(
    out: torch.Tensor,          # (B,T,H,d_v) = Σ_i α_{t,i} exp(v_i - M)
    v_max: torch.Tensor,        # (B,1,H,d_v)
    eps: Optional[float] = None,
    detach_M: bool = True,
    check_finite: bool = False,
    sanitize: bool = False
) -> torch.Tensor:
    """
    log( Σ_i α_{t,i} exp(v_i) + eps_t )
    = logaddexp( log(out_t), log(eps_t) - M ) + M
    where out_t = Σ_i α_{t,i} exp(v_i - M),  M = max_i v_i (per (B,H,d_v), shared across t).
    This is algebraically exact and does not leak M.
    """
    work_dtype = out.dtype if out.dtype in (torch.float32, torch.float64) else torch.float32
    if eps is None:
        eps = torch.finfo(work_dtype).tiny
    if not torch.is_tensor(eps):
        eps = torch.tensor(eps, dtype=work_dtype, device=out.device)

    outw = out.to(work_dtype)
    Mw   = v_max.to(work_dtype)
    if detach_M:
        Mw = Mw.detach()

    if check_finite:
        if not torch.isfinite(outw).all() or not torch.isfinite(Mw).all():
            if sanitize:
                outw = torch.nan_to_num(outw, nan=0.0, posinf=0.0, neginf=0.0)
                Mw   = torch.nan_to_num(Mw,   nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raise FloatingPointError("non-finite inputs to stable_log_restore")

    # log(out): allow zeros → -inf, avoid NaN
    outw = torch.clamp_min(outw, 0.0)
    log_out = torch.where(outw > 0.0, torch.log(outw), torch.full_like(outw, float("-inf")))

    safe = torch.logaddexp(log_out, torch.log(eps.to(work_dtype)) - Mw) + Mw
    return safe.to(out.dtype)


# -----------------------------------------------------------------------------
# Priors: softmax / linear / GLA / AFT-like RNN / Mamba
# -----------------------------------------------------------------------------
def _softmax_read_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       causal: bool, dropout_p: float, training: bool) -> torch.Tensor:
    # torch SDPA: expects (B,H,T,D) -> we pass (B,T,H,D) and transpose around
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=None,
        dropout_p=dropout_p if training else 0.0,
        is_causal=causal,
    ).transpose(1, 2)


def _softmax_read_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        causal: bool, dropout_p: float, training: bool) -> torch.Tensor:
    if not (_HAS_FLASH_ATTN and q.is_cuda and k.is_cuda and v.is_cuda):
        return _softmax_read_sdpa(q, k, v, causal, dropout_p, training)
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    return flash_attn_func(q, k, v, dropout_p=dropout_p if training else 0.0, causal=causal)


def _linear_read_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float, causal: bool) -> torch.Tensor:
    qn = _phi(q, eps)
    kn = _phi(k, eps)
    if causal:
        kv = torch.einsum("bthd,bthe->bthde", kn, v)
        S = kv.cumsum(dim=1)
        Ksum = kn.cumsum(dim=1)
        num = torch.einsum("bthd,bthde->bthe", qn, S)
        den = (qn * Ksum).sum(dim=-1, keepdim=True).clamp_min(eps)
        return num / den
    else:
        S_all = torch.einsum("bthd,bthe->bhde", kn, v)
        K_all = kn.sum(dim=1)
        num = torch.einsum("bthd,bhde->bthe", qn, S_all)
        den = torch.einsum("bthd,bhd->bth", qn, K_all).unsqueeze(-1).clamp_min(eps)
        return num / den


def _linear_read_fla(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
    if not _HAS_FLA or chunk_simple_gla is None:
        warnings.warn("[FEM][linear] FLA not found; using naive.", stacklevel=1)
        return _linear_read_naive(q, k, v, eps=eps, causal=True)
    q = _phi(q, eps)
    k = _phi(k, eps)
    v_aug = torch.cat([v, torch.ones_like(v[:, :, :, :1])], dim=-1)
    out_aug, _ = chunk_simple_gla(q=q, k=k, v=v_aug, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None)
    num, den = out_aug[:, :, :, :-1], out_aug[:, :, :, -1:].clamp_min(eps)
    return num / den


def _gla_read_naive(x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    W_decay: nn.Linear, eps: float) -> torch.Tensor:
    qn = _phi(q, eps)
    kn = _phi(k, eps)

    a = -_phi(W_decay(x), eps)  # (B,T,H)
    c = a.cumsum(dim=1).clip(-60, 50)
    D = torch.exp(c)
    invD = torch.exp(-c)

    kv = torch.einsum("bthd,bthe->bthde", kn, v)
    S = (kv * invD.unsqueeze(-1).unsqueeze(-1)).cumsum(dim=1) * D.unsqueeze(-1).unsqueeze(-1)
    Ksum = (kn * invD.unsqueeze(-1)).cumsum(dim=1) * D.unsqueeze(-1)

    num = torch.einsum("bthd,bthde->bthe", qn, S)
    den = (qn * Ksum).sum(dim=-1, keepdim=True) + 1e-5
    return num / den


def _gla_read_fla(x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  W_decay: nn.Linear, eps: float) -> torch.Tensor:
    if not _HAS_FLA_GLA or chunk_gla is None:
        warnings.warn("[FEM][gla] FLA GLA not found; using naive.", stacklevel=1)
        return _gla_read_naive(x, q, k, v, W_decay, eps)
    q = _phi(q, eps)
    k = _phi(k, eps)
    g = -_phi(W_decay(x), eps).unsqueeze(-1).expand(-1, -1, -1, k.size(-1))
    v_aug = torch.cat([v, torch.ones_like(v[:, :, :, :1])], dim=-1)
    out_aug, _ = chunk_gla(q=q, k=k, v=v_aug, g=g, initial_state=None, output_final_state=False, cu_seqlens=None)
    num, den = out_aug[:, :, :, :-1], out_aug[:, :, :, -1:] + 1e-5
    return num / den


def _aft_read_expand(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
    """AFT-like linear RNN with expand trick: ensures v_dim multiple of qk_dim."""
    B, T, H, d = q.shape
    E = v.shape[-1]
    assert E % d == 0, f"V dim {E} must be multiple of qk dim {d} for expand_qk"
    r = E // d
    q_rep = q.repeat(1, 1, 1, r)
    k_rep = k.repeat(1, 1, 1, r)

    q_pos = _softplus_pos(q_rep, eps)
    q_pos = F.normalize(q_pos, dim=-1, eps=1e-5) * math.sqrt(q_pos.size(-1))

    # time-wise -max trick for k (safe for AFT)
    k_max = k_rep.max(dim=1, keepdim=True).values
    k_exp = torch.exp(k_rep - k_max) + eps

    kv_cum = (k_exp * v).cumsum(dim=1)
    k_cum = k_exp.cumsum(dim=1).clamp_min(eps)
    return q_pos * (kv_cum / k_cum)


# ----------------- Mamba SSM (efficient) + EMA fallback ----------------------
def _ssm_mamba_scan(u: torch.Tensor, u_exp: Optional[torch.Tensor],
                    delta: torch.Tensor, A_log: torch.Tensor,
                    B_t: torch.Tensor, C_t: torch.Tensor,
                    D_diag: torch.Tensor, normalize: bool, eps: float) -> torch.Tensor:
    dtype = u.dtype
    A = (-F.softplus(A_log)).float()
    Dp = _softplus_pos(D_diag, eps).float()

    out = selective_scan_fn(
        u=u.transpose(1, 2),
        delta=delta.transpose(1, 2),
        A=A, B=B_t.transpose(1, 2), C=C_t.transpose(1, 2), D=Dp,
    ).to(dtype).transpose(1, 2)
    if normalize:
        ones = torch.ones_like(u, dtype=u.dtype)
        norm = selective_scan_fn(
            u=ones.transpose(1, 2),
            delta=delta.transpose(1, 2),
            A=A, B=B_t.transpose(1, 2), C=C_t.transpose(1, 2), D=Dp,
        ).to(dtype).transpose(1, 2)
        out = out / norm.clamp_min(eps)

    if u_exp is not None:
        out_exp = selective_scan_fn(
            u=u_exp.transpose(1, 2),
            delta=delta.transpose(1, 2),
            A=A, B=B_t.transpose(1, 2), C=C_t.transpose(1, 2), D=Dp,
        ).to(dtype).transpose(1, 2)
        if normalize:
            out_exp = out_exp / norm.clamp_min(eps)
        out = torch.cat([out, out_exp], dim=-1)
    return out


def _mamba_read(x: torch.Tensor, V: torch.Tensor, V_exp: Optional[torch.Tensor],
                W_p: nn.Linear, A_log: torch.Tensor, D_diag: torch.Tensor,
                rank: int, eps: float, normalize: bool) -> torch.Tensor:
    B, T, H, d_v = V.shape
    E = H * d_v
    u = V.reshape(B, T, E)
    u_dtype = u.dtype
    u_exp_flat = None if V_exp is None else V_exp.reshape(B, T, E).to(u_dtype)

    p = W_p(x).reshape(B, T, -1)  # (B,T,E+2R)
    delta = torch.exp(-F.softplus(p[:, :, :E]))
    Br = _softplus_pos(p[:, :, E:E+rank], eps)
    Cr = _softplus_pos(p[:, :, E+rank:E+2*rank], eps)

    out_flat = _ssm_mamba_scan(u, u_exp_flat, delta.to(u_dtype), A_log.to(u_dtype), Br.to(u_dtype), Cr.to(u_dtype), D_diag.to(u_dtype), normalize, eps)
    if out_flat.shape[-1] == E:
        return out_flat.reshape(B, T, H, d_v)
    return out_flat.reshape(B, T, 2, H, d_v).transpose(2, 3).reshape(B, T, H, 2 * d_v)


def _mamba_read_ema(x: torch.Tensor, V: torch.Tensor, eps: float, W_delta: nn.Linear) -> torch.Tensor:
    B, T, H, E = V.shape
    d = torch.sigmoid(W_delta(x)).unsqueeze(-1)  # (B,T,H,1)
    S = V.new_zeros(B, H, E)
    N = V.new_zeros(B, H, 1)
    outs = []
    for t in range(T):
        dt = d[:, t]
        vt = V[:, t]
        S = dt * S + vt
        N = dt * N + 1.0
        outs.append((S / (N + eps)).unsqueeze(1))
    return torch.cat(outs, dim=1)


# -----------------------------------------------------------------------------
# FEM module
# -----------------------------------------------------------------------------
class FreeEnergyMixer(nn.Module):
    """
    Free Energy Mixer:
      - Takes a prior selection (softmax / linear / GLA / AFT-RNN / Mamba)
      - Reads two branches per channel: μ (expectation) and F_max (log-sum-exp at β_max)
      - Learns per-channel λ and outer gate g, combining:
            out = g ⊙ [ (1-λ) ⊙ μ + λ ⊙ F_max ]
      - Preserves the asymptotic complexity of the chosen prior
    """

    def __init__(self, cfg: FreeEnergyMixerConfig):
        super().__init__()
        self.cfg = cfg

        D = cfg.n_embd
        H = cfg.n_head

        # -------- fem_dim (E on value path) --------
        if cfg.fem_dim is None:
            # target by ratio; ensure per-head dim divisible by 4 (for RoPE-friendly packing)
            E_target = int(round(cfg.fem_ratio * D))
            E = _round_channels(
                x=E_target,
                multiple=getattr(cfg, "align_multiple", 1),
                n_head=H,
                per_head_unit=4,
                mode="nearest",
            )
        else:
            E = _round_channels(
                x=int(cfg.fem_dim),
                multiple=getattr(cfg, "align_multiple", 1),
                n_head=H,
                per_head_unit=4,
                mode="nearest",
            )
        assert E % H == 0, "fem_dim must be divisible by n_head"

        self.D, self.H, self.E = D, H, E
        self.d_v = E // H

        pt = cfg.prior_type.lower()

        # -------- qk_dim (for attention-like priors) --------
        if pt in {"softmax", "linear", "gla", "rnn_softmax"}:
            if getattr(cfg, "qk_dim", None) is not None:
                qk_total = _round_channels(
                    x=int(cfg.qk_dim),
                    multiple=getattr(cfg, "align_multiple", 1),
                    n_head=H,
                    per_head_unit=2,
                    mode="nearest",
                )
            else:
                qk_target = int(round(0.5 * cfg.p_t_to_fem_ratio * E))
                qk_target = max(qk_target, 2 * H)
                qk_total = _round_channels(
                    x=qk_target,
                    multiple=getattr(cfg, "align_multiple", 1),
                    n_head=H,
                    per_head_unit=2,
                    mode="nearest",
                )
            assert qk_total % H == 0
            self.d_k = qk_total // H
            assert (self.d_k % 2) == 0, "per-head qk dim must be even for RoPE"
        else:
            self.d_k = 0  # mamba-only branch doesn't use Q/K

        if pt == "rnn_softmax" and self.d_k > 0:
            if (self.d_v % self.d_k) != 0:
                new_dk = None
                for c in range(self.d_v, 1, -1):
                    if (c % 2 == 0) and (self.d_v % c == 0):
                        new_dk = c
                        break
                if new_dk is None:
                    new_dk = self.d_v if (self.d_v % 2 == 0) else max(2, self.d_v - 1)
                self.d_k = new_dk

        self.eps = float(cfg.eps)
        self.causal = bool(cfg.causal)
        self.drop_attn = float(cfg.dropout)
        self.drop_val = nn.Dropout(cfg.dropout)
        self.prior_type = cfg.prior_type.lower()
        self.softmax_backend = cfg.softmax_backend.lower()

        # Value / out projections
        self.W_v = nn.Linear(D, E, bias=cfg.bias)
        self.W_o = nn.Linear(E, D, bias=cfg.bias)

        # Q/K only for attention-like priors
        if self.prior_type in {"softmax", "linear", "gla", "rnn_softmax"}:
            self.W_q = nn.Linear(D, H * self.d_k, bias=cfg.bias)
            self.W_k = nn.Linear(D, H * self.d_k, bias=cfg.bias)
        else:
            self.W_q = self.W_k = None

        # GLA decay
        self.W_decay = nn.Linear(D, H, bias=cfg.bias) if self.prior_type == "gla" else None

        # Mamba params
        self.ssm_rank = int(cfg.ssm_rank)
        if self.prior_type == "mamba":
            if _HAS_MAMBA:
                self.W_p = nn.Linear(D, E + 2 * self.ssm_rank, bias=cfg.bias)
                self.A_log = nn.Parameter(-torch.ones(E, self.ssm_rank))
                self.D_diag = nn.Parameter(torch.ones(E))
            else:
                self.W_delta = nn.Linear(D, H, bias=cfg.bias)

        # FEM gates
        self.W_lambda = nn.Linear(D, E, bias=cfg.bias) if cfg.use_temperature or cfg.use_lse else None
        self.W_gate = nn.Linear(D, E, bias=cfg.bias) if cfg.use_outer_gate else None

        # β_max (per-channel, shared over time)
        self._beta_param = nn.Parameter(torch.full((E,), float(cfg.beta_max_init)))

        # RoPE uses d_k (may differ from D/H)
        self.rope = SimpleRoPE(dim=self.d_k, base=cfg.rope_theta) if cfg.use_rope and (self.W_q is not None) else None

        # TDC modulation output size must reflect new d_k
        self.tdc = None
        if cfg.use_conv:
            extra = (E if (cfg.use_temperature or cfg.use_lse) else 0) \
                    + (E if cfg.use_outer_gate else 0) \
                    + (H if self.prior_type == "gla" else 0)
            # scales for: V (E) + Q/K (2 * H * d_k) + extras
            tdc_out = E + 2 * H * self.d_k + extra
            self.tdc = TimeDecayCondition(cfg, hidden_dim=cfg.conv_hidden,
                                          norm_first=cfg.conv_norm_first,
                                          bidirectional=cfg.conv_bidirectional,
                                          output_dim=tdc_out)

        self.reset_parameters()

        # Backend logs
        msg = []
        if _HAS_FLASH_ATTN: msg.append("flash_attn")
        if _HAS_FLA: msg.append("fla")
        if _HAS_MAMBA: msg.append("mamba_ssm")
        if msg:
            print("[FEM] Detected backends:", ", ".join(msg))

    # -- init -----------------------------------------------------------------
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -- combine branches -----------------------------------------------------
    def _combine_free_energy(self, mu: torch.Tensor, fmax: Optional[torch.Tensor],
                             lamb: Optional[torch.Tensor], gate: Optional[torch.Tensor]) -> torch.Tensor:
        if fmax is None or lamb is None:
            out = mu
        else:
            out = (1.0 - lamb) * mu + lamb * fmax
        if gate is not None:
            out = gate * out
        return out

    # -- forward --------------------------------------------------------------
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.D
        pt = self.prior_type

        # Projections
        V = _split_heads(self.W_v(x), self.H)  # (B,T,H,d_v)
        Q = K = None
        if self.W_q is not None:
            Q = _split_heads(self.W_q(x), self.H)
            K = _split_heads(self.W_k(x), self.H)

        # Mask pads (1/True=keep, 0/False=pad)
        if attention_mask is not None:
            m = attention_mask.to(x.dtype).unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)
            if K is not None: K = K * m
            V = V * m

        # TDC modulation (multiplicative 1 + conv)
        if self.tdc is not None:
            tdc_out = self.tdc(x, gated_output=False)
            idx = 0
            scale_v = tdc_out[:, :, idx:idx + self.E]; idx += self.E
            V = V * (1 + _split_heads(scale_v, self.H))
            s_qk = tdc_out[:, :, idx:idx + 2 * self.H * self.d_k]; idx += 2 * self.H * self.d_k
            if Q is not None:
                Q = Q * (1 + _split_heads(s_qk[:, :, :self.H * self.d_k], self.H))
                K = K * (1 + _split_heads(s_qk[:, :, self.H * self.d_k:], self.H))
            s_lam = tdc_out[:, :, idx:idx + self.E] if (self.cfg.use_temperature or self.cfg.use_lse) else None
            if s_lam is not None:
                idx += self.E
            s_gate = tdc_out[:, :, idx:idx + self.E] if self.cfg.use_outer_gate else None
            if s_gate is not None:
                idx += self.E
            if self.prior_type == "gla":
                _ = tdc_out[:, :, idx:idx + self.H]  # decay hint (ignored here; W_decay is learned)
        else:
            s_lam = s_gate = None

        # RoPE
        if self.rope is not None and (Q is not None):
            Q, K = self.rope(Q, K)

        # Gates
        lamb = None
        if self.cfg.use_temperature or self.cfg.use_lse:
            lamb = _split_heads(self.W_lambda(x), self.H)
            if s_lam is not None:
                lamb = lamb * (1 + _split_heads(s_lam, self.H))
            lamb = torch.sigmoid(lamb)

        gate = None
        if self.cfg.use_outer_gate:
            gate = _split_heads(self.W_gate(x), self.H)
            if s_gate is not None:
                gate = gate * (1 + _split_heads(s_gate, self.H))
            gate = F.softplus(gate) + self.eps
            gate = F.normalize(gate, dim=-1, eps=1e-5) * math.sqrt(gate.size(-1))

        # Mamba: optional squash helps stability
        if pt == "mamba":
            V = torch.sigmoid(V)

        # ---------------- LSE/temperature preparation (stable) ----------------
        use_branch = (self.cfg.use_temperature or self.cfg.use_lse)
        beta_max: Optional[torch.Tensor] = None
        V_exp: Optional[torch.Tensor] = None
        v_max_shift: Optional[torch.Tensor] = None
        V_val = V

        if use_branch:
            if self.cfg.use_temperature:
                # β learned; pre-clip on V s.t. β·V ∈ [-cap, cap]
                beta_vec = _softplus_pos(self._beta_param, self.eps)           # (E,)
                beta_max = _expand_per_channel(beta_vec, V)                    # (1,1,H,d_v)

                # clip V by elementwise limit cap/β
                v_max_limit = (self.cfg.value_logexp_cap / beta_max).to(V.dtype)  # (1,1,H,d_v)
                V_val = torch.maximum(torch.minimum(V, v_max_limit), -v_max_limit)

                v_temp = beta_max * V_val                                        # (B,T,H,d_v)
                v_max_shift = v_temp.max(dim=1, keepdim=True).values.detach()    # (B,1,H,d_v)
                V_exp = torch.exp(v_temp - v_max_shift)                          # (B,T,H,d_v)
            else:
                # LSE only (β=1). No pre-clip on V.
                beta_max = V.new_ones(1, 1, self.H, self.d_v)
                v_max_shift = V.max(dim=1, keepdim=True).values.detach()         # (B,1,H,d_v)
                V_exp = torch.exp(V - v_max_shift)                               # (B,T,H,d_v)

        # ---------------- prior read ----------------
        mu: torch.Tensor
        fmax: Optional[torch.Tensor] = None

        if pt == "softmax":
            if use_branch:
                V_in = torch.cat([V_val, V_exp], dim=-1)
                out_cat = _softmax_read_sdpa(Q, K, V_in, causal=self.causal, dropout_p=self.drop_attn, training=self.training)
                mu, z_exp_scaled = torch.split(out_cat, [self.d_v, self.d_v], dim=-1)
                fmax = stable_log_restore(z_exp_scaled, v_max_shift, detach_M=True) / beta_max
            else:
                if self.softmax_backend == "flash" and _HAS_FLASH_ATTN and (self.d_v == self.d_k):
                    mu = _softmax_read_flash(Q, K, V, causal=self.causal, dropout_p=self.drop_attn, training=self.training)
                else:
                    mu = _softmax_read_sdpa(Q, K, V, causal=self.causal, dropout_p=self.drop_attn, training=self.training)

        elif pt == "linear":
            if use_branch:
                V_in = torch.cat([V_val, V_exp], dim=-1)
                if _HAS_FLA and chunk_simple_gla is not None:
                    out_cat = _linear_read_fla(Q, K, V_in, eps=self.eps)
                else:
                    out_cat = _linear_read_naive(Q, K, V_in, eps=self.eps, causal=self.causal)
                mu, z_exp_scaled = torch.split(out_cat, [self.d_v, self.d_v], dim=-1)
                fmax = stable_log_restore(z_exp_scaled, v_max_shift, detach_M=True) / beta_max
            else:
                mu = _linear_read_fla(Q, K, V, eps=self.eps) if _HAS_FLA else _linear_read_naive(Q, K, V, eps=self.eps, causal=self.causal)

        elif pt == "gla":
            if self.W_decay is None:
                raise RuntimeError("GLA prior requires W_decay")
            if use_branch:
                V_in = torch.cat([V_val, V_exp], dim=-1)
                if _HAS_FLA_GLA and chunk_gla is not None:
                    out_cat = _gla_read_fla(x, Q, K, V_in, self.W_decay, eps=self.eps)
                else:
                    out_cat = _gla_read_naive(x, Q, K, V_in, self.W_decay, eps=self.eps)
                mu, z_exp_scaled = torch.split(out_cat, [self.d_v, self.d_v], dim=-1)
                fmax = stable_log_restore(z_exp_scaled, v_max_shift, detach_M=True) / beta_max
            else:
                mu = _gla_read_fla(x, Q, K, V, self.W_decay, eps=self.eps) if _HAS_FLA_GLA else _gla_read_naive(x, Q, K, V, self.W_decay, eps=self.eps)

        elif pt == "rnn_softmax":
            if use_branch:
                V_in = torch.cat([V_val, V_exp], dim=-1)
                out_cat = _aft_read_expand(Q, K, V_in, eps=self.eps)
                mu, z_exp_scaled = torch.split(out_cat, [self.d_v, self.d_v], dim=-1)
                fmax = stable_log_restore(z_exp_scaled, v_max_shift, detach_M=True) / beta_max
            else:
                mu = _aft_read_expand(Q, K, V, eps=self.eps)

        elif pt == "mamba":
            if _HAS_MAMBA:
                if use_branch:
                    out_cat = _mamba_read(x, V_val, V_exp, self.W_p, self.A_log, self.D_diag, self.ssm_rank, self.eps, self.cfg.mamba_normalize)
                    mu, z_exp_scaled = torch.split(out_cat, [self.d_v, self.d_v], dim=-1)
                    fmax = stable_log_restore(z_exp_scaled, v_max_shift, detach_M=True) / beta_max
                else:
                    mu = _mamba_read(x, V, None, self.W_p, self.A_log, self.D_diag, self.ssm_rank, self.eps, self.cfg.mamba_normalize)
            else:
                mu = _mamba_read_ema(x, V, self.eps, self.W_delta if hasattr(self, "W_delta") else nn.Linear(self.D, self.H, bias=True))

        else:
            raise ValueError(f"Unknown prior_type: {pt}")

        # Combine branches and project
        out_h = self._combine_free_energy(mu, fmax, lamb, gate)     # (B,T,H,d_v)
        out = self.W_o(self.drop_val(_merge_heads(out_h)))          # (B,T,D)
        return out


__all__ = ["FreeEnergyMixer", "FreeEnergyMixerConfig"]
