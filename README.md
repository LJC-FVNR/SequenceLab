# SequenceLab — Improved Attention & Temporal Mixers for Better Sequence Modeling

**SequenceLab** is an official research-code hub for our research team. It stores and publishes the implementations accompanying our published papers on improving sequence modeling.  

More model architectures and implementations will be continuously added to this repository.  This repository leverages LLMs for optimized code structure and maintainability.

> **Status:** SequenceLab is currently in its prototype phase — the codebase and APIs may change as we continue to refine and expand the project. 

---

## Implemented Methods

> Minimal notation: $w_{t,i}$ (weights), $v_i$ (values), $q_t,k_i$ (query/key), $p_t$ (prior), $\odot$ (Hadamard).

| Method | Core formula (concise) | Category | Paper / Code |
|---|---|---|---|
| **WAVE** (Weighted Autoregressive Varying Gate; ARMA-Attention) | $o_t = o_t^{\mathrm{AR}} + o_t^{\mathrm{MA}}, \, o_t^{\mathrm{AR}}=\sum_{i\le t} w_{t,i} v_i$ <br> $o_t^{\mathrm{MA}}=\sum_{j\le t-1} \beta_{t-1,j} r_j, r_j = v_{j+1} - o_j^{\mathrm{AR}}, \, \beta_{t-1,j}=\phi_q^{\mathrm{MA}}(q_{t-1})\,\phi_k^{\mathrm{MA}}(k_j^{\mathrm{MA}})$ | AR decoder-only with ARMA add-on (keeps complexity if base is linear) | [Paper](https://openreview.net/forum?id=Qqn5ktBUxH) / [Repo](https://github.com/LJC-FVNR/ARMA-Attention) |
| **ZeroS** (Zero-Sum Linear Attention) | $w_{t,i}=\sigma_t^1\frac{\delta_{t,i}}{t}+\sigma_t^h\varepsilon_{t,i}$ with $\delta_{t,i}=s_i-\bar s_t$, $\varepsilon_{t,i}=\text{softmax}(s_i)-\tfrac1t-\tfrac{\delta_{t,i}}{t}$; $o_t=\sum_{i\le t} w_{t,i}\,\cos\theta\, v_i,\ \cos\theta=\hat q_t\hat k_i^\top$ | **Linear-time** zero-sum attention; contrastive, signed weights | [Paper](https://openreview.net/forum?id=Ms6IXbfzzX) / [Repo](https://github.com/LJC-FVNR/ZeroS) |
| **FEM** (Free Energy Mixer) | Per-channel free-energy read on a prior $p_t$: $o_{t,j}=g_{t,j}\left[(1-\lambda_{t,j})\sum\nolimits_i p_t(i)v_{i,j}+\lambda_{t,j}\frac{1}{\beta_{\max}}\log\sum\nolimits_i p_t(i)e^{\beta_{\max}v_{i,j}}\right]$ | **Value-aware** posterior read; plug-and-play on softmax/linear/SSM priors; preserves prior complexity |  [Repo](https://github.com/LJC-FVNR/FreeEnergyMixer) |

> More methods and experimental architectures are under development and will be added soon.

**Notes**

- **WAVE** adds an MA branch via *indirect* weight generation; compute AR first, form residuals $r_j$, then apply a linear-time MA read (same order as the base attention).
- **ZeroS** removes the softmax zero-order $1/t$ term and reweights residuals with gates; implemented via prefix scans for $O(N)$ time; optional angular term $\cos\theta$ restores directional effects (nice with RoPE).
- **FEM** is a readout on top of a selection prior $p_t$ (softmax, gated linear, RNN/SSM). It adds one log-sum-exp per channel (with linearized temperature learning), keeping the prior’s asymptotic complexity.

---

## Install

> **Requirements**
>
> * Python ≥ 3.9
> * PyTorch ≥ 2.0 (uses `scaled_dot_product_attention`)

**1) Install PyTorch first** (choose the command for your platform from pytorch.org):

**2) Install `sequencelab` (dev / editable):**

```bash
git clone https://github.com/LJC-FVNR/SequenceLab.git
cd SequenceLab
pip install -e .
```

**3) (Optional) Acceleration backends**

The FEM layer auto-detects these if present; all are optional and CPU-safe fallbacks are provided.

```bash
# FlashAttention (Linux + CUDA; requires matching CUDA/toolchain)
pip install flash-attn --no-build-isolation

# FLA kernels (Gated Linear Attention)
pip install flash-linear-attention

# Mamba-SSM
pip install mamba-ssm
```

**4) Run tests (sanity checks)**

```bash
pytest -q
```

---

## Quick start

Below are minimal, end‑to‑end snippets. The convention is **input `x`: `[B, T, D]` → output `y`: `[B, T, D]`**. Use the factory `build_attention(cfg)` with small dataclass configs.

### 1) WAVE / ARMA (softmax / linear / GLA / AFT-like)

```python
import torch
from sequencelab.build import build_attention
from sequencelab.config import WaveConfig

B, T, D, H = 2, 128, 512, 8
x = torch.randn(B, T, D)

cfg = WaveConfig(
    variant="softmax_arma",  # or: "linear_arma", "gla_arma", "aft_arma"
    n_embd=D,
    n_head=H,
    dropout=0.0,
    ma_dropout=0.0,          # ARMA branch dropout
    bias=True,
    block_size=1024,         # only relevant for some softmax fallbacks
    decay=True               # used by GLA; safe to leave True
)

layer = build_attention(cfg).eval()
y = layer(x)                 # shape: (B, T, D)
print(y.shape)
```

### 2) ZeroS (associative attention with RoPE)

```python
import torch
from sequencelab.build import build_attention
from sequencelab.config import ZeroSConfig

B, T, D, H = 1, 32, 256, 4
x = torch.randn(B, T, D)

cfg = ZeroSConfig(
    n_embd=D, n_head=H,
    dropout=0.0, bias=True,
    block_size=T,            # RoPE length
    is_causal=True,
    init_params=True,        # GPT-2 style init for quick smoke tests
    init_n_layers=1,
    use_norm=True,
    use_associative=True
)

layer = build_attention(cfg).eval()
y = layer(x)
print(y.shape)
```

### 3) FEM (Free Energy Mixer)

```python
import torch
from sequencelab.build import build_attention
from sequencelab.config import FEMConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.bfloat16 if device == "cuda" else torch.float32

B, T, D, H = 2, 128, 512, 8
x = torch.randn(B, T, D, device=device, dtype=dtype)

# Optional boolean mask (True=keep, False=pad)
mask = torch.ones(B, T, dtype=torch.bool, device=device)
mask[:, -3:] = torch.tensor([1, 0, 1], dtype=torch.bool, device=device)

cfg = FEMConfig(
    n_embd=D, n_head=H,
    prior_type="softmax",        # "softmax" | "linear" | "gla" | "rnn_softmax" | "mamba"
    dropout=0.0, causal=True, bias=True,

    # Parameter budget (paper-aligned examples)
    fem_ratio=2/3,               # value path width (~ 2D/3)
    p_t_to_fem_ratio=2.0,        # total |Q|+|K| relative to fem_dim

    # Free-energy branch
    use_temperature=True,
    use_lse=True,
    use_outer_gate=True,

    # Extras (fully-parallel, CPU-safe)
    use_rope=True,
    use_conv=True,
    conv_hidden=64,
    conv_norm_first=True,
    conv_bidirectional=False,
)

layer = build_attention(cfg).to(device=device, dtype=dtype).eval()
with torch.no_grad():
    y = layer(x, attention_mask=mask)
print(y.shape)                    # (B, T, D)
```

---

## API overview

### Factory

```python
from sequencelab.build import build_attention
```

* **`build_attention(cfg)`** → `nn.Module`
  Unified entry point that returns an attention layer with signature:

  * **WAVE / ZeroS**: `y = layer(x)`
  * **FEM**: `y = layer(x, attention_mask: Optional[Bool[B,T]] = None)`
    All layers follow **`x: [B, T, D] → y: [B, T, D]`**.

---

### Config dataclasses

```python
from sequencelab.config import WaveConfig, ZeroSConfig, FEMConfig
```

* **`WaveConfig`** (WAVE / ARMA family)

  * `variant`: `"softmax_arma" | "linear_arma" | "gla_arma" | "aft_arma"`
  * `n_embd: int`, `n_head: int`, `dropout: float = 0.0`, `ma_dropout: float = 0.0`, `bias: bool = True`
  * `block_size: int | None = None`, `max_len: int | None = None` (used in some fallbacks)
  * `decay: bool = True` (for GLA; safe to keep True)

* **`ZeroSConfig`** (ZeroSAttention)

  * `n_embd: int`, `n_head: int`, `dropout: float = 0.0`, `bias: bool = True`
  * `block_size: int` (RoPE length), `is_causal: bool = True`
  * `init_params: bool = False`, `init_n_layers: int = 12` (GPT‑2 style scaling for `out_proj`)
  * `use_norm: bool = True`, `use_associative: bool = True`

* **`FEMConfig`** (FreeEnergyMixer)

  * Core: `n_embd: int`, `n_head: int`, `prior_type: str`
    (`"softmax" | "linear" | "gla" | "rnn_softmax" | "mamba"`)
  * Runtime: `dropout: float = 0.0`, `causal: bool = True`, `bias: bool = True`
  * Sizing:

    * `fem_dim: int | None = None` (if `None`, computed from `fem_ratio`)
    * `fem_ratio: float = 0.5`  → `fem_dim ≈ n_embd * fem_ratio` (rounded per head)
    * `qk_dim: int | None = None` (if `None`, derived from `p_t_to_fem_ratio`)
    * `p_t_to_fem_ratio: float = 4.0` → total `|Q|+|K| = p_t_to_fem_ratio * fem_dim`
  * Free‑energy branch:

    * `use_temperature: bool = True` (learn β; safe log‑exp path)
    * `use_lse: bool = True` (LSE branch; used if β=1)
    * `use_outer_gate: bool = True`
    * Numeric knobs: `value_logexp_cap: float = 40.0`, `eps: float = 1e-6`
  * Extras:

    * `use_rope: bool = True`, `rope_theta: float = 10000.0`
    * `use_conv: bool = True`, `conv_hidden: int = 64`, `conv_norm_first: bool = True`, `conv_bidirectional: bool = False`
  * Mamba (when `prior_type="mamba"`):

    * `ssm_rank: int = 16`, `mamba_normalize: bool = True`

---

### Direct layer access (optional)

If you prefer to bypass the factory:

```python
from sequencelab.layers.wave  import (
    CausalSelfAttentionARMA, LinearAttentionARMA,
    GatedLinearAttentionARMA, TwoStageSelfgatingRNNARMA,
)
from sequencelab.layers.zeros import ZeroSAttention
from sequencelab.layers.fem   import FreeEnergyMixer
```

All layers are plain `nn.Module`s. WAVE/ZeroS take keyword arguments or a small `SimpleNamespace` config, and FEM takes a `FEMConfig`.