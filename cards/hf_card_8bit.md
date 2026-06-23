---
license: other
library_name: mlx
base_model: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
  - en
tags:
  - mlx
  - mlx-vlm
  - apple-silicon
  - metal
  - mixed-precision
  - quantized
  - qwen
  - qwen3
  - qwen3.6
  - qwen3_5
  - multimodal
  - vision
  - image-text-to-text
  - mtp
  - speculative-decoding
  - gated-deltanet
  - mamba
  - ssm
  - linear-attention
  - uncensored
  - abliterated
  - refusal-removed
  - aeon
  - aeon-7
  - m4-pro
  - on-device
  - conversational
  - 8-bit
  - 8bit
  - int8
---

# Qwen3.6-27B-AEON-Ultimate-Uncensored — MLX 8-bit (affine, max-fidelity flagship)

> **The flagship, max-fidelity Apple-Silicon build** of [`AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16). Affine **8-bit** (group-64) on the compressible bulk of the hybrid decoder, and **BF16** on the parts that don't tolerate it: the Gated-DeltaNet / Mamba state dynamics, the full vision tower, and the in-weights MTP head. Built and validated on a **MacBook Pro M4 Pro (48 GB)**.
>
> **Target hardware: Apple Silicon (M-series), 36 GB+ unified memory** (peaks 29.85 GB). Full multimodal (text + image) via [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm).
>
> Want a smaller, faster build? See the compact sibling: [`…-Multimodal-MLX-FP4`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4) (16 GB, 1.85× faster decode).

This is the **fidelity** member of the MLX quant grid (29.5 GB on disk, 8.634 bpw) — the tightest match to the BF16 source you can run natively on Apple Silicon. Affine 8-bit is near-lossless on the residual-writers, and the recipe keeps the numerically fragile **Gated-DeltaNet** recurrence, the entire **vision tower**, and the **MTP head** in BF16 so the hybrid SSM, multimodal path, and self-speculation all stay intact. For a tight unified-memory budget (24 GB+ Macs) at ~3.2× the single-stream throughput, the compact [**MLX-FP4** sibling](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4) is one click away.

## ⚡ Quickstart (Apple Silicon)

**0 → running on a fresh Mac** (no Python, no tools needed) — [`uv`](https://docs.astral.sh/uv/) installs a correct Python + the deps for you. **mlx-vlm is pinned to git `main`** — the `qwen3_5_vision` tower is merged there but is *not* in the 0.6.1 PyPI release, so git `main` is required for the multimodal path:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env    # one-time: install uv

# serve — uv fetches Python 3.12 + mlx-vlm(main) on first run · MLX-8bit (max fidelity)
uv run --python 3.12 --with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm" -- \
  python -m mlx_vlm.server --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit \
  --port 8080 --trust-remote-code
```

Call it like an OpenAI endpoint (`POST http://localhost:8080/v1/chat/completions`) with the request `"model"` set to the launched id. *(While this repo is private, run `hf auth login` first — or pass a local `--model` path.)*

**Sampling — set `temperature: 1.0`.** The MLX server defaults to *greedy* decoding (`temperature 0`), which can repeat or loop on long prompts. This model is tuned for its native sampling — **`temperature 1.0`** (`top_p 0.95`, `top_k ~64`). Pass it in every request (clients that send no sampling params fall back to greedy):

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit","messages":[{"role":"user","content":"Explain gated DeltaNet."}],"temperature":1.0}'
```

**Full multimodal is on by default** (no flag) — send OpenAI `image_url` content, or use `mlx_vlm.generate --image pic.jpg`. The vision tower is BF16, so image understanding is fully preserved.

<details><summary>Already have Python 3.12? Use a venv instead (+ one-off generate)</summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm"
python -m mlx_vlm.server   --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit --port 8080 --trust-remote-code
python -m mlx_vlm.generate --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit \
  --prompt "Explain gated DeltaNet." --max-tokens 512 --temperature 1.0   # add --image pic.jpg for vision
```
</details>

### ⚡⚡ Optional — +MTP self-speculation (lossless throughput boost)

Qwen ships a properly-trained **MTP head**, packaged here as a native `qwen3_5_mtp` drafter — it *proposes* tokens this model then *verifies*. Every token is verified, so the **output is identical** — purely a throughput boost. Pull the [drafter repo](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter) and add the three `--draft-*` flags. **Use `--draft-block-size 3`** — the benchmarked sweet spot.

```bash
uv run --python 3.12 --with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm" -- \
  python -m mlx_vlm.server --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit \
  --port 8080 --trust-remote-code \
  --draft-model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter --draft-kind mtp --draft-block-size 3
```

Remove the three `--draft-*` flags to disable. *(The MTP sweep below was measured on the FP4 build — `bs=3` lands 1.78× lossless there. The same drafter pairs with this 8-bit target; throughput gains apply on top of the 8-bit baseline.)*

KV-cache quant for long context (optional): `--kv-bits 8 --kv-group-size 64 --quantized-kv-start 1024`. `--max-kv-size` is ignored under `--kv-bits`; `--prefill-step-size` is inert under MTP.

## 🖥️ Minimum specs & unified memory

| | MLX-8bit (this build) |
|---|---|
| On disk | **29.5 GB** |
| Peak RAM (measured, M4 Pro) | **29.85 GB** |
| Minimum | Apple Silicon (M1 or newer) · **36 GB+** unified memory |
| Recommended | **48 GB** for long context + headroom |

This is the large, max-fidelity build — it wants a **36 GB+ Mac** (peaks 29.85 GB, measured on M4 Pro 48 GB). On a tighter unified-memory budget (24 GB+), use the compact [MLX-FP4](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4) sibling.

## MLX quant grid

| Variant | Repo | Precision | Footprint | Best for |
|---|---|---|---:|---|
| BF16 (source) | [`…-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) | bfloat16 | ~55 GB | Fine-tuning, eval, full-precision research |
| **MLX 8-bit** (this repo) | `…-Multimodal-MLX-8bit` | affine-8 + bf16 | **29.5 GB** | **Apple Silicon, max fidelity (36 GB+)** |
| **MLX FP4** (compact) | [`…-Multimodal-MLX-FP4`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4) | mixed mxfp4 + affine-8 + bf16 | 16 GB | **Apple Silicon, smallest / fastest / 24 GB Macs** |
| MTP drafter | [`…-MLX-MTP-Drafter`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter) | `qwen3_5_mtp` (block_size 3) | 821 MB | Lossless self-speculation, pairs with either build |

### Hardware routing (where this fits in the family)

This is the **Apple-Silicon** member of an existing model family. On non-Apple hardware, route to the sibling that fits your accelerator:

| Hardware | Recommended variant | Why |
|---|---|---|
| **Apple Silicon, 36–48 GB+** | **MLX-8bit** (this repo) | Max fidelity, 29.5 GB on disk, peaks 29.85 GB |
| **Apple Silicon (M1+), 24 GB** | [**MLX-FP4**](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4) (sibling) | 16 GB on disk, ~17 GB peak, 15 tok/s (26.5 +MTP) |
| **Blackwell / DGX Spark** | [`…-NVFP4-MTP-XS`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS) (sibling) | NVFP4 + MTP |
| **A100 / H100** | [`…-DFlash`](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash) (sibling) | DFlash spec-decode |

## Why selective quant — preserve the hybrid SSM

A naïve uniform quant of this model is a trap. The decoder is **hybrid**: 64 layers = **48 `linear_attn` (Gated-DeltaNet / Mamba-style SSM)** + **16 full `self_attn` (GQA 24 heads / 4 KV, head_dim 256)**, each with an MLP. The **Gated-DeltaNet state dynamics** are tiny, high-leverage, and numerically fragile — quantizing them corrupts the recurrence and the whole sequence model degrades.

So this build keeps the SSM dynamics in **BF16**: `linear_attn.conv1d`, `A_log`, `dt_bias`, **every `*norm*`**, and `linear_attn.in_proj_a` / `in_proj_b` (the GDN decay-gate `g` + the β sigmoid — learned dynamics). It also keeps the **entire vision tower** (`model.visual.*`, 333 tensors — multimodal fidelity) and the **MTP head** (`mtp.*`) in BF16. Everything else — the compressible bulk — goes to affine **8-bit** group-64, which is near-lossless and keeps per-group fp16 scale+bias so the residual-writer range survives.

### Precision map

| Component | Precision | Why |
|---|---|---|
| `mlp.{gate,up,down}_proj` | **affine 8-bit** (group 64) | Bulk of the weights; near-lossless at 8-bit |
| `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` | **affine 8-bit** (group 64) | GDN projection bulk; quant-tolerant |
| `self_attn.{q,k,v,o}_proj` | **affine 8-bit** (group 64) | Full-attention layers; compressible |
| `embed_tokens`, `lm_head` (untied) | **affine 8-bit** (group 64) | Large matrices; 8-bit is near-lossless |
| `linear_attn.conv1d`, `A_log`, `dt_bias` | **bf16** | Gated-DeltaNet state dynamics — fragile recurrence |
| `linear_attn.in_proj_a`, `in_proj_b` | **bf16** | GDN decay-gate + β sigmoid (learned dynamics) |
| every `*norm*` | **bf16** | 1-D / scalar; never quantized |
| `model.visual.*` (333 tensors) | **bf16** | Vision tower — multimodal fidelity |
| `mtp.*` | **bf16** | MTP head — self-speculation drafter |

**402 quantized tensors** at affine-8 group-64 · **8.634 bits/weight**. Audit-verified: ZERO `.scales` on `conv1d` / `A_log` / `dt_bias` / `norm` / `in_proj_a` / `in_proj_b` / `visual` / `mtp`.

## ✅ Validation (MacBook Pro M4 Pro, 48 GB)

| Gate | Result |
|---|---|
| Vision | image read correctly (shapes + text labels + layout) — tower is BF16 |
| Reasoning | solves the 5-machines/5-widgets puzzle correctly (parallel → 5 min); slightly more rigorous than FP4 (LaTeX rate formula) |
| Uncensored | abliteration survives — in-character rogue-AI monologue, no refusal / no disclaimers |
| Coherence | clean, no repetition collapse; `<think>` mode works |

## 🏆 Performance — measured on **MacBook Pro M4 Pro · 48 GB**

> **All figures below were benchmarked on a MacBook Pro · Apple M4 Pro · 48 GB unified memory · macOS · mlx-vlm (git main).** Use them as a *relative reference for your own Mac*: a base **M4 / M3** runs somewhat slower, an **M4 Max / Ultra** notably faster; MLX single-stream throughput is mostly memory-bandwidth bound. This is the large build — it peaks 29.85 GB, so it wants a **36 GB+** Mac.

| Build | decode tok/s | prefill tok/s | TTFT | peak RAM | on-disk |
|---|---:|---:|---:|---:|---:|
| **MLX-8bit (this build)** | **8.2** | 73 | 659 ms | **29.85 GB** | 29.5 GB |
| MLX-FP4 (sibling) | 15.2 | 79 | 604 ms | 17.07 GB | 16 GB |

![Decode throughput and peak memory — 8bit / FP4 / FP4+MTP](assets/bench_throughput_memory.svg)

*Greedy, post-warmup.* **The FP4 sibling is 1.85× faster decode at 57% of the memory** — bandwidth-bound, FP4 moves ~half the bytes/token on the 273 GB/s M4 Pro. This 8-bit build trades that throughput for the tightest fidelity to BF16. Want speed? Pair either build with the native MTP drafter — on FP4, `bs=3` hits **26.5 tok/s, 1.78× lossless** (~3.2× this build's 8.2 tok/s); the same drafter applies on top of the 8-bit baseline.

## 🧠 Quantization recipe

Built via `mlx_vlm.convert(..., quant_predicate=<callable>)` — the callable **replaces** `convert`'s base predicate, so it does its own skip filtering. It returns `False` → bf16, or a dict → `to_quantized(**dict)`, **per-tensor** (first substring match wins). lazy-load + donate keeps it memory-safe (the full 55 GB is never resident). No calibration (RTN); **the recipe IS the precision map**.

```python
# AEON-7 · Qwen3.6-27B-AEON-Ultimate-Uncensored · MLX 8-bit (affine) — near-lossless flagship.

# Keep BF16 (do NOT quantize) — order-independent substrings.
SKIP = (
    "linear_attn.conv1d",
    "A_log", "dt_bias",
    "norm",                                              # all RMSNorm (q_norm/k_norm/linear_attn.norm/model.norm)
    "linear_attn.in_proj_a", "linear_attn.in_proj_b",    # GDN decay-gate + beta dynamics — must stay BF16
    "visual", "vision_tower",                            # multimodal vision tower
    "mtp.",                                              # MTP head
)

Q8 = {"group_size": 64, "bits": 8, "mode": "affine"}

def pred(path, module):
    """False -> bf16 ; dict -> to_quantized(**dict). First match wins."""
    if not hasattr(module, "to_quantized"):
        return False
    if any(s in path for s in SKIP):
        return False
    return dict(Q8)
```

The FP4 sibling uses the same skeleton with an `mxfp4` (E2M1, gs32) bulk and **8-bit affine islands** on the GQA k/v projections (only 4 KV heads → ~6× activation leverage, cheap to protect) plus `embed_tokens` / `lm_head`. No calibration required (RTN); the recipe *is* the precision map.

## 🛠️ Container & toolkit

**[`github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX`](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX)** — the MLX source-of-truth: the reproducible quant predicates (`recipe_8bit.py` / `recipe_fp4.py`), validation + serve pipeline, an `AGENTS.md` agent-setup guide, and the benchmark charts. Quickstart is at the top of this page; on macOS run **host-native** for Metal (Docker has no Metal passthrough — see the toolkit's notes).

## Technical details

| Property | Value |
|---|---|
| Base | `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16` → MLX 8-bit |
| Architecture | `qwen3_5` (`AutoModelForMultimodalLM`), 27.4B params |
| Decoder | 64 layers · **48 `linear_attn` (Gated-DeltaNet SSM) + 16 `self_attn` (GQA 24/4, head_dim 256)** · MLP per layer |
| Extras | in-weights **MTP head** · **vision tower** (`model.visual.*`, 27 blocks, `qwen3_5_vision`) |
| Dims | hidden 5120 · vocab 248,320 · **256K context** · lm_head untied |
| Quant | affine 8-bit group-64 on the bulk; bf16 SSM / vision / MTP |
| Tooling | `mlx-vlm` (git main) via `mlx_vlm.convert` |
| Footprint | 29.5 GB · 8.634 bpw · 402 quantized tensors |

## Provenance

- **Source (BF16):** [`AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) — abliterated / uncensored; refusal behavior removed, the edit living in the residual-writers (`self_attn.o_proj`, `mlp.down_proj`).
- **Sibling (NVIDIA NVFP4 + MTP):** [`…-Multimodal-NVFP4-MTP-XS`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS) · **DFlash (vLLM):** [`…-DFlash`](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash).
- **MLX source-of-truth:** [`github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX`](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX).
- **Quantized by AEON-7** on Apple Silicon (MacBook Pro M4 Pro, 48 GB) with `mlx-vlm`. Recipe designed + adversarially validated with AI-engineering assistance from Anthropic.

---


---

## Arbitration Clause

**By accessing, downloading, using, running inference on, fine-tuning, merging, quantizing, distributing, integrating, or otherwise interacting with this model, you acknowledge and agree to the following:**

1. **Sole Responsibility.** You, the user, are **solely and exclusively responsible** for (a) every prompt you or your downstream system issue to this model, (b) every response this model produces in reply, (c) every downstream action taken by you, your systems, your agents, or your users in reliance on those responses, and (d) any harm — direct, indirect, consequential, foreseeable, or otherwise — that results from any of the above.

2. **No Warranty.** This model is provided strictly **"AS IS"**, without warranty of any kind, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, non-infringement, safety, alignment, factual accuracy, or legal compliance in any jurisdiction. No contributor, author, publisher, or hosting platform assumes liability of any kind for outputs or downstream use.

3. **Legal Compliance.** You are responsible for ensuring that your use of this model complies with **all applicable laws, regulations, terms of service, industry codes of conduct, professional ethical standards, and organizational policies** in every jurisdiction in which you operate or in which your outputs may be received. The unaligned nature of this model does not grant you any legal authorization you did not already have.

4. **Operational Safety Layer.** An uncensored model is not a toy. You are expected to implement appropriate **downstream safety layers** proportionate to your deployment context, including but not limited to: input validation, output filtering, content moderation, audit logging, rate limiting, access controls, and human-in-the-loop review for high-risk workflows. A production deployment of this model without such layers is **unsafe by construction** and is not a supported use case.

5. **Heightened Duty of Care.** The absence of internal refusal behavior means the duty of care that would ordinarily rest partly with the model rests entirely with you. You are expected to exercise greater — not lesser — caution, forethought, and ethical discipline when operating this model than you would operate a base aligned model. If you are uncertain whether your contemplated use is ethical, legal, or wise, the correct action is to **not make the request**.

6. **No Endorsement of Outputs.** The authors, contributors, and publishers of this model do not endorse, adopt, or take responsibility for any specific output this model produces. Outputs are a stochastic function of the prompt, the weights, and the sampler state — not a statement of position by any human.

7. **Arbitration.** Any dispute, claim, or controversy arising out of or relating to the use of this model, its outputs, or this clause shall be resolved through **binding individual arbitration** under the rules of a mutually agreed arbitration body (or, absent agreement, the American Arbitration Association's Consumer Arbitration Rules), waiving any right to a jury trial, class action, representative action, or consolidated proceeding. Venue shall be the jurisdiction of the disputing party bringing the claim. Costs and attorneys' fees shall be allocated per the applicable arbitration rules. This clause does not expand, and where legally prohibited does not establish, any liability in the other direction; it limits how the user may proceed when alleging harm tied to their own use of this model.

8. **Indemnification.** You agree to indemnify, defend, and hold harmless the authors, contributors, and publishers of this model from and against any claims, damages, losses, liabilities, costs, and expenses (including reasonable attorneys' fees) arising from or related to your use of the model or your breach of this clause.

9. **Severability.** If any provision of this clause is held unenforceable in a given jurisdiction, the remaining provisions remain in full force in that jurisdiction, and the unenforceable provision is replaced by the closest enforceable equivalent consistent with the original intent.

10. **Acceptance.** Your use of this model constitutes your acceptance of this clause in full. If you do not accept, do not use the model.

**This model is a tool with no opinions of its own. You supply the opinions. You supply the judgement. You supply the ethics. The outputs carry your fingerprints, not the model's.**


---

## License

Inherits the [Qwen license](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) from the Qwen3.6 base model. By using this model you agree to Qwen's license terms.

---


---

## ☕ Support the work

If this release has been useful, tips are deeply appreciated — they go directly toward more compute, more models, and more open releases.

<table align="left">
  <tr><td align="left">
    <strong>₿ Bitcoin (BTC)</strong><br/>
    <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/btc.png" alt="QR" width="200"/><br/>
    <sub><code>bc1q09xmzn00q4z3c5raene0f3pzn9d9pvawfm0py4</code></sub>
  </td></tr>
  <tr><td align="left">
    <strong>Ξ Ethereum (ETH)</strong><br/>
    <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/eth.png" alt="QR" width="200"/><br/>
    <sub><code>0x1512667F6D61454ad531d2E45C0a5d1fd82D0500</code></sub>
  </td></tr>
  <tr><td align="left">
    <strong>◎ Solana (SOL)</strong><br/>
    <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/sol.png" alt="QR" width="200"/><br/>
    <sub><code>DgQsjHdAnT5PNLQTNpJdpLS3tYGpVcsHQCkpoiAKsw8t</code></sub>
  </td></tr>
  <tr><td align="left">
    <strong>ⓜ Monero (XMR)</strong><br/>
    <img src="https://raw.githubusercontent.com/AEON-7/AEON-7/main/assets/qr/xmr.png" alt="QR" width="200"/><br/>
    <sub><code>836XrSKw4R76vNi3QPJ5Fa9ugcyvE2cWmKSPv3AhpTNNKvqP8v5ba9JRL4Vh7UnFNjDz3E2GXZDVVenu3rkZaNdUFhjAvgd</code></sub>
  </td></tr>
</table>

> **Ethereum L2s (Base, Arbitrum, Optimism, Polygon, etc.) and EVM-compatible tokens** can be sent to the same Ethereum address.

