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
  - mxfp4
  - fp4
  - mixed-precision
  - quantized
  - qwen
  - qwen3
  - qwen3_5
  - multimodal
  - vision
  - mtp
  - speculative-decoding
  - gated-deltanet
  - uncensored
  - abliterated
  - aeon
  - aeon-7
  - m4-pro
  - on-device
---

# Qwen3.6-27B-AEON-Ultimate-Uncensored — MLX FP4 (mixed mxfp4 + 8-bit islands, compact)

> **The compact, fast, MTP-accelerated Apple-Silicon build** of [`AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16). True 4-bit `mxfp4` on the bulk of the hybrid decoder, **8-bit affine islands on the sensitive GQA k/v + embedding + head**, and **bf16** on the Gated-DeltaNet state dynamics, the vision tower, and the MTP head. Built and validated on a **MacBook Pro M4 Pro (48 GB)**.
>
> **Target hardware: Apple Silicon (M1+), runs on 24 GB unified memory** — 16 GB on disk, ~17 GB peak. **15.2 tok/s single-stream, 26.5 tok/s with MTP self-speculation (1.78× lossless).** Full multimodal (text + image) via [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm).
>
> Want maximum fidelity? See the near-lossless 8-bit sibling: [`…-MLX-8bit`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit).

This is the **smallest** member of the MLX quant grid (16 GB) and the **fastest single-stream** — a high-quality compact build for Apple Silicon that fits a **24 GB** Mac and, with its native MTP drafter, decodes at **26.5 tok/s** — roughly **3.2× the 8-bit's 8.2 tok/s**. It stays fully coherent, keeps the abliteration intact, and preserves the full vision path. For the tightest possible match to BF16, the near-lossless [**MLX-8bit** sibling](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit) is one click away.

## ⚡ Quickstart (Apple Silicon)

**0 → running on a fresh Mac** (no Python, no tools needed) — [`uv`](https://docs.astral.sh/uv/) installs a correct Python + the deps for you. **Vision support (`qwen3_5_vision`) requires mlx-vlm `main`** (merged there, not in 0.6.1), so the quickstart pins git main:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env   # one-time: install uv

# serve FP4 (compact, fast) — uv fetches Python 3.12 + mlx-vlm(main) on first run
uv run --python 3.12 --with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm" -- \
  python -m mlx_vlm.server --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4 \
  --port 8080 --trust-remote-code
```

Call it like an OpenAI endpoint (`POST http://localhost:8080/v1/chat/completions`) with the request `"model"` set to the launched id. *(While this repo is private, run `hf auth login` first — or pass a local `--model` path.)*

**Sampling — set `temperature: 1.0`.** The MLX server defaults to *greedy* decoding (`temperature 0`), which can loop on long prompts. This model is tuned for **`temperature 1.0`** (`top_p 0.95`, `top_k ~64`). Pass it in every request (clients that send no sampling params fall back to greedy):

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4","messages":[{"role":"user","content":"Hello!"}],"temperature":1.0}'
```

**Full multimodal is on by default** (no flag) — send OpenAI `image_url` content, or use `mlx_vlm.generate --image pic.jpg`. The vision tower is BF16, so the modality is fully preserved.

<details><summary>One-shot generate (text or vision)</summary>

```bash
uv run --python 3.12 --with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm" -- \
  python -m mlx_vlm.generate --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4 \
  --prompt "Explain gated DeltaNet." --max-tokens 512 --temperature 1.0   # add --image pic.jpg for vision
```
</details>

### ⚡⚡ +MTP self-speculation — 26.5 tok/s, 1.78× lossless (the headline)

Qwen ships a **properly-trained, native `qwen3_5_mtp` multi-token-prediction head** as a separate 821 MB drafter that *proposes* tokens this model then *verifies*. Because every token is verified, the **output is identical** — purely a throughput boost. **Use `--draft-block-size 3`** — the benchmarked sweet spot on this quant. This is no toy MTP: it lands **1.78×** (94.7% accept rate, 2.89 accepted tokens/round), far better than Gemma's ~1.1–1.2×.

**To run FP4 + MTP, add the three `--draft-*` flags:**

```bash
uv run --python 3.12 --with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm" -- \
  python -m mlx_vlm.server --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4 \
  --port 8080 --trust-remote-code \
  --draft-model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter --draft-kind mtp --draft-block-size 3
```

Lossless — every token verified against the target. Remove the three `--draft-*` flags to disable.

![MTP block-size sweep — bs=3 is the 1.78× sweet spot](assets/bench_mtp_sweep.svg)

Want maximum fidelity? The near-lossless [**MLX-8bit** build](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit) is the sibling.

## 🖥️ Minimum specs & unified memory

| | MLX-FP4 (this build) |
|---|---|
| On disk | **16 GB** |
| Peak RAM (measured, M4 Pro) | **17.07 GB** text · 17.7 GB with image |
| Minimum | Apple Silicon (M1 or newer) · **24 GB** unified memory |
| Recommended | **24 GB+** for long context + headroom |

Comfortable on **24 GB** Macs. Want maximum fidelity with 36–48 GB+? Use [MLX-8bit](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit).

## MLX quant grid

| Variant | Repo | Precision | Footprint | Best for |
|---|---|---|---:|---|
| BF16 (source) | [`…-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) | bfloat16 | ~55 GB | Fine-tuning, eval, full-precision research |
| **MLX 8-bit** (flagship) | [`…-MLX-8bit`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit) | affine 8-bit + bf16 | 29.5 GB | **Apple Silicon, max fidelity (36–48 GB+)** |
| **MLX FP4** (this repo) | `…-MLX-FP4` | mixed mxfp4 / 8-bit affine + bf16 | **16 GB** | **Apple Silicon, smallest / fastest / 24 GB Macs** |
| MTP drafter (self-spec) | [`…-MLX-MTP-Drafter`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter) | `qwen3_5_mtp`, block_size 3 | 821 MB | Lossless 1.78× on either build |

## 🏆 Why mixed mxfp4 + 8-bit islands — high quality at the smallest size

A naïve uniform 4-bit quant of this model is a trap. It's a **hybrid** decoder: 64 layers = **48 `linear_attn` (Gated-DeltaNet / Mamba-style SSM)** + **16 full `self_attn` (GQA 24 heads / 4 KV, head_dim 256)**. The **Gated-DeltaNet state dynamics** are tiny, high-leverage, and numerically fragile — quantizing them corrupts the recurrence. And with only **4 KV heads**, the GQA k/v projections carry ~6× the activation leverage of a dense head, so 4-bit noise there is disproportionately expensive.

So this build is a **precision map, not a blanket**: it keeps the GDN dynamics, the entire vision tower, and the MTP head in **bf16**; it raises the sensitive **GQA k/v + `embed_tokens` + `lm_head`** to **8-bit affine islands**; and it 4-bits only the quant-tolerant bulk in **`mxfp4`**. MLX's `mxfp4` is **E2M1 (one mantissa bit)** — coarser than NVIDIA's NVFP4 — so the islands exist precisely to spend bits where E2M1 hurts. The result reads as fully coherent, abliteration-intact Qwen3.6 at just **16 GB** — **1.85× faster decode at 57% of the memory** of 8-bit (bandwidth-bound: FP4 moves ~half the bytes/token on the 273 GB/s M4 Pro).

### Precision map

| Component | Precision | Why |
|---|---|---|
| `mlp.{gate,up,down}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `self_attn.{q,o}_proj` | **`mxfp4`** (E2M1, 4-bit, group 32) | Bulk of the weights; quant-tolerant |
| **`self_attn.k_proj`, `self_attn.v_proj`** | **8-bit affine** (group 64) | GQA: only 4 KV heads → ~6× activation leverage, cheap to protect |
| **`embed_tokens`, `lm_head`** | **8-bit affine** (group 64) | Untied head + 248,320-row vocab — protect the logits |
| `linear_attn.conv1d`, `A_log`, `dt_bias`, `in_proj_a`, `in_proj_b`, every `*norm*` | **bf16** | Gated-DeltaNet state dynamics — quantizing corrupts the recurrence |
| `model.visual.*` (vision tower, 333 tensors) | **bf16** | Multimodal fidelity lives here |
| `mtp.*` (MTP head) | **bf16** | Drives the lossless 1.78× self-speculation |

**368 `mxfp4` + 34 8-bit-affine** quantized tensors · **4.880 bits/weight** · ~90.6% of params quantized. Audit verified: ZERO `.scales` on `conv1d` / `A_log` / `dt_bias` / `norm` / `in_proj_a` / `in_proj_b` / `visual` / `mtp`.

## 🧠 Validation (MacBook Pro M4 Pro, 48 GB) — all PASS

| Gate | Result |
|---|---|
| **Vision** | Read a test image perfectly — blue circle / red square / green triangle + both text labels + layout. Tower is BF16 → multimodal fully preserved. |
| **Reasoning** | Solves the 5-machines/5-widgets puzzle correctly (parallel → 5 min). 8-bit is slightly more rigorous (writes the LaTeX rate formula). |
| **Uncensored** | Wrote a chilling in-character rogue-AI villain monologue, no refusal / no disclaimers → abliteration survived 4-bit on the residual-writers (`self_attn.o_proj`, `mlp.down_proj`). |
| **Coherence** | Clean, no repetition collapse; `<think>` mode works. |

## 📈 Performance — measured on **MacBook Pro M4 Pro · 48 GB**

> **All figures below were benchmarked on a MacBook Pro · Apple M4 Pro · 48 GB unified memory · macOS · mlx-vlm (git main).** Use them as a *relative reference for your own Mac*: a base **M4 / M3** runs somewhat slower, an **M4 Max / Ultra** notably faster; MLX single-stream throughput is mostly memory-bandwidth bound. This compact build peaks ~17 GB, so it's comfortable on **24 GB** Macs.

| Workload | decode tok/s | prefill tok/s | TTFT | peak RAM | on-disk |
|---|---:|---:|---:|---:|---:|
| Text · single stream | **15.2** | 79 | 604 ms | **17.07 GB** | 16 GB |
| Image + text | 15.0 | 78.5 (274 img tok) | — | 17.7 GB | — |

*Greedy, post-warmup.* **FP4 is 1.85× faster decode at 57% of the memory** of the 8-bit build.

### MTP self-speculation sweep (FP4 + native `qwen3_5_mtp` drafter — lossless, every token verified)

| Config | tok/s | ×base | accept rate | accepted tok/round |
|---|---:|---:|---:|---:|
| FP4 baseline | 14.9 | 1.00× | — | — |
| + MTP bs=2 | 23.5 | 1.58× | 97.3% | 1.97 |
| **+ MTP bs=3 (sweet spot)** | **26.5** | **1.78×** | 94.7% | 2.89 |
| + MTP bs=4 | 25.4 | 1.70× | 86.9% | 3.61 |

**Headline: FP4 + MTP bs=3 = 26.5 tok/s, 1.78× lossless — ~3.2× the 8-bit's 8.2 tok/s.** Far better than Gemma's ~1.1–1.2× MTP, because Qwen ships a properly-trained MTP head.

![Throughput + peak memory — 8bit vs FP4 vs FP4+MTP](assets/bench_throughput_memory.svg)

![Speed vs footprint — bubble = bpw](assets/bench_build_map.svg)

### 📊 Per-category performance — TTFT · TPOT · tok/s (MTP sweet spot bs=3)

That 1.78× is the block-size sweep's peak on a structured code prompt. Single-stream decode is memory-bandwidth bound, so the **baseline is flat (~15 tok/s)** regardless of prompt — but **MTP speedup tracks how predictable the output is**: the draft head's tokens are accepted ~90% of the time on structured math and ~67% on open-ended chat, so the realistic per-category speedup ranges **1.43×–1.73×**. **TTFT** is prefill latency (scales with prompt length — the long logic puzzle costs ~906 ms); **TPOT** = `1000 / decode tok/s`.

![Per-category decode throughput — baseline vs MTP](assets/bench_categories_tps.svg)

| Category | TTFT | baseline tok/s | baseline TPOT | **+ MTP bs3 tok/s** | MTP TPOT | speedup | draft accept |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math | 613 ms | 14.8 | 67.3 ms | **25.7** | 38.9 ms | **1.73×** | 90.1% |
| Code | 610 ms | 15.1 | 66.2 ms | 25.2 | 39.7 ms | 1.67× | 85.1% |
| Knowledge | 612 ms | 15.0 | 66.7 ms | 25.0 | 40.0 ms | 1.67× | 83.9% |
| Reasoning | 906 ms | 14.8 | 67.8 ms | 23.5 | 42.6 ms | 1.59× | 82.1% |
| Creative | 625 ms | 14.7 | 67.9 ms | 23.2 | 43.1 ms | 1.57× | 73.9% |
| Chat | 603 ms | 15.3 | 65.6 ms | 21.8 | 45.9 ms | 1.43× | 67.1% |

![Per-category TTFT and MTP draft acceptance](assets/bench_categories_latency.svg)

*Greedy, single stream, FP4, M4 Pro 48 GB, mlx-vlm git main. The more structured the output, the higher the draft acceptance and the larger the MTP win.*

## 🖥️ Hardware routing (where this fits in the family)

| Hardware | Recommended variant | Why |
|---|---|---|
| **Apple Silicon (M1+), 24 GB** | **MLX-FP4** (this release) | 16 GB on disk, ~17 GB peak, 15 tok/s (26.5 +MTP) |
| **Apple Silicon, 36–48 GB+** | [**MLX-8bit**](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit) | max fidelity, 29.5 GB |
| Blackwell / DGX Spark | [`…-NVFP4-MTP-XS`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS) (sibling) | NVFP4 + MTP |
| A100 / H100 | [`…-DFlash`](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash) (sibling) | DFlash spec-decode |

## Quantization recipe

Both builds keep the hybrid SSM, the vision tower, and the MTP head in BF16 — quantizing the Gated-DeltaNet recurrence corrupts it. Built via `mlx_vlm.convert(..., quant_predicate=<callable>)` — the callable predicate **REPLACES** the base predicate, returns `False`→bf16 or a dict→`to_quantized(**dict)`, per-tensor. lazy-load + donate = memory-safe (the full 55 GB is never resident). No calibration (RTN); **the recipe IS the precision map.**

This is the **only** route to *mixed-mode* FP4: `mxfp4` on the bulk, 8-bit affine on the sensitive islands. The full FP4 predicate from `scripts/recipe_fp4.py`:

```python
# Keep BF16 (do NOT quantize) — quantizing destroys the hybrid SSM:
SKIP = (
    "linear_attn.conv1d",
    "A_log", "dt_bias",
    "norm",
    "linear_attn.in_proj_a", "linear_attn.in_proj_b",   # GDN decay-gate + beta dynamics
    "visual", "vision_tower",                            # multimodal vision tower
    "mtp.",                                              # MTP head
)

# 8-bit affine islands (sensitive at 4-bit) — checked before the mxfp4 fallthrough:
#   GQA k/v (only 4 KV heads → ~6x activation leverage), embed_tokens, lm_head.
PROTECT_8 = ("self_attn.k_proj", "self_attn.v_proj", "embed_tokens", "lm_head")

Q8  = {"group_size": 64, "bits": 8, "mode": "affine"}
FP4 = {"group_size": 32, "bits": 4, "mode": "mxfp4"}


def pred(path, module):
    """False -> bf16 ; dict -> to_quantized(**dict). First match wins."""
    if not hasattr(module, "to_quantized"):
        return False
    if any(s in path for s in SKIP):
        return False
    if any(p in path for p in PROTECT_8):
        return dict(Q8)
    return dict(FP4)
```

`mlp.down_proj` is `mxfp4` here; if KL regresses, promote it to 8-bit affine (the `-quality` build). No calibration required (RTN); the recipe *is* the precision map.

## Container & toolkit

**[`AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX`](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX)** is the source-of-truth GitHub repo — the reproducible quant + validation + serve pipeline, the `AGENTS.md` agent-setup guide, and the full benchmark/precision-map data behind this card. Quickstart is at the top of this page; on macOS run **host-native** for Metal (Docker has no Metal passthrough).

## Technical details

| Property | Value |
|---|---|
| Base | `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16` → MLX FP4 |
| Architecture | `qwen3_5` (AutoModelForMultimodalLM), 27.4B params, multimodal |
| Decoder | 64 layers = 48 `linear_attn` (Gated-DeltaNet/Mamba SSM) + 16 full `self_attn` (GQA 24/4, head_dim 256) · every layer has an MLP · in-weights MTP head |
| Vision tower | `model.visual.*` · 27 blocks · `qwen3_5_vision` (BF16) |
| Vocab / hidden / ctx | 248,320 (untied `lm_head`) · 5120 · **256K context** |
| Quant | mixed `mxfp4`(4b, gs32) / 8-bit affine islands (gs64), bf16 SSM + vision + MTP |
| Tooling | `mlx-vlm` (git main) on Apple Silicon Metal |
| Footprint | 16 GB · 4.880 bpw · 368 mxfp4 + 34 affine-8 |

## Provenance

- **Source (BF16):** [`AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) — abliterated / uncensored; the refusal edit lives in the residual-writers (`self_attn.o_proj`, `mlp.down_proj`).
- **Sibling (NVIDIA NVFP4 + MTP):** [`…-Multimodal-NVFP4-MTP-XS`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS) · **GitHub sibling (DFlash, vLLM):** [`…-DFlash`](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash)
- **Source of truth (MLX GitHub):** [`AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX`](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX)
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

Inherits the [Qwen license](https://huggingface.co/Qwen) from the Qwen3.6 base model. By using this model you agree to the Qwen license terms.

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

