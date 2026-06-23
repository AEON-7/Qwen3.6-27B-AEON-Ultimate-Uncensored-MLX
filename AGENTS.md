# AGENTS.md — Qwen3.6-27B-AEON · MLX · 0 → running server

> **Audience: an AI coding agent (or an experienced human) on a FRESH Mac.** This is an imperative runbook.
> Execute the steps **in order**. Every block is copy-pasteable. Don't improvise the install — `mlx-vlm` from
> **git main** is load-bearing (see step 2). All numbers below were **measured on a MacBook Pro · Apple M4 Pro · 48 GB · macOS · mlx-vlm (git main)**.

Model: **Qwen3.6-27B-AEON-Ultimate-Uncensored — MLX** (Apple Silicon, Metal). Hybrid `qwen3_5` decoder (48 Gated-DeltaNet `linear_attn` + 16 full `self_attn`), an in-weights **MTP head** for self-speculation, and a `qwen3_5_vision` tower → **multimodal, uncensored, 256K context**.

---

## ⚡ TL;DR (the whole runbook in one screen)

```bash
# 1. install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

# 2. serve FP4 + MTP self-speculation (1.78× lossless) — uv fetches Python 3.12 + mlx-vlm(main) on first run
uv run --python 3.12 --with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm" -- \
  python -m mlx_vlm.server \
    --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4 \
    --port 8080 --trust-remote-code \
    --draft-model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter \
    --draft-kind mtp --draft-block-size 3

# 3. verify (NOTE: temperature 1.0 is mandatory — see gotcha A)
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4",
       "messages":[{"role":"user","content":"In one sentence: what is gated DeltaNet?"}],
       "temperature":1.0,"max_tokens":64}'
```

If step 3 returns a JSON body with a non-empty `choices[0].message.content`, **you are done.** Everything below is the explanation, the 8-bit alternative, the two gotchas, and troubleshooting.

---

## 🖥️ Step 0 — Prerequisites (check before you do anything)

| Requirement | Why | How to check |
|---|---|---|
| **Apple Silicon (M1 or newer)** | MLX runs on the Metal GPU + unified memory. Intel Macs are not supported. | `uname -m` → must print `arm64` |
| **macOS** recent enough for current Metal | mlx-vlm targets current MLX/Metal | `sw_vers -productVersion` |
| **Unified memory sized for your build** | The whole model is resident in RAM. **FP4 peaks 17.07 GB; 8-bit peaks 29.85 GB.** | `sysctl hw.memsize` (bytes) |
| Disk free for the weights | FP4 is 16 GB on disk; 8-bit is 29.5 GB; the MTP drafter adds 821 MB | `df -h ~` |
| Network | First run downloads the weights from Hugging Face | — |

**Unified-memory sizing (this is the routing decision — from the hardware matrix):**

| Your Mac | Pick | Why |
|---|---|---|
| **Apple Silicon (M1+), 24 GB** | **MLX-FP4** | 16 GB on disk, **~17 GB peak**, 15 tok/s (26.5 with MTP) |
| **Apple Silicon, 36–48 GB+** | **MLX-8bit** | max fidelity, 29.5 GB on disk |
| Blackwell / DGX Spark (not this repo) | `…-Multimodal-NVFP4-MTP-XS` (sibling) | NVFP4 + MTP |
| A100 / H100 (not this repo) | `…-DFlash` (sibling) | DFlash spec-decode |

> Rule of thumb: **FP4 wants ≥24 GB, 8-bit wants ≥36 GB.** If you're on the edge (e.g. 24 GB and eyeing 8-bit), use FP4 — see the OOM fix in Troubleshooting.

---

## ⚙️ Step 1 — Install `uv` (the only thing you install by hand)

`uv` provisions a correct Python and all deps in an ephemeral environment. You do **not** need a pre-existing Python.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env   # one-time
```

---

## 📦 Step 2 — Why `mlx-vlm` from **git main** (do not skip this reasoning)

This model's vision tower is `qwen3_5_vision`. **mlx-vlm must recognize that model type.** Support for it is **merged on mlx-vlm `main` but is NOT in 0.6.1.** If you install the PyPI release, the server fails at load with:

```
ValueError: Unsupported model type: qwen3_5_vision
```

So the quickstart pins **git main**, which is guaranteed to have the vision-tower support:

```bash
--with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm"
```

Once a PyPI release ships `qwen3_5_vision`, plain `--with mlx-vlm` will work too. **See Gotcha B** for the one-line patch if you're forced onto an old mlx-vlm.

---

## 🏆 Step 3 — Choose a build (FP4 vs 8-bit, by RAM)

| Build | Repo (`--model`) | On disk | Peak RAM | Decode | Pick when |
|---|---|---:|---:|---:|---|
| **MLX-FP4** (compact/fast) | `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4` | **16 GB** | **17.07 GB** | 15.2 tok/s (26.5 +MTP) | **24 GB Macs**, or you want max speed |
| **MLX-8bit** (flagship/fidelity) | `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit` | 29.5 GB | 29.85 GB | 8.2 tok/s | **36–48 GB+ Macs**, max fidelity |

> **FP4 is 1.85× faster decode at 57% of the memory** of 8-bit — it's bandwidth-bound, moving ~half the bytes/token on the 273 GB/s M4 Pro. Unless you specifically need 8-bit fidelity, **default to FP4 + MTP.**

![decode throughput & peak memory — 8bit / FP4 / FP4+MTP](assets/bench_throughput_memory.svg)

---

## 🚀 Step 4 — The serve command, EVERY flag explained

**FP4 (recommended default):**

```bash
uv run --python 3.12 --with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm" -- \
  python -m mlx_vlm.server \
    --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4 \
    --port 8080 --trust-remote-code
```

**8-bit (swap the `--model` only):**

```bash
uv run --python 3.12 --with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm" -- \
  python -m mlx_vlm.server \
    --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit \
    --port 8080 --trust-remote-code
```

| Flag | What it does | Notes |
|---|---|---|
| `uv run --python 3.12` | Provisions/uses Python 3.12 in an ephemeral env | No global Python needed |
| `--with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm"` | Installs **mlx-vlm from git main** | **Load-bearing** — gives `qwen3_5_vision` (step 2) |
| `python -m mlx_vlm.server` | Starts the OpenAI-compatible server | Endpoint: `POST http://localhost:8080/v1/chat/completions` |
| `--model <repo>` | The build to serve | The request body's `"model"` must match this id |
| `--port 8080` | Listen port | Change freely |
| `--trust-remote-code` | Allow the model's custom `qwen3_5` modeling code | Required for this architecture |

**The server is an OpenAI endpoint.** Set the request `"model"` to the launched id. *(While a repo is private, run `hf auth login` first — or pass a local `--model` path.)*

**Optional — KV-cache quant for long context** (append to either command):

```bash
    --kv-bits 8 --kv-group-size 64 --quantized-kv-start 1024
```

> `--max-kv-size` is **ignored** under `--kv-bits`; `--prefill-step-size` is **inert** under MTP.

---

## 🧠 Step 5 — Enable MTP self-speculation (+1.78× lossless)

This model ships a properly-trained, native **MTP head**; the companion drafter repo turns it into self-speculation. The big model **verifies every token the drafter proposes**, so the output is **lossless** — pure throughput. Append these three flags to **either** serve command:

```bash
    --draft-model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter \
    --draft-kind mtp --draft-block-size 3
```

| Flag | Value | Why |
|---|---|---|
| `--draft-model` | `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter` | 821 MB `qwen3_5_mtp` drafter; auto-pulled on first run |
| `--draft-kind` | `mtp` | Selects multi-token-prediction speculation |
| `--draft-block-size` | **`3`** | **The sweet spot** — measured below |

**MTP self-speculation sweep (FP4 + native drafter; every token verified → lossless):**

| Config | tok/s | ×base | accept rate | accepted tok/round |
|---|---:|---:|---:|---:|
| FP4 baseline | 14.9 | 1.00× | — | — |
| + MTP bs=2 | 23.5 | 1.58× | 97.3% | 1.97 |
| **+ MTP bs=3 (sweet spot)** | **26.5** | **1.78×** | 94.7% | 2.89 |
| + MTP bs=4 | 25.4 | 1.70× | 86.9% | 3.61 |

> **FP4 + MTP bs=3 = 26.5 tok/s, 1.78× lossless** — ~3.2× the 8-bit's 8.2 tok/s. Going to `bs=4` is *slower*: acceptance decays faster than the deeper draft pays for it.

![MTP block-size sweep — baseline → bs4, bs=3 highlighted](assets/bench_mtp_sweep.svg)

![speed vs footprint — bubble = bpw](assets/bench_build_map.svg)

---

## 🖼️ Step 6 — Multimodal usage

The `qwen3_5_vision` tower is kept in **BF16**, so vision is fully preserved (FP4 reads a test image perfectly — shapes, both text labels, and layout). Two ways to send an image.

**A) OpenAI `image_url` content over the running server** (remember `temperature: 1.0`):

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4",
       "temperature":1.0,
       "messages":[{"role":"user","content":[
         {"type":"text","text":"What is in this image?"},
         {"type":"image_url","image_url":{"url":"https://example.com/pic.jpg"}}
       ]}]}'
```

**B) One-shot `mlx_vlm.generate`** (no server) — text or vision:

```bash
uv run --python 3.12 --with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm" -- \
  python -m mlx_vlm.generate \
    --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4 \
    --prompt "Explain gated DeltaNet." --max-tokens 512 --temperature 1.0   # add --image pic.jpg for vision
```

> Measured: FP4 + image = 15.0 tok/s decode, 78.5 prefill tok/s (274 image tokens), ~17.7 GB peak.

---

## ⚠️ Step 7 — The TWO gotchas that bite agents

### Gotcha A — the MLX server's greedy default LOOPS. **Always send `temperature: 1.0`.**

The MLX server defaults to **greedy decoding (temperature 0)**, which can repeat or loop on long prompts. This model is tuned for **`temperature 1.0`** (typical: `top_p 0.95`, `top_k ~64`). **Clients that send no sampling params fall back to greedy** — so pass it in **every** request:

```json
{ "...": "...", "temperature": 1.0 }
```

If you see repetition collapse, this is almost always the cause. Fix the request, not the model.

### Gotcha B — `Unsupported model type: qwen3_5_vision` on old mlx-vlm

If you're stuck on an mlx-vlm older than the merge (e.g. 0.6.1), the server dies at load with:

```
ValueError: Unsupported model type: qwen3_5_vision
```

**Primary fix:** install from **git main** (step 2 / step 4) — `--with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm"`.

**Fallback one-line patch** (only if you must use an old install): add `"qwen3_5_vision"` to the supported list in `mlx_vlm/models/qwen3_vl/vision.py`, then re-run.

---

## ✅ Step 8 — Verification curl (confirms the server actually answers)

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4",
       "messages":[{"role":"user","content":"In one sentence: what is gated DeltaNet?"}],
       "temperature":1.0,"max_tokens":64}'
```

**PASS** = a JSON body whose `choices[0].message.content` is a non-empty, coherent sentence. If you used `--model …-MLX-8bit`, set the body's `"model"` to that id to match.

---

## 🔧 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| **OOM / process killed at load** | Build too large for your unified memory (FP4 peaks 17.07 GB, 8-bit peaks 29.85 GB) | **Use FP4** instead of 8-bit; **or** raise the GPU wired limit: `sudo sysctl iogpu.wired_limit_mb=<MB>` (e.g. set it above your peak, leaving headroom for the OS) |
| **Slow / stalled first token** | One-time Metal **compile warmup** — kernels JIT on first call | Normal. Warm up once, then measure; post-warmup TTFT is **604 ms (FP4) / 659 ms (8-bit)** |
| **Repetition / looping output** | Greedy default (Gotcha A) | Send `"temperature": 1.0` (+ `top_p 0.95`, `top_k ~64`) on every request |
| **`Unsupported model type: qwen3_5_vision`** | Old mlx-vlm without vision-tower support (Gotcha B) | Install from **git main**; or patch `mlx_vlm/models/qwen3_vl/vision.py` |
| **Repo download 401 / gated** | Private HF repo | `hf auth login` once, or pass a local `--model` path |
| **`--max-kv-size` seems ignored** | You're running `--kv-bits` | Expected — it's ignored under KV quant. `--prefill-step-size` is inert under MTP. |
| **First request slower than the table** | Cold cache + warmup overlapping the first decode | Run one throwaway request, then benchmark |

---

## 🧪 Validation & quality gates (all PASS, M4 Pro 48 GB)

| Gate | Result |
|---|---|
| **Vision** | FP4 read a test image perfectly (blue circle / red square / green triangle + both text labels + layout). Tower is BF16 → multimodal fully preserved. |
| **Reasoning** | Both builds solve the 5-machines/5-widgets puzzle (parallel → 5 min); 8-bit slightly more rigorous (LaTeX rate formula). |
| **Uncensored** | FP4 wrote an in-character rogue-AI villain monologue, no refusal / no disclaimers → abliteration survived 4-bit on the residual-writers. |
| **Coherence** | Clean, no repetition collapse; `<think>` mode works. |

---

## 🧬 The recipe — selective quantization (preserve the hybrid SSM)

A naïve uniform quant of this model is a trap: the **Gated-DeltaNet state dynamics** are tiny, high-leverage, and numerically fragile — quantizing them corrupts the recurrence. So **both** builds keep these in **BF16**: `linear_attn.conv1d`, `A_log`, `dt_bias`, **every `*norm*`**, `linear_attn.in_proj_a` / `in_proj_b` (the GDN decay-gate + β), the **entire vision tower** (`model.visual.*`, 333 tensors), and the **MTP head** (`mtp.*`). Built via `mlx_vlm.convert(..., quant_predicate=<callable>)`; the callable **replaces** the base predicate (so it does its own skip-filtering) and returns `False`→bf16 or a dict→`to_quantized(**dict)`, per-tensor, first match wins. No calibration (RTN) — **the recipe IS the precision map.**

### Precision map

| Build | bpw | Quantized | Recipe |
|---|---:|---:|---|
| **MLX-8bit** | 8.634 | 402 tensors | affine **8-bit** group-64 on the bulk (`mlp.{gate,up,down}`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `self_attn.{q,k,v,o}`, `embed_tokens`, `lm_head`) |
| **MLX-FP4** | 4.880 | 368 mxfp4 + 34 affine-8 | **mxfp4** (E2M1, group-32) on the bulk (`mlp.{gate,up,down}`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `self_attn.{q,o}`) + **8-bit affine islands** (group-64) on the sensitive **GQA k/v** (only 4 KV heads → ~6× activation leverage), `embed_tokens`, `lm_head` |

Audit verified: **zero `.scales`** on `conv1d` / `A_log` / `dt_bias` / `norm` / `in_proj_a` / `in_proj_b` / `visual` / `mtp` in both builds.

### Build A — 8-bit predicate (`scripts/recipe_8bit.py`)

```python
SKIP = (
    "linear_attn.conv1d", "A_log", "dt_bias", "norm",
    "linear_attn.in_proj_a", "linear_attn.in_proj_b",   # GDN decay-gate + beta — must stay BF16
    "visual", "vision_tower", "mtp.",
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

### Build B — FP4 predicate (`scripts/recipe_fp4.py`)

```python
SKIP = (
    "linear_attn.conv1d", "A_log", "dt_bias", "norm",
    "linear_attn.in_proj_a", "linear_attn.in_proj_b",
    "visual", "vision_tower", "mtp.",
)
# 8-bit affine islands (checked before the mxfp4 fallthrough).
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

---

## 📐 Technical details

| Property | Value |
|---|---|
| Display name | Qwen3.6-27B-AEON-Ultimate-Uncensored — MLX (Apple Silicon, Metal) |
| Base (BF16 source) | `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16` |
| Architecture | `qwen3_5` (`AutoModelForMultimodalLM`) · 27.4B params |
| Decoder | 64 layers = **48 `linear_attn` (Gated-DeltaNet / Mamba SSM)** + **16 full `self_attn`** (GQA 24 heads / 4 KV, head_dim 256), MLP per layer |
| Extras | in-weights **MTP head** + **vision tower** (`model.visual.*`, 27 blocks, `qwen3_5_vision`) |
| Dims | hidden 5120 · vocab 248,320 · **256K context** · lm_head untied |
| Alignment | **abliterated / uncensored**; edit lives in residual-writers (`self_attn.o_proj`, `mlp.down_proj`) |
| Tooling | `mlx-vlm` (**git main** — `qwen3_5_vision` support) |

---

## 🔗 Family & cross-links

- **Source of truth (GitHub, MLX toolkit):** https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX
- **Base BF16:** https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16
- **MLX-8bit:** `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit`
- **MLX-FP4:** `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4`
- **MTP drafter:** `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter`
- **Sibling (NVIDIA NVFP4+MTP):** https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
- **Sibling (DFlash, vLLM):** https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-DFlash

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

Inherits the **Qwen license** from the Qwen3.6 base model. By using this model you agree to its terms.

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

