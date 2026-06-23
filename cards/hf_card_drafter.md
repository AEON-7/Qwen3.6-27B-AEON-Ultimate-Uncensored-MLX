---
license: other
language:
  - en
library_name: mlx
base_model: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16
base_model_relation: quantized
tags:
  - mlx
  - mtp
  - speculative-decoding
  - qwen3_5_mtp
  - qwen3_5
  - aeon-7
---

# Qwen3.6-27B-AEON — MLX MTP Drafter (native multi-token-prediction head)

> **This is not a chat model.** It is the **split-out native multi-token-prediction (MTP) head** of [`AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16), packaged as a standalone **speculative drafter** (821 MB · `model_type` `qwen3_5_mtp` · `block_size 3`). It **proposes** tokens that a full MLX target model then **verifies** — purely a throughput boost. Do not load it on its own; it has no business answering prompts.

Qwen ships a properly-trained MTP head in this architecture, so unlike a bolted-on draft model it accepts deep. On the FP4 target this drafter hits **1.78× lossless decode** at `block_size 3` — far better than the ~1.1–1.2× you get from generic MTP heads on other families. Measured on a **MacBook Pro · M4 Pro · 48 GB**.

## ⚡ Quickstart — attach it to a target

The drafter is a flag, not a server. Launch either MLX target and add three `--draft-*` flags:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env   # one-time: install uv

# serve FP4 target + this MTP drafter (1.78× lossless) — uv fetches Python 3.12 + mlx-vlm(main) on first run
uv run --python 3.12 --with "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm" -- \
  python -m mlx_vlm.server \
  --model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4 --port 8080 --trust-remote-code \
  --draft-model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter \
  --draft-kind mtp --draft-block-size 3
```

The three flags that matter:

```bash
  --draft-model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter \
  --draft-kind mtp \
  --draft-block-size 3
```

- Pairs with **either** MLX target — [`…-MLX-FP4`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4) (compact/fast, where it shines) or [`…-MLX-8bit`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit) (max fidelity). Just point `--model` at the target you want.
- `--draft-block-size 3` is the benchmarked sweet spot — see the sweep below. Drafting deeper (`bs=4`) accepts more raw tokens per round but its lower accept rate drags net throughput back down.
- Remove the three `--draft-*` flags to disable speculation. `--prefill-step-size` is inert under MTP.
- **Sampling:** the MLX server defaults to greedy (`temperature 0`), which can loop on long prompts. This family is tuned for **`temperature 1.0`** (`top_p 0.95`, `top_k ~64`) — pass it in every request. Speculation stays lossless under sampling: the target verifies every proposed token against its own distribution.

## 🧠 What it actually is

The full model carries an in-weights MTP head (`mtp.*`) that, conditioned on the target's hidden state, predicts the next few tokens in one shot. Both MLX builds keep that head in **BF16** during quantization (it's never quantized — see either target's recipe). This repo is that head, extracted and shipped on its own so the server can load it as a lightweight side model:

1. The drafter proposes a block of up to 3 tokens from the target's last hidden state.
2. The full target runs **one** forward pass over the proposed block and verifies it against its own next-token distribution.
3. Accepted tokens are kept; the first rejection truncates the block and the target's own token is used.

Because step 2 is the target verifying against itself, **the output distribution is identical to running the target alone** — every token is verified. This is a speed optimization with **zero quality cost**. It is **lossless**.

## 🏆 Measured speedups (M4 Pro 48 GB, mlx-vlm git main; greedy, post-warmup)

Full block-size sweep, **FP4 target + this native `qwen3_5_mtp` drafter** — lossless, every token verified:

| Config | tok/s | ×base | accept rate | accepted tok/round |
|---|---:|---:|---:|---:|
| FP4 baseline | 14.9 | 1.00× | — | — |
| + MTP bs=2 | 23.5 | 1.58× | 97.3% | 1.97 |
| **+ MTP bs=3 (sweet spot)** | **26.5** | **1.78×** | 94.7% | 2.89 |
| + MTP bs=4 | 25.4 | 1.70× | 86.9% | 3.61 |

**Headline: FP4 + MTP bs=3 = 26.5 tok/s, 1.78× lossless** — about **3.2× the 8-bit's 8.2 tok/s**. The accept rate stays above 94% at `bs=3` because Qwen trained this head properly; push to `bs=4` and acceptance falls to 86.9%, so net throughput regresses. `bs=3` is the knee.

![MTP block-size sweep — baseline to bs=4, bs=3 highlighted](assets/bench_mtp_sweep.svg)

![Decode throughput and peak memory — 8bit / FP4 / FP4+MTP](assets/bench_throughput_memory.svg)

## 🖥️ Pairs with

| Target | Repo | Why |
|---|---|---|
| **MLX-FP4** (compact/fast) | [`…-Multimodal-MLX-FP4`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4) | 16 GB on disk · 15.2 tok/s → **26.5 tok/s with this drafter (1.78×)** |
| **MLX-8bit** (max fidelity) | [`…-Multimodal-MLX-8bit`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit) | 29.5 GB · max fidelity; same drafter attaches |
| Base BF16 (source) | [`…-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) | the head this drafter was split out of |
| **Source of truth + toolkit** | [github.com/AEON-7/…-MLX](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX) | reproducible build + serve pipeline |

It is the same head regardless of target — the FP4 and 8-bit builds both keep `mtp.*` in BF16, so this one drafter is correct for both.

## 📋 Technical details

| Property | Value |
|---|---|
| Role | Speculative drafter (MTP). **Not a standalone model.** |
| `model_type` | `qwen3_5_mtp` |
| `block_size` | 3 |
| Footprint | 821 MB |
| Precision | BF16 (the head is never quantized in either target build) |
| Source head | `mtp.*` of [`…-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) |
| Engine | `mlx-vlm` (git `main`), `--draft-kind mtp` |
| Lossless | Yes — every proposed token verified by the target |

## 🙏 Provenance

- **Base (BF16):** [`AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16) — the `qwen3_5` hybrid (Gated-DeltaNet SSM + GQA self-attn) that ships the native MTP head.
- **Targets:** [`…-MLX-FP4`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4) · [`…-MLX-8bit`](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit).
- **Source of truth (GitHub):** [github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX).
- Split out + benchmarked by **AEON-7** on Apple Silicon (MacBook Pro M4 Pro, 48 GB) with `mlx-vlm`.

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

