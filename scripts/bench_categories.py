"""
Per-prompt-category benchmark for the FP4 build: baseline vs MTP (block-size 2/3/4).
Captures decode tok/s, prefill tok/s, TTFT, TPOT, peak RAM, and MTP acceptance per category.
Load-once: FP4 stays resident; the 821 MB drafter is reloaded per (cat,bs) to reset accept stats.
Greedy (temp 0) for reproducibility + parity with the 1.78x headline.
"""
import json, re, os
from mlx_vlm import load, generate
from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
from mlx_vlm.speculative.utils import format_speculative_stats
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

ROOT = "/Users/albert/qwen36-mlx"
TARGET = f"{ROOT}/out/mxfp4-mixed"
DRAFTER = f"{ROOT}/mtp-drafter"
MAXTOK = 200

CATEGORIES = {
 "Code": "Implement an LRU cache class in Python with O(1) get and put, backed by a dict plus a doubly linked list. Include docstrings, type hints, and a short usage example.",
 "Math": "A tank fills at 3 liters per minute and simultaneously drains at 1.2 liters per minute. It starts empty and holds 90 liters. Work through it step by step and state how many minutes until it is full.",
 "Reasoning": "There are five houses in a row, each a different color, each owner drinking a different beverage and keeping a different pet. The green house is immediately right of the ivory house; milk is drunk in the middle house; the dog owner lives in the first house. Reason step by step about what can be deduced and which constraints remain.",
 "Creative": "Write the opening three paragraphs of a gothic horror short story set in an abandoned lighthouse during a violent storm. Use atmospheric, literary prose with vivid sensory detail.",
 "Knowledge": "Explain in depth how CRISPR-Cas9 gene editing works: the role of the guide RNA, the PAM site, how Cas9 creates a double-strand break, and how HDR versus NHEJ repair pathways lead to different editing outcomes.",
 "Chat": "I'm planning a five-day trip to Tokyo in spring. Give me a friendly, practical day-by-day itinerary covering neighborhoods, food to try, and a couple of lesser-known spots locals like.",
}

print(f"[load] {TARGET}", flush=True)
model, processor = load(TARGET)
config = load_config(TARGET)

def parse_accept(s):
    if not s: return None
    out = {}
    m = re.search(r"([\d.]+) accepted tokens/round", s); out["tok_per_round"] = float(m.group(1)) if m else None
    m = re.search(r"([\d.]+)% of drafted", s);            out["pct_drafted"] = float(m.group(1)) if m else None
    m = re.search(r"over (\d+) rounds", s);               out["rounds"] = int(m.group(1)) if m else None
    return out

def run(text, draft=None, kind=None, bs=None, mt=MAXTOK):
    p = apply_chat_template(processor, config, text, num_images=0)
    kw = dict(max_tokens=mt, temperature=0.0, verbose=False)
    if draft is not None:
        kw.update(draft_model=draft, draft_kind=kind, draft_block_size=bs)
    r = generate(model, processor, p, **kw)
    ttft = (r.prompt_tokens / r.prompt_tps * 1000.0) if r.prompt_tps else 0.0
    tpot = (1000.0 / r.generation_tps) if r.generation_tps else 0.0
    rec = dict(decode_tps=round(r.generation_tps, 2), prefill_tps=round(r.prompt_tps, 1),
               prompt_tokens=int(r.prompt_tokens), gen_tokens=int(r.generation_tokens),
               ttft_ms=round(ttft), tpot_ms=round(tpot, 2), peak_gb=round(r.peak_memory, 2))
    if draft is not None:
        rec["accept"] = parse_accept(format_speculative_stats(draft))
    return rec

print("[warmup]", flush=True)
run("Say hello in one sentence.", mt=16)
_d, _k = load_drafter(DRAFTER, kind="mtp"); run("Say hello in one sentence.", draft=_d, kind=_k, bs=3, mt=16)

results = {}
for cat, prompt in CATEGORIES.items():
    base = run(prompt)
    mtp = {}
    for bs in [2, 3, 4]:
        draft, kind = load_drafter(DRAFTER, kind="mtp")   # fresh -> reset accept stats
        validate_drafter_compatibility(model, draft, kind)
        mtp[f"bs{bs}"] = run(prompt, draft=draft, kind=kind, bs=bs)
    results[cat] = {"baseline": base, "mtp": mtp}
    best = max(mtp.items(), key=lambda kv: kv[1]["decode_tps"])
    print(f"  {cat:10} base {base['decode_tps']:5.1f} tok/s  TTFT {base['ttft_ms']:4.0f}ms  TPOT {base['tpot_ms']:5.2f}ms  |  "
          f"best MTP {best[0]} {best[1]['decode_tps']:5.1f} tok/s ({best[1]['decode_tps']/base['decode_tps']:.2f}x, "
          f"{best[1]['accept']['pct_drafted']}% acc)", flush=True)

json.dump(results, open(f"{ROOT}/research/bench_categories.json", "w"), indent=2)
print("DONE -> research/bench_categories.json", flush=True)
