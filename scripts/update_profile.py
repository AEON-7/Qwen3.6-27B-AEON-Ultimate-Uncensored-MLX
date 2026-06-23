"""Add the Qwen3.6 MLX repo to the AEON-7 GitHub profile README (Qwen 3.6 + Apple Silicon MLX sections)."""
import json, base64, urllib.request, os
TOKEN = os.environ["GHCR_PAT"]
URL = "https://api.github.com/repos/AEON-7/AEON-7/contents/README.md"

def gh(method, data=None):
    req = urllib.request.Request(URL, data=json.dumps(data).encode() if data else None, method=method)
    req.add_header("Authorization", f"token {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    return json.load(urllib.request.urlopen(req))

info = gh("GET")
sha = info["sha"]
content = base64.b64decode(info["content"]).decode()

REPS = [
    # broaden the Apple Silicon MLX intro
    ("Metal-accelerated MLX builds of abliterated Gemma 4 —",
     "Metal-accelerated MLX builds of abliterated Gemma 4 and Qwen 3.6 —"),
    # Apple Silicon MLX — new table row after the Gemma toolkit row
    ("gemma4-aeon-abliterated-mlx-toolkit?style=flat&label=) |",
     "gemma4-aeon-abliterated-mlx-toolkit?style=flat&label=) |\n"
     "| **[Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX)** | "
     "The flagship Qwen 3.6 27B comes to the Mac. Native MLX/Metal quants of the AEON Ultimate Uncensored line: "
     "**MLX-8bit (29.5 GB)** max-fidelity flagship + compact **MLX-FP4 (16 GB)** for 24 GB Macs, plus a native "
     "**MTP drafter** for up to **1.78x lossless** self-speculation. Selective quant keeps the Gated-DeltaNet SSM, "
     "vision tower, and MTP head in BF16 — fully multimodal. Per-category benchmarks + 0-to-hero `uv` quickstart. | "
     "![](https://img.shields.io/github/stars/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX?style=flat&label=) |"),
    # Apple Silicon MLX — new "grab the weights" line after the Gemma one
    ("[🤗 K4-BF16 source](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16)",
     "[🤗 K4-BF16 source](https://huggingface.co/AEON-7/Gemma-4-12B-it-AEON-Abliterated-K4-BF16)\n\n"
     "**Qwen 3.6 MLX weights:** [🤗 MLX-8bit](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit) · "
     "[🤗 MLX-FP4](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4) · "
     "[🤗 MTP-Drafter](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter)"),
    # Qwen 3.6 Models — new table row after the DDTree row
    ("Qwen3.6-27B-AEON-Ultimate-Uncensored-DDTree?style=flat&label=) |",
     "Qwen3.6-27B-AEON-Ultimate-Uncensored-DDTree?style=flat&label=) |\n"
     "| **[Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX](https://github.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX)** | "
     "Qwen 3.6 27B AEON Ultimate Uncensored | Dense · 🍎 Apple Silicon MLX | "
     "**The on-device Mac build.** Native MLX/Metal quants — MLX-8bit (29.5 GB) + MLX-FP4 (16 GB, 24 GB Macs) + a native "
     "MTP drafter for up to **1.78x lossless** self-speculation. Selective quant preserves the Gated-DeltaNet SSM, vision "
     "tower, and MTP head in BF16. Per-category benchmarks + 0-to-hero quickstart. "
     "[🤗 weights](https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4) | "
     "![](https://img.shields.io/github/stars/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX?style=flat&label=) |"),
]

for a, b in REPS:
    assert content.count(a) >= 1, f"ANCHOR NOT FOUND: {a[:60]!r}"
    content = content.replace(a, b, 1)

gh("PUT", {
    "message": "Add Qwen3.6-27B MLX (Apple Silicon) — feature in Qwen 3.6 + Apple Silicon MLX sections",
    "content": base64.b64encode(content.encode()).decode(),
    "sha": sha,
})
print("✓ profile README updated (4 edits applied)")
