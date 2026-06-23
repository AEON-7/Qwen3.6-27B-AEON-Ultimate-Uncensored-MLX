"""Rewrite the HF card frontmatter tag blocks: correct precision tags + enrich for search."""
import re
ROOT = "/Users/albert/qwen36-mlx/cards"

COMMON = ["mlx", "mlx-vlm", "apple-silicon", "metal", "mixed-precision", "quantized",
          "qwen", "qwen3", "qwen3.6", "qwen3_5", "multimodal", "vision", "image-text-to-text",
          "mtp", "speculative-decoding", "gated-deltanet", "mamba", "ssm", "linear-attention",
          "uncensored", "abliterated", "refusal-removed", "aeon", "aeon-7", "m4-pro", "on-device", "conversational"]
TAGS = {
    "hf_card_8bit.md": COMMON + ["8-bit", "8bit", "int8"],
    "hf_card_fp4.md":  COMMON + ["mxfp4", "fp4", "4-bit"],
    "hf_card_drafter.md": ["mlx", "mlx-vlm", "mtp", "speculative-decoding", "self-speculation",
                           "qwen3_5_mtp", "qwen3_5", "qwen3", "qwen3.6", "qwen", "draft-model",
                           "multi-token-prediction", "apple-silicon", "metal", "on-device", "aeon", "aeon-7"],
}

for fn, tags in TAGS.items():
    p = f"{ROOT}/{fn}"
    s = open(p).read()
    head, fm, body = s.split("---\n", 2)              # operate only on the frontmatter
    block = "tags:\n" + "".join(f"  - {t}\n" for t in tags)
    fm2 = re.sub(r"tags:\n(?:  - .*\n)+", block, fm, count=1)
    assert fm2 != fm, f"tags block not matched in {fn}"
    open(p, "w").write("---\n".join([head, fm2, body]))
    print(f"  {fn}: {len(tags)} tags")
print("done.")
