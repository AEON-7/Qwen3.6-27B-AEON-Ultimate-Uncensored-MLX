"""
AEON-7 · Qwen3.6-27B-AEON-Ultimate-Uncensored · MLX 8-bit (affine) — near-lossless flagship.
BUILD A. Selective quant: 8-bit affine (gs64) on the compressible bulk; BF16 on the
Gated-DeltaNet / Mamba state dynamics, the vision tower, and the MTP head.

Source-verified recipe (research wf_ddb435ea-371) against the installed mlx_vlm qwen3_5 graph.
A custom callable REPLACES mlx_vlm.convert's base predicate, so this must do its own skip
filtering. Match = substring on the sanitized leaf path; first match wins.

PRESERVE in BF16 — quantizing these destroys the hybrid SSM:
  linear_attn.conv1d, A_log, dt_bias, every *norm*,
  linear_attn.in_proj_a / in_proj_b  (the decay-gate `g` + the β sigmoid — learned GDN dynamics),
  the vision tower (model.visual.*), and the MTP head (mtp.*).
QUANTIZE at 8-bit affine gs64 (near-lossless — per-group fp16 scale+bias keeps residual-writer
range): mlp.{gate,up,down}_proj, linear_attn.{in_proj_qkv,in_proj_z,out_proj},
  self_attn.{q,k,v,o}_proj, embed_tokens, lm_head.   (~90.6% of params → ≈31.6 GB)
"""

# Keep BF16 (do NOT quantize) — order-independent substrings.
SKIP = (
    "linear_attn.conv1d",
    "A_log", "dt_bias",
    "norm",                                              # all RMSNorm (q_norm/k_norm/linear_attn.norm/model.norm)
    "linear_attn.in_proj_a", "linear_attn.in_proj_b",    # GDN decay-gate + beta dynamics — must stay BF16
    "visual", "vision_tower",                            # multimodal vision tower
    "mtp.",                                              # MTP head (dropped by sanitize; explicit for fork-safety)
)

Q8 = {"group_size": 64, "bits": 8, "mode": "affine"}


def pred(path, module):
    """False -> bf16 ; dict -> to_quantized(**dict). First match wins."""
    if not hasattr(module, "to_quantized"):
        return False
    if any(s in path for s in SKIP):
        return False
    return dict(Q8)
