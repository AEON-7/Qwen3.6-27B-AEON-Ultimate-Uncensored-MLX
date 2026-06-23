"""
AEON-7 · Qwen3.6-27B-AEON-Ultimate-Uncensored · MLX FP4 (mxfp4 + 8-bit islands) — compact.
BUILD B. mxfp4 (E2M1, gs32) on the bulk; 8-bit affine (gs64) islands on the GQA k/v projections
and the embedding/head; BF16 on the Gated-DeltaNet/Mamba state dynamics, vision tower, MTP head.

Source-verified recipe (research wf_ddb435ea-371). Match = substring on sanitized path; first wins.

PRESERVE in BF16 (quantizing destroys the hybrid SSM):
  linear_attn.conv1d, A_log, dt_bias, every *norm*, linear_attn.in_proj_a / in_proj_b,
  model.visual.* / vision_tower, mtp.*.
8-bit affine ISLANDS (sensitive at 4-bit): self_attn.k_proj, self_attn.v_proj  (GQA: only 4 KV
  heads → ~6x activation leverage, cheap to protect), embed_tokens, lm_head.
mxfp4 gs32 (the bulk): mlp.{gate,up,down}_proj, linear_attn.{in_proj_qkv,in_proj_z,out_proj},
  self_attn.{q,o}_proj.   (~90.6% of params quantized → ≈19.8 GB)
NOTE: mlp.down_proj is mxfp4 here; if KL regresses, promote it to 8-bit affine (the `-quality` build).
"""

SKIP = (
    "linear_attn.conv1d",
    "A_log", "dt_bias",
    "norm",
    "linear_attn.in_proj_a", "linear_attn.in_proj_b",
    "visual", "vision_tower",
    "mtp.",
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
