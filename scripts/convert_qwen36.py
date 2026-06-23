"""
Generalized mlx_vlm.convert driver for the Qwen3.6-27B-AEON MLX builds.
Picks a recipe module (exports `pred`) + the matching quant globals.

The callable predicate REPLACES mlx_vlm.convert's base predicate (so the recipe does its own
skip filtering). convert() loads lazy + donates quantized leaves -> memory-safe, no full-model
float residency. dtype=bfloat16 keeps the preserved (skipped) tensors in BF16.
"""
import argparse, os, sys, time, importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--hf-path", required=True, help="local BF16 source dir")
ap.add_argument("--mlx-path", required=True, help="output dir")
ap.add_argument("--recipe", required=True, help="recipe module in scripts/ exporting pred (e.g. recipe_8bit)")
ap.add_argument("--q-mode", default="mxfp4")
ap.add_argument("--q-bits", type=int, default=4)
ap.add_argument("--q-group-size", type=int, default=32)
args = ap.parse_args()

pred = importlib.import_module(args.recipe).pred
from mlx_vlm import convert  # noqa: E402

t0 = time.time()
print(f"[convert] {args.hf_path} -> {args.mlx_path}", flush=True)
print(f"[convert] recipe={args.recipe} globals={args.q_mode}/{args.q_bits}b/gs{args.q_group_size} dtype=bfloat16", flush=True)
convert(
    hf_path=args.hf_path,
    mlx_path=args.mlx_path,
    quantize=True,
    q_mode=args.q_mode,
    q_bits=args.q_bits,
    q_group_size=args.q_group_size,
    dtype="bfloat16",
    trust_remote_code=True,
    quant_predicate=pred,
)
print(f"[convert] DONE in {(time.time()-t0)/60:.1f} min -> {args.mlx_path}", flush=True)
