"""
Upload the 3 MLX artifacts' WEIGHTS to private HF repos (cards added separately later).
Reads HF_TOKEN from env. Sequential (8bit -> FP4 -> drafter). README is added in a later pass.
"""
import os, time
from huggingface_hub import HfApi, create_repo

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
ROOT = "/Users/albert/qwen36-mlx"

JOBS = [
    ("AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit", f"{ROOT}/out/mlx8bit"),
    ("AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4",  f"{ROOT}/out/mxfp4-mixed"),
    ("AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter",     f"{ROOT}/mtp-drafter"),
]

for repo, d in JOBS:
    t0 = time.time()
    print(f"[upload] {d} -> {repo} (private)", flush=True)
    create_repo(repo, private=True, exist_ok=True, repo_type="model", token=token)
    api.upload_folder(
        folder_path=d, repo_id=repo, repo_type="model",
        ignore_patterns=[".cache/*", "*.lock", "README.md"],
        commit_message="Add MLX weights — Apple Silicon (M4 Pro validated)",
    )
    print(f"[upload] DONE {repo} in {(time.time()-t0)/60:.1f} min", flush=True)

print("[upload] ALL WEIGHTS UPLOADED", flush=True)
