"""
Upload the finalized model cards as README.md to each HF model repo (weights already uploaded).
Reads HF_TOKEN from env.
"""
import os
from huggingface_hub import HfApi
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
ROOT = "/Users/albert/qwen36-mlx"

MAP = [
    ("AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-8bit", "cards/hf_card_8bit.md"),
    ("AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-MLX-FP4",  "cards/hf_card_fp4.md"),
    ("AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX-MTP-Drafter",     "cards/hf_card_drafter.md"),
]
for repo, card in MAP:
    # charts first (so the relative assets/ paths in the card resolve on HF)
    api.upload_folder(
        folder_path=f"{ROOT}/assets", path_in_repo="assets",
        repo_id=repo, repo_type="model", allow_patterns=["*.svg", "*.png"],
        commit_message="Add benchmark charts",
    )
    api.upload_file(
        path_or_fileobj=f"{ROOT}/{card}", path_in_repo="README.md",
        repo_id=repo, repo_type="model", commit_message="Add model card",
    )
    print(f"  card + assets -> https://huggingface.co/{repo}")
print("done.")
