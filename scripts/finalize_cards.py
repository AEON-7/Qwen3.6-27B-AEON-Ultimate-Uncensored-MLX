"""
Splice the verbatim Arbitration Clause + Support footer into the authored docs,
replacing the <!--ARBITRATION_CLAUSE--> / <!--SUPPORT_FOOTER--> markers.
Reports marker counts so a missing marker is obvious.
"""
import os
ROOT = "/Users/albert/qwen36-mlx"
clause = open(f"{ROOT}/cards/_clause.md").read().strip()
support = open(f"{ROOT}/cards/_support.md").read().strip()
# normalize chart references to relative paths (render on GitHub + HF, public or private)
RAW = "https://raw.githubusercontent.com/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-MLX/main/assets/"

FILES = ["README.md", "AGENTS.md",
         "cards/hf_card_8bit.md", "cards/hf_card_fp4.md", "cards/hf_card_drafter.md"]

for f in FILES:
    p = f"{ROOT}/{f}"
    if not os.path.exists(p):
        print(f"  MISSING {f}")
        continue
    s = open(p).read()
    s = s.replace(RAW, "assets/")
    nc, ns = s.count("<!--ARBITRATION_CLAUSE-->"), s.count("<!--SUPPORT_FOOTER-->")
    s = s.replace("<!--ARBITRATION_CLAUSE-->", "\n---\n\n" + clause + "\n")
    s = s.replace("<!--SUPPORT_FOOTER-->", "\n---\n\n" + support + "\n")
    open(p, "w").write(s)
    flag = "" if (nc and ns) or f == "AGENTS.md" else "  <-- CHECK markers"
    print(f"  {f}: clause x{nc}, support x{ns}, {len(s.splitlines())} lines{flag}")
print("done.")
