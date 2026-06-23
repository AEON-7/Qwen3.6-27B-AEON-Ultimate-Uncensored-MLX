"""
Per-category SVG charts from research/bench_categories.json (FP4 build, MTP sweet spot bs=3).
Two dark cards: (1) decode tok/s baseline vs MTP per category, (2) TTFT + MTP acceptance per category.
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets")
data = json.load(open(os.path.join(ROOT, "research", "bench_categories.json")))
CATS = list(data.keys())

BG, GRID, TXT, MUT = "#0d1117", "#30363d", "#e6edf3", "#8b949e"
CF, CM, CA = "#fbbf24", "#34d399", "#818cf8"   # baseline amber · MTP emerald · accent indigo
FONT = "system-ui,-apple-system,Segoe UI,Roboto,sans-serif"
BS = "bs3"  # the chosen MTP sweet spot

def frame(w, h, title, sub):
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">'
            f'<rect width="{w}" height="{h}" rx="14" fill="{BG}"/>'
            f'<rect x="1" y="1" width="{w-2}" height="{h-2}" rx="13" fill="none" stroke="{GRID}"/>'
            f'<text x="28" y="40" fill="{TXT}" font-size="21" font-weight="700">{title}</text>'
            f'<text x="28" y="63" fill="{MUT}" font-size="12.5">{sub}</text>')

def legend(x, y, items):
    out = []
    for label, col in items:
        out.append(f'<rect x="{x}" y="{y-9}" width="11" height="11" rx="2" fill="{col}"/>')
        out.append(f'<text x="{x+16}" y="{y}" fill="{MUT}" font-size="11.5">{label}</text>')
        x += 22 + len(label) * 7
    return "".join(out)

# ---- Chart 1: decode tok/s baseline vs MTP(bs3) per category ----
def chart_tps():
    w, h = 820, 440
    s = [frame(w, h, "Per-category decode throughput — baseline vs MTP",
               "FP4 · MTP block-size 3 (lossless) · decode tok/s, single stream · M4 Pro 48 GB")]
    s.append(legend(540, 40, [("baseline", CF), ("+ MTP bs3", CM)]))
    bx, by, bw, bh = 70, 110, 720, 250
    vmax = 30
    for i in range(4):
        gy = by + bh - bh * i / 3
        s.append(f'<line x1="{bx}" y1="{gy:.0f}" x2="{bx+bw}" y2="{gy:.0f}" stroke="{GRID}"/>')
        s.append(f'<text x="{bx-8}" y="{gy+4:.0f}" fill="{MUT}" font-size="10" text-anchor="end">{vmax*i/3:.0f}</text>')
    n = len(CATS); slot = bw / n
    for i, cat in enumerate(CATS):
        base = data[cat]["baseline"]["decode_tps"]
        mtp = data[cat]["mtp"][BS]["decode_tps"]
        acc = data[cat]["mtp"][BS].get("accept", {}).get("pct_drafted")
        cx = bx + slot * i + slot / 2
        bw2 = slot * 0.30
        for j, (val, col) in enumerate([(base, CF), (mtp, CM)]):
            x = cx - bw2 * (1.05 if j == 0 else -0.05) - bw2*0.0
            x = cx + (j - 0.5) * (bw2 + 5) - bw2/2
            bh2 = bh * val / vmax
            s.append(f'<rect x="{x:.1f}" y="{by+bh-bh2:.1f}" width="{bw2:.1f}" height="{bh2:.1f}" rx="3" fill="{col}"/>')
            s.append(f'<text x="{x+bw2/2:.1f}" y="{by+bh-bh2-6:.1f}" fill="{TXT}" font-size="11.5" font-weight="600" text-anchor="middle">{val:.1f}</text>')
        spd = mtp / base if base else 0
        s.append(f'<text x="{cx:.1f}" y="{by+bh+20:.0f}" fill="{TXT}" font-size="12.5" text-anchor="middle" font-weight="600">{cat}</text>')
        lab = f"{spd:.2f}x" + (f" · {acc:.0f}% acc" if acc else "")
        s.append(f'<text x="{cx:.1f}" y="{by+bh+37:.0f}" fill="{CM}" font-size="10.5" text-anchor="middle">{lab}</text>')
    s.append(f'<text x="{bx}" y="{by-12}" fill="{MUT}" font-size="10">tok/s</text>')
    s.append('</svg>')
    return "".join(s)

# ---- Chart 2: TTFT (bars) + MTP acceptance (line) per category ----
def chart_latency():
    w, h = 820, 440
    s = [frame(w, h, "Per-category TTFT &amp; MTP draft acceptance",
               "FP4 · TTFT = prefill latency (varies with prompt length) · acceptance drives MTP speedup")]
    s.append(legend(560, 40, [("TTFT (ms)", CA), ("accept %", CM)]))
    bx, by, bw, bh = 70, 110, 720, 250
    ttfts = [data[c]["baseline"]["ttft_ms"] for c in CATS]
    tmax = max(ttfts) * 1.25
    for i in range(4):
        gy = by + bh - bh * i / 3
        s.append(f'<line x1="{bx}" y1="{gy:.0f}" x2="{bx+bw}" y2="{gy:.0f}" stroke="{GRID}"/>')
        s.append(f'<text x="{bx-8}" y="{gy+4:.0f}" fill="{MUT}" font-size="10" text-anchor="end">{tmax*i/3:.0f}</text>')
    n = len(CATS); slot = bw / n
    def cx(i): return bx + slot * i + slot / 2
    for i, cat in enumerate(CATS):
        t = data[cat]["baseline"]["ttft_ms"]
        bw2 = slot * 0.34
        bh2 = bh * t / tmax
        s.append(f'<rect x="{cx(i)-bw2/2:.1f}" y="{by+bh-bh2:.1f}" width="{bw2:.1f}" height="{bh2:.1f}" rx="3" fill="{CA}" opacity="0.85"/>')
        s.append(f'<text x="{cx(i):.1f}" y="{by+bh-bh2-6:.1f}" fill="{TXT}" font-size="11" text-anchor="middle">{t:.0f}</text>')
        s.append(f'<text x="{cx(i):.1f}" y="{by+bh+20:.0f}" fill="{TXT}" font-size="12.5" text-anchor="middle" font-weight="600">{cat}</text>')
    # acceptance line (right axis 0-100%)
    pts = []
    for i, cat in enumerate(CATS):
        acc = data[cat]["mtp"][BS].get("accept", {}).get("pct_drafted") or 0
        pts.append((cx(i), by + bh - bh * acc / 100.0))
    path = " ".join(f'{"M" if i==0 else "L"}{x:.1f},{y:.1f}' for i, (x, y) in enumerate(pts))
    s.append(f'<path d="{path}" fill="none" stroke="{CM}" stroke-width="2.5"/>')
    for i, cat in enumerate(CATS):
        acc = data[cat]["mtp"][BS].get("accept", {}).get("pct_drafted") or 0
        x, y = pts[i]
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{CM}" stroke="{BG}" stroke-width="2"/>')
        s.append(f'<text x="{x:.1f}" y="{y-10:.1f}" fill="{CM}" font-size="10.5" text-anchor="middle">{acc:.0f}%</text>')
    s.append(f'<text x="{bx}" y="{by-12}" fill="{MUT}" font-size="10">ms / %</text>')
    s.append('</svg>')
    return "".join(s)

for name, fn in [("bench_categories_tps.svg", chart_tps), ("bench_categories_latency.svg", chart_latency)]:
    open(os.path.join(OUT, name), "w").write(fn())
    print("wrote assets/" + name)
