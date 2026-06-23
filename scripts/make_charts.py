"""
Pure-SVG benchmark charts for the Qwen3.6-27B-AEON MLX builds (no matplotlib).
Dark, theme-agnostic cards that render crisply on GitHub + HF.
Data: measured on MacBook Pro M4 Pro (48 GB), mlx-vlm 0.6.x.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(OUT, exist_ok=True)

# ---- palette ---------------------------------------------------------------
BG, PANEL, GRID = "#0d1117", "#161b22", "#30363d"
TXT, MUT = "#e6edf3", "#8b949e"
C8, CF, CM = "#818cf8", "#fbbf24", "#34d399"   # 8bit indigo · fp4 amber · fp4+mtp emerald
FONT = "system-ui,-apple-system,Segoe UI,Roboto,sans-serif"

def frame(w, h, title, sub):
    return (
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" font-family="{FONT}">'
        f'<rect width="{w}" height="{h}" rx="14" fill="{BG}"/>'
        f'<rect x="1" y="1" width="{w-2}" height="{h-2}" rx="13" fill="none" stroke="{GRID}"/>'
        f'<text x="28" y="42" fill="{TXT}" font-size="22" font-weight="700">{title}</text>'
        f'<text x="28" y="66" fill="{MUT}" font-size="13">{sub}</text>'
    )

# ============================================================ Chart 1: throughput + memory
def chart1():
    w, h = 760, 430
    s = [frame(w, h, "Decode throughput &amp; memory footprint",
               "tokens/sec (single stream) · peak unified memory · M4 Pro 48 GB")]
    builds = [("MLX-8bit", 8.2, 29.9, C8), ("MLX-FP4", 15.2, 17.1, CF), ("FP4 + MTP", 26.5, 18.7, CM)]
    # left panel: tok/s ; right panel: GB
    def panel(x0, w0, title, vals, vmax, unit, fmt):
        out = [f'<text x="{x0}" y="108" fill="{MUT}" font-size="12" font-weight="600">{title}</text>']
        bx, by, bw, bh = x0, 130, w0, 230
        for i in range(5):  # gridlines
            gy = by + bh - bh * i / 4
            out.append(f'<line x1="{bx}" y1="{gy:.0f}" x2="{bx+bw}" y2="{gy:.0f}" stroke="{GRID}" stroke-width="1"/>')
            out.append(f'<text x="{bx-8}" y="{gy+4:.0f}" fill="{MUT}" font-size="10" text-anchor="end">{vmax*i/4:.0f}</text>')
        n = len(vals); slot = bw / n
        for i, (name, v, col) in enumerate(vals):
            barw = slot * 0.5
            cx = bx + slot * i + slot / 2
            bh2 = bh * v / vmax
            out.append(f'<rect x="{cx-barw/2:.1f}" y="{by+bh-bh2:.1f}" width="{barw:.1f}" height="{bh2:.1f}" rx="4" fill="{col}"/>')
            out.append(f'<text x="{cx:.1f}" y="{by+bh-bh2-8:.1f}" fill="{TXT}" font-size="14" font-weight="700" text-anchor="middle">{fmt(v)}</text>')
            out.append(f'<text x="{cx:.1f}" y="{by+bh+18:.0f}" fill="{MUT}" font-size="11" text-anchor="middle">{name}</text>')
        out.append(f'<text x="{bx+bw}" y="{by+bh+18:.0f}" fill="{MUT}" font-size="10" text-anchor="end">{unit}</text>')
        return "".join(out)
    s.append(panel(70, 280, "DECODE SPEED", [(b[0], b[1], b[3]) for b in builds], 28, "tok/s", lambda v: f"{v:.1f}"))
    s.append(panel(440, 280, "PEAK MEMORY", [(b[0], b[2], b[3]) for b in builds], 32, "GB", lambda v: f"{v:.1f}"))
    s.append('</svg>')
    return "".join(s)

# ============================================================ Chart 2: MTP sweep
def chart2():
    w, h = 760, 430
    s = [frame(w, h, "MTP self-speculation — block-size sweep (FP4)",
               "native qwen3_5_mtp drafter · lossless (every token verified) · M4 Pro")]
    pts = [("baseline", 14.9, None), ("bs=2", 23.5, 97.3), ("bs=3", 26.5, 94.7), ("bs=4", 25.4, 86.9)]
    bx, by, bw, bh = 80, 150, 600, 210
    vmax = 30
    for i in range(4):
        gy = by + bh - bh * i / 3
        s.append(f'<line x1="{bx}" y1="{gy:.0f}" x2="{bx+bw}" y2="{gy:.0f}" stroke="{GRID}"/>')
        s.append(f'<text x="{bx-8}" y="{gy+4:.0f}" fill="{MUT}" font-size="10" text-anchor="end">{vmax*i/3:.0f}</text>')
    n = len(pts); slot = bw / n
    def cx(i): return bx + slot * i + slot / 2
    def cy(v): return by + bh - bh * v / vmax
    # line
    path = " ".join(f'{"M" if i==0 else "L"}{cx(i):.1f},{cy(p[1]):.1f}' for i, p in enumerate(pts))
    s.append(f'<path d="{path}" fill="none" stroke="{CM}" stroke-width="3"/>')
    for i, (name, v, acc) in enumerate(pts):
        peak = (name == "bs=3")
        col = CM if peak else CF
        r = 9 if peak else 6
        s.append(f'<circle cx="{cx(i):.1f}" cy="{cy(v):.1f}" r="{r}" fill="{col}" stroke="{BG}" stroke-width="2"/>')
        s.append(f'<text x="{cx(i):.1f}" y="{cy(v)-16:.1f}" fill="{TXT}" font-size="15" font-weight="700" text-anchor="middle">{v:.1f}</text>')
        spd = "" if acc is None else f"×{v/14.9:.2f} · {acc:.0f}% acc"
        s.append(f'<text x="{cx(i):.1f}" y="{by+bh+22:.0f}" fill="{TXT}" font-size="12" text-anchor="middle" font-weight="600">{name}</text>')
        s.append(f'<text x="{cx(i):.1f}" y="{by+bh+39:.0f}" fill="{MUT}" font-size="10.5" text-anchor="middle">{spd}</text>')
    s.append(f'<text x="{bx+bw}" y="{by-12}" fill="{MUT}" font-size="10" text-anchor="end">tok/s</text>')
    # peak callout
    s.append(f'<rect x="{cx(2)-66:.0f}" y="{cy(26.5)-58:.0f}" width="132" height="24" rx="12" fill="{CM}" opacity="0.16"/>')
    s.append(f'<text x="{cx(2):.0f}" y="{cy(26.5)-42:.0f}" fill="{CM}" font-size="11.5" font-weight="700" text-anchor="middle">1.78× · sweet spot</text>')
    s.append('</svg>')
    return "".join(s)

# ============================================================ Chart 3: build selection map
def chart3():
    w, h = 760, 440
    s = [frame(w, h, "Pick your build — speed vs footprint",
               "bubble = effective precision (bits/weight) · top-left wins · M4 Pro 48 GB")]
    bx, by, bw, bh = 80, 110, 600, 250
    xmin, xmax = 14, 32   # GB
    ymin, ymax = 0, 30    # tok/s
    def X(gb): return bx + bw * (gb - xmin) / (xmax - xmin)
    def Y(t): return by + bh - bh * (t - ymin) / (ymax - ymin)
    for i in range(5):
        gy = by + bh * i / 4
        s.append(f'<line x1="{bx}" y1="{gy:.0f}" x2="{bx+bw}" y2="{gy:.0f}" stroke="{GRID}"/>')
        s.append(f'<text x="{bx-8}" y="{gy+4:.0f}" fill="{MUT}" font-size="10" text-anchor="end">{ymax-ymax*i/4:.0f}</text>')
    for i in range(5):
        gx = bx + bw * i / 4
        s.append(f'<text x="{gx:.0f}" y="{by+bh+20:.0f}" fill="{MUT}" font-size="10" text-anchor="middle">{xmin+(xmax-xmin)*i/4:.0f}</text>')
    s.append(f'<text x="{bx+bw/2:.0f}" y="{by+bh+40:.0f}" fill="{MUT}" font-size="11" text-anchor="middle">peak memory (GB) →</text>')
    s.append(f'<text x="34" y="{by+bh/2:.0f}" fill="{MUT}" font-size="11" text-anchor="middle" transform="rotate(-90 34 {by+bh/2:.0f})">decode tok/s →</text>')
    pts = [("MLX-8bit", 29.9, 8.2, 8.63, C8, "max fidelity"),
           ("MLX-FP4", 17.1, 15.2, 4.88, CF, "compact"),
           ("FP4 + MTP bs3", 18.7, 26.5, 4.88, CM, "fastest")]
    for name, gb, t, bpw, col, tag in pts:
        r = 10 + bpw * 1.7
        s.append(f'<circle cx="{X(gb):.1f}" cy="{Y(t):.1f}" r="{r:.1f}" fill="{col}" opacity="0.22"/>')
        s.append(f'<circle cx="{X(gb):.1f}" cy="{Y(t):.1f}" r="5" fill="{col}"/>')
        dy = -r - 10 if name != "MLX-FP4" else r + 20
        s.append(f'<text x="{X(gb):.1f}" y="{Y(t)+dy:.1f}" fill="{TXT}" font-size="13" font-weight="700" text-anchor="middle">{name}</text>')
        s.append(f'<text x="{X(gb):.1f}" y="{Y(t)+dy+15:.1f}" fill="{MUT}" font-size="10.5" text-anchor="middle">{bpw:.2f} bpw · {tag}</text>')
    s.append('</svg>')
    return "".join(s)

for name, fn in [("bench_throughput_memory.svg", chart1), ("bench_mtp_sweep.svg", chart2), ("bench_build_map.svg", chart3)]:
    p = os.path.join(OUT, name)
    open(p, "w").write(fn())
    print("wrote", p, f"({os.path.getsize(p)} bytes)")
