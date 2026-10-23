import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig = plt.figure(figsize=(12, 5.9))
FX=lambda x,y,w,h:[x/120,(y+3)/59,w/120,h/59]
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 120); ax.set_ylim(-3, 56); ax.axis("off")

C_T = "#2f6db5"; C_S = "#d9822b"; C_R = "#3b8b5a"; C_G = "#6b6b6b"

def box(x, y, w, h, title, body="", fc="#ffffff", ec="#333", tc="#111", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.0",
                                fc=fc, ec=ec, lw=1.1))
    if body:
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=fs, weight="bold", color=tc)
        ax.text(x + w / 2, y + h - 4.6, body, ha="center", va="top", fontsize=7.2, color="#222", linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, weight="bold", color=tc)

def arrow(x1, y1, x2, y2, c="#333"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color=c, shrinkA=0, shrinkB=0))

def group(x, y, w, h, label, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                fc=c + "12", ec=c, lw=1.0, ls="--"))
    ax.text(x + 0.8, y + h - 0.6, label, ha="left", va="top", fontsize=8.5, weight="bold", color=c)

# ---------------- input video (stack of frames) ----------------
for k in range(4):
    x0, y0 = 2 + k * 0.9, 33 - k * 0.9
    ax.add_patch(Rectangle((x0, y0), 10, 7, fc="#f4f6fa", ec="#555", lw=0.8))
x0, y0 = 4.7, 30.3
for r in range(5):
    ax.add_patch(Rectangle((x0 + 1, y0 + 5.3 - r * 0.9), 4.5 - (r % 2) * 1.5, 0.35, fc="#333", ec="none"))
ax.add_patch(Rectangle((x0 + 6.2, y0 + 1.2), 3, 2.6, fc="#9cc3e6", ec="none"))
ax.text(9.7, 28.3, "Distorted SCV\n$V=\\{F_t\\}_{t=1}^{N}$", ha="center", va="top", fontsize=8)

# ---------------- Pass 1 ----------------
group(16, 30, 64, 24, "Pass 1: cheap full-sequence analysis (480 px, gray)", C_T)
box(18, 37, 11, 11, "Downscale", "width 480,\ngrayscale,\n[0, 1]")
box(31.5, 37, 14, 11, "Change signal", "(Eq. 1) fraction of\nchanged pixels $d_t$,\n$\\tau_p=0.05$")
# mini plot of d_t
pa = fig.add_axes(FX(48, 38, 15, 10.5))
rng = np.random.default_rng(3)
t = np.arange(150); d = np.abs(rng.normal(0, 0.0004, 150))
for p in (30, 75, 118): d[p] = 0.012 + rng.random() * 0.004
d[50:58] = 0; d[58] = 0.009                          # stall + catch-up
d[95:105] = 0.006 + rng.random(10) * 0.002          # scrolling
segs = [(0, 29, "s"), (30, 30, "c"), (31, 74, "s"), (75, 75, "c"), (76, 94, "s"), (95, 104, "c"), (105, 117, "s"), (118, 118, "c"), (119, 149, "s")]
for s, e, k in segs:
    pa.axvspan(s - 0.5, e + 0.5, color=("#cfe0f3" if k == "s" else "#f6c9a0"), lw=0)
pa.plot(t, d, color="#222", lw=0.8)
pa.axhline(0.003, color="#c0392b", lw=0.7, ls="--")
pa.text(149, 0.0034, "$\\tau_c$ (Eq. 2)", color="#c0392b", fontsize=6.5, ha="right", va="bottom")
for s, e, k in segs:
    if k == "s":
        pa.plot((s + e) / 2, -0.0012, marker="^", color=C_T, ms=4, clip_on=False)
for kf in range(95, 105, 3):
    pa.plot(kf, -0.0012, marker="^", color=C_S, ms=4, clip_on=False)
pa.annotate("stall", xy=(54, 0.0003), xytext=(48, 0.010), fontsize=6.5, color="#c0392b",
            arrowprops=dict(arrowstyle="->", lw=0.6, color="#c0392b"))
pa.set_ylim(-0.0005, 0.017); pa.set_xlim(0, 149)
pa.set_xticks([]); pa.set_yticks([])
pa.set_xlabel("frame $t$", fontsize=6.5, labelpad=9)
pa.set_ylabel("$d_t$", fontsize=7, labelpad=1)
for sp in pa.spines.values(): sp.set_linewidth(0.6)

box(65.5, 37, 13, 11, "Key frames", "centre of stable seg.\n+ stride 3 in change\nseg.; $K\\leq 40$", fc="#eef4fb")
ax.text(55.5, 33.0, "blue: stable   orange: change   \u25b2 key frame", fontsize=6.5, ha="center", color="#333")
arrow(14.5, 37, 18, 42.5)
arrow(29.3, 42.5, 31.5, 42.5)
arrow(45.8, 42.5, 47.6, 42.5)
arrow(63.2, 42.5, 65.5, 42.5)

# ---------------- Temporal module ----------------
group(16, 2, 40, 26, "Temporal module (from $d_t$ only)", C_S)
box(18, 14.5, 17, 10.5, "Frozen-run analysis", "static: $d_t<10^{-4}$\nrun length $L$, motion\nbefore, jump $J$ after")
box(37, 14.5, 17, 10.5, "Stall rule (Eq. 7)", "moving before,\n$2\\leq L\\leq 2r$,\n$J>\\gamma m$, $\\gamma=1.2$")
box(18, 3.5, 36, 8.5, "8 temporal features  $\\mathbf{t}$",
    "stall rate, stalled-time fraction, longest stall, mean jump,\nstatic fraction, median motion, motion CV, page-change rate", fc="#fdf1e6")
arrow(36, 37, 26.5, 25.3)
arrow(35, 19.8, 37, 19.8)
arrow(45.5, 14.5, 45.5, 12)

# ---------------- Pass 2 ----------------
group(60, 2, 40, 26, "Pass 2: key frames only (1280 px)", C_R)
box(62, 17.5, 21, 7.5, "Region partition", "32$\\times$32 blocks: text/graphic,\npictorial, flat")
# mini partition map
pm = fig.add_axes(FX(85.5, 17.3, 6, 7.5)); pm.axis("off")
lab = np.zeros((6, 10)); lab[0, 1:7] = 1; lab[1:5, 1:5] = 1; lab[2:5, 6:9] = 2; lab[5, 1:4] = 1
from matplotlib.colors import ListedColormap
pm.imshow(lab, cmap=ListedColormap(["#f2f2f2", C_T, "#7fb77e"]), interpolation="nearest")
ax.text(92.2, 22, "text", color=C_T, fontsize=6.5); ax.text(92.2, 19.6, "pictorial", color="#4c8b4b", fontsize=6.5)

box(62, 3.5, 11.5, 11.5, "Text (8)", "gradient mean/p90\nedge ratio, ringing\nblockiness, chroma\nbleeding, contrast,\nedge density", fc="#eef4fb", fs=8)
box(75, 3.5, 11, 11.5, "Pictorial (6)", "BRISQUE-style\nGGD of MSCN +\nH/V products", fc="#eef8ef", fs=8)
box(88, 3.5, 10.5, 11.5, "Global (5)", "contrast,\nsaturation,\nLaplacian var.,\ntext/pict. ratios", fc="#f3f3f3", fs=8)
arrow(67.75, 17.5, 67.75, 15.2); arrow(72.5, 17.5, 80.5, 15.2); arrow(77, 17.5, 90, 15.2)
ax.text(96.2, 31, "19-d $\\mathbf{s}_k$\nper key frame", fontsize=7, ha="right", color="#333")

# key frames -> pass 2
ax.annotate("", xy=(72.5, 25.3), xytext=(72, 36.8),
            arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333", connectionstyle="arc3,rad=0"))
ax.text(73, 30.2, "decode only\nkey frames", fontsize=6.8, ha="left", color="#333")

# ---------------- pooling + regression ----------------
box(83, 35.5, 16, 16, "Temporal pooling", "per spatial feature:\nmean, $P_{10}$, $P_{90}$\n(19$\\times$3 = 57)\n+ text-sharpness\nfluctuation CV (1)", fc="#f7f7f7")
arrow(97, 15.2, 97, 35.3)
box(103, 35.5, 15, 16, "Descriptor", "$\\mathbf{z}\\in\\mathbb{R}^{66}$\n(Eq. 8)\n57 + 1 + 8", fc="#fffbe6")
arrow(99.3, 43.5, 103, 43.5)
# temporal features -> descriptor
ax.plot([36, 36, 101, 101], [3.3, -1.0, -1.0, 40], color=C_S, lw=1.2)
arrow(101, 40, 103, 40, c=C_S)
ax.text(78, -1.4, "temporal features $\\mathbf{t}$ (8)", color=C_S, fontsize=7, ha="center", va="top")
box(103, 13, 15, 13, "Regression", "standardise\n(train stats)\nSVR-RBF\n(RF / GBR / MLP)", fc="#eef8ef")
# descriptor -> regression : route right side
ax.annotate("", xy=(116, 26.3), xytext=(116, 35.3), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333"))
box(103, 2.5, 15, 7, "Quality $\\hat q$", fc="#2f6db5", ec="#2f6db5", tc="white", fs=9.5)
arrow(110.5, 13, 110.5, 9.7)

# remove placeholder arrow drawn earlier by re-drawing white? simpler: arrow pass1 -> pooling is real
fig.savefig("fig1_framework.pdf"); fig.savefig("fig1_framework.png", dpi=220)
