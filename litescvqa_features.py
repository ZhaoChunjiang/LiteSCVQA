"""
LiteSCVQA - Step 1: key-frame selection + feature extraction
轻量级无参考屏幕内容视频质量评价 —— 第一步：关键帧选取与特征提取

Dependencies (CPU only):  pip install numpy scipy opencv-python-headless
Usage:
    python litescvqa_features.py video.mp4              # print features of one video
    python litescvqa_features.py videos_dir/ -o feats.csv   # batch (recursive), write CSV
    python litescvqa_features.py dataset/ -o feats.csv --skip-orig   # skip reference orig.* files

Modules
    M1  Content-change-driven key-frame selection   (select_key_frames)
    M2  Text/graphic vs pictorial block partition    (partition_blocks)
    M3  Spatial quality features                     (spatial_features)
    M4  Temporal stall / flicker features            (temporal_features)
    Pooling                                          (extract_video_features)
"""
import os
import sys
import csv
import glob
import argparse
import time

import numpy as np
import cv2
from scipy.special import gamma

# ----------------------------------------------------------------------------
# Parameters (tune on the training split only)
# ----------------------------------------------------------------------------
ANALYSIS_WIDTH = 480        # width used for frame-difference analysis
PIX_THR = 0.05              # a pixel "changed" if |diff| > 0.05 (on 0-1 scale)
STATIC_THR = 1e-4           # changed-pixel fraction below which two frames are "identical"
CHANGE_THR = 0.003          # changed-pixel fraction for a content change (page turn, scroll) ...
CHANGE_K = 8.0              # ... and above median + CHANGE_K * MAD (robust to compression noise)
STALL_JUMP = 3.0            # ...and the jump after it > STALL_JUMP x typical motion (catch-up); 1.2 -> 3.0 after handwriting test 2026-10-16, re-tune on training data
FRAMES_PER_STABLE = 1       # key frames taken from each stable segment
CHANGE_STRIDE = 3           # sampling stride inside change segments
MAX_KEY_FRAMES = 40         # hard cap for speed
FEATURE_WIDTH = 1280        # key frames are resized to this width for spatial features
BLOCK = 32                  # block size for region partition


# ----------------------------------------------------------------------------
# Video reading
# ----------------------------------------------------------------------------
def read_gray_small(path, width=ANALYSIS_WIDTH):
    """Read every frame once as a small grayscale image (cheap pass)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = width / w
        g = cv2.cvtColor(cv2.resize(f, (width, int(round(h * s))), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        frames.append(g.astype(np.float32) / 255.0)
    cap.release()
    return frames, fps


def read_frames_at(path, indices, width=FEATURE_WIDTH):
    """Second pass: decode only the selected key frames at higher resolution (BGR uint8)."""
    want = set(indices)
    out = {}
    cap = cv2.VideoCapture(path)
    i = 0
    last = max(indices) if indices else -1
    while i <= last:
        ok = cap.grab()
        if not ok:
            break
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                h, w = f.shape[:2]
                if w > width:
                    s = width / w
                    f = cv2.resize(f, (width, int(round(h * s))), interpolation=cv2.INTER_AREA)
                out[i] = f
        i += 1
    cap.release()
    return [out[k] for k in indices if k in out]


# ----------------------------------------------------------------------------
# M1  Key-frame selection
# ----------------------------------------------------------------------------
def frame_differences(frames):
    """
    Changed-pixel fraction between consecutive frames, d[t] = mean(|F_t - F_{t-1}| > PIX_THR), d[0]=0.
    Screen content is sparse (thin text on flat background), so a page turn changes only ~0.5% of
    pixels; the fraction of changed pixels is far more discriminative than the mean difference.
    """
    d = np.zeros(len(frames), dtype=np.float64)
    for t in range(1, len(frames)):
        d[t] = float(np.mean(np.abs(frames[t] - frames[t - 1]) > PIX_THR))
    return d


def change_threshold(d):
    if len(d) < 3:
        return CHANGE_THR
    x = d[1:]
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return max(CHANGE_THR, med + CHANGE_K * mad)


def segment_video(d):
    """Split the timeline into stable / change segments. Returns list of (start, end, kind)."""
    segs = []
    n = len(d)
    if n == 0:
        return segs
    thr = change_threshold(d)
    kind = lambda v: "change" if v > thr else "stable"
    start, cur = 0, kind(d[0])
    for t in range(1, n):
        k = kind(d[t])
        if k != cur:
            segs.append((start, t - 1, cur))
            start, cur = t, k
    segs.append((start, n - 1, cur))
    return segs


def select_key_frames(d, max_frames=MAX_KEY_FRAMES):
    """Pick representative frames: middle of each stable segment + strided frames in change segments."""
    segs = segment_video(d)
    idx = []
    for s, e, k in segs:
        if k == "stable":
            L = e - s + 1
            for j in range(FRAMES_PER_STABLE):
                idx.append(s + int((j + 0.5) * L / FRAMES_PER_STABLE))
        else:
            idx.extend(range(s, e + 1, CHANGE_STRIDE))
    idx = sorted(set(idx))
    if len(idx) > max_frames:   # uniform thinning if still too many
        keep = np.linspace(0, len(idx) - 1, max_frames).round().astype(int)
        idx = [idx[k] for k in keep]
    return idx, segs


# ----------------------------------------------------------------------------
# M2  Region partition (text/graphic vs pictorial)
# ----------------------------------------------------------------------------
def partition_blocks(gray_u8, block=BLOCK):
    """
    Classify each block as text/graphic (1), pictorial (2) or flat background (0).
    Heuristics: screen text/graphics -> high edge density, few distinct gray levels,
    bimodal histogram; pictorial -> many gray levels, moderate gradients.
    Returns label map (H//block, W//block).
    """
    edges = cv2.Canny(gray_u8, 50, 150)
    H, W = gray_u8.shape
    hb, wb = H // block, W // block
    lab = np.zeros((hb, wb), np.uint8)
    for i in range(hb):
        for j in range(wb):
            g = gray_u8[i*block:(i+1)*block, j*block:(j+1)*block]
            e = edges[i*block:(i+1)*block, j*block:(j+1)*block]
            if g.std() < 2:
                continue                      # flat background
            ed = e.mean() / 255.0
            hist = np.bincount((g >> 3).ravel(), minlength=32)
            bimodal = np.sort(hist)[-2:].sum() / g.size   # mass of the 2 dominant levels
            if ed > 0.04 and bimodal > 0.6:
                lab[i, j] = 1                 # text / graphics: sharp edges on few colours
            else:
                lab[i, j] = 2                 # pictorial: smooth, many levels
    return lab


def block_mask(lab, shape, value, block=BLOCK):
    m = (lab == value).astype(np.uint8)
    m = cv2.resize(m, (lab.shape[1] * block, lab.shape[0] * block), interpolation=cv2.INTER_NEAREST)
    full = np.zeros(shape, np.uint8)
    full[:m.shape[0], :m.shape[1]] = m
    return full.astype(bool)


# ----------------------------------------------------------------------------
# M3  Spatial features
# ----------------------------------------------------------------------------
def _ggd_fit(x):
    """Moment-matching fit of a generalized Gaussian: returns (alpha shape, sigma^2)."""
    x = x[np.isfinite(x)]
    if x.size < 16:
        return 0.0, 0.0
    alphas = np.arange(0.2, 10.0, 0.001)
    r_gam = gamma(1 / alphas) * gamma(3 / alphas) / gamma(2 / alphas) ** 2
    s2 = np.mean(x ** 2)
    e = np.mean(np.abs(x))
    rho = s2 / (e ** 2 + 1e-12)
    return float(alphas[np.argmin((r_gam - rho) ** 2)]), float(s2)


def mscn(gray_f):
    mu = cv2.GaussianBlur(gray_f, (7, 7), 7 / 6)
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(gray_f * gray_f, (7, 7), 7 / 6) - mu * mu))
    return (gray_f - mu) / (sigma + 1.0 / 255)


def text_region_features(gray_f, bgr, mask):
    """Text/graphic readability features (computed only inside mask)."""
    if mask.sum() < 256:
        return [0.0] * 8
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy)
    g_in = gm[mask]
    # 1-2 gradient sharpness
    f_gmean = float(g_in.mean())
    f_gp90 = float(np.percentile(g_in, 90))
    # 3 edge width (blur): ratio of strong-edge pixels after vs before dilation-free threshold
    edges = cv2.Canny((gray_f * 255).astype(np.uint8), 50, 150) > 0
    e_in = edges & mask
    strong = gm > 0.5 * gm[mask].max() if g_in.size else gm > 0
    f_edge_width = float((gm[e_in] > 0).sum() / (np.sum(strong & mask) + 1))
    # 4 ringing: high-frequency energy in a thin band around edges but not on them
    band = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ring_zone = band & ~cv2.dilate(edges.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool) & mask
    lap = np.abs(cv2.Laplacian(gray_f, cv2.CV_32F))
    f_ringing = float(lap[ring_zone].mean()) if ring_zone.any() else 0.0
    # 5 blockiness on 8x8 grid (horizontal + vertical boundary discontinuity ratio)
    dh = np.abs(np.diff(gray_f, axis=1))
    dv = np.abs(np.diff(gray_f, axis=0))
    bh = dh[:, 7::8].mean() / (dh.mean() + 1e-6)
    bv = dv[7::8, :].mean() / (dv.mean() + 1e-6)
    f_block = float((bh + bv) / 2)
    # 6 chroma bleeding: chroma gradient near luma edges vs luma gradient
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32) / 255.0
    cg = sum(np.abs(cv2.Sobel(ycc[..., c], cv2.CV_32F, 1, 0, ksize=3)) +
             np.abs(cv2.Sobel(ycc[..., c], cv2.CV_32F, 0, 1, ksize=3)) for c in (1, 2))
    f_chroma = float(cg[band & mask].mean() / (gm[band & mask].mean() + 1e-6)) if (band & mask).any() else 0.0
    # 7-8 contrast between text strokes and background
    v = gray_f[mask]
    f_contrast = float(np.percentile(v, 95) - np.percentile(v, 5))
    f_edge_density = float(e_in.sum() / mask.sum())
    return [f_gmean, f_gp90, f_edge_width, f_ringing, f_block, f_chroma, f_contrast, f_edge_density]


def pictorial_features(gray_f, mask):
    """NSS (BRISQUE-style) features of pictorial region: GGD of MSCN + paired products."""
    if mask.sum() < 1024:
        return [0.0] * 6
    m = mscn(gray_f)
    a, s2 = _ggd_fit(m[mask])
    feats = [a, s2]
    for sh in [(0, 1), (1, 0)]:
        p = m * np.roll(m, sh, axis=(0, 1))
        a2, s22 = _ggd_fit(p[mask])
        feats += [a2, s22]
    return feats


def global_features(bgr, gray_f):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
    lap_var = float(cv2.Laplacian(gray_f, cv2.CV_32F).var())   # global sharpness
    return [float(gray_f.std()), float(hsv[..., 1].mean()), lap_var]


def spatial_features(bgr):
    gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray_u8.astype(np.float32) / 255.0
    lab = partition_blocks(gray_u8)
    tmask = block_mask(lab, gray_u8.shape, 1)
    pmask = block_mask(lab, gray_u8.shape, 2)
    n = lab.size
    ratios = [float((lab == 1).sum() / n), float((lab == 2).sum() / n)]
    return (text_region_features(gray_f, bgr, tmask) + pictorial_features(gray_f, pmask)
            + global_features(bgr, gray_f) + ratios)


SPATIAL_NAMES = (["txt_grad_mean", "txt_grad_p90", "txt_edge_width", "txt_ringing", "txt_blockiness",
                  "txt_chroma_bleed", "txt_contrast", "txt_edge_density"]
                 + ["pic_mscn_alpha", "pic_mscn_var", "pic_h_alpha", "pic_h_var", "pic_v_alpha", "pic_v_var"]
                 + ["glob_contrast", "glob_saturation", "glob_lap_var"]
                 + ["ratio_text", "ratio_pictorial"])


# ----------------------------------------------------------------------------
# M4  Temporal features: stall vs. normal static content
# ----------------------------------------------------------------------------
def temporal_features(d, fps):
    """
    Key idea: a slide that is legitimately static gives a long run of ~0 diffs that ends with a
    single page change; a stall/frame drop gives short repeated-frame runs that end with an
    abnormally large jump (motion that 'catches up'). We describe the joint statistics of
    (repeated-run length, jump after the run).
    """
    n = len(d)
    if n < 3:
        return [0.0] * 8
    dur = n / fps
    static = d < STATIC_THR
    runs = []          # (run_length_frames, jump_after, moving_before)
    t = 1
    while t < n:
        if static[t]:
            s = t
            while t < n and static[t]:
                t += 1
            jump = d[t] if t < n else 0.0
            before = s >= 2 and not static[s - 1]
            runs.append((t - s, jump, before))
        t += 1
    moving = d[~static & (np.arange(n) > 0)]
    med_move = float(np.median(moving)) if moving.size else 0.0
    # A stall = content was moving, froze for 2 frames .. 2 s, then moved again with a catch-up jump.
    # A legitimately static slide is usually long and is not sandwiched between motion.
    stalls = [(L, j) for L, j, b in runs
              if b and 2 <= L <= 2 * fps and med_move > 0 and j > STALL_JUMP * med_move]
    f_stall_rate = len(stalls) / dur                         # stalls per second
    f_stall_time = sum(L for L, _ in stalls) / n             # fraction of time stalled
    f_stall_maxlen = max([L for L, _ in stalls], default=0) / fps
    f_jump_ratio = float(np.mean([j / (med_move + 1e-6) for _, j in stalls])) if stalls else 0.0
    f_static_frac = float(static[1:].mean())
    f_motion_med = med_move
    f_motion_cv = float(moving.std() / (moving.mean() + 1e-9)) if moving.size else 0.0
    f_switch_rate = float(np.sum(d > change_threshold(d)) / dur)      # page changes per second
    return [f_stall_rate, f_stall_time, f_stall_maxlen, f_jump_ratio,
            f_static_frac, f_motion_med, f_motion_cv, f_switch_rate]


TEMPORAL_NAMES = ["stall_rate", "stall_time_frac", "stall_max_sec", "stall_jump_ratio",
                  "static_frac", "motion_median", "motion_cv", "switch_rate"]


# ----------------------------------------------------------------------------
# Pooling
# ----------------------------------------------------------------------------
def extract_video_features(path, verbose=False):
    t0 = time.time()
    small, fps = read_gray_small(path)
    d = frame_differences(small)
    keys, segs = select_key_frames(d)
    frames = read_frames_at(path, keys)
    S = np.array([spatial_features(f) for f in frames]) if frames else np.zeros((1, len(SPATIAL_NAMES)))
    # temporal pooling of spatial features: mean + worst-10% (human sensitivity to worst parts)
    mean = S.mean(axis=0)
    k = max(1, int(np.ceil(0.1 * len(S))))
    # "worse" direction differs per feature; use the 10th and 90th percentile and let the regressor decide
    p10 = np.percentile(S, 10, axis=0)
    p90 = np.percentile(S, 90, axis=0)
    # temporal fluctuation of text sharpness (bitrate switching)
    flick = float(S[:, 0].std() / (S[:, 0].mean() + 1e-9)) if len(S) > 1 else 0.0
    T = temporal_features(d, fps)
    feats = np.concatenate([mean, p10, p90, [flick], T])
    info = {"frames": len(small), "fps": fps, "key_frames": len(keys),
            "key_ratio": len(keys) / max(1, len(small)), "segments": len(segs),
            "seconds": round(time.time() - t0, 2)}
    if verbose:
        print(f"[{os.path.basename(path)}] {info}")
    return feats, info


def feature_names():
    return ([f"{n}_mean" for n in SPATIAL_NAMES] + [f"{n}_p10" for n in SPATIAL_NAMES]
            + [f"{n}_p90" for n in SPATIAL_NAMES] + ["txt_sharp_flicker"] + TEMPORAL_NAMES)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LiteSCVQA feature extraction")
    ap.add_argument("input", help="video file or directory")
    ap.add_argument("-o", "--out", help="CSV output (batch mode)")
    ap.add_argument("--skip-orig", action="store_true", help="skip reference files named orig.*")
    args = ap.parse_args()
    names = feature_names()
    if os.path.isdir(args.input):
        # recursive: works for flat folders and for "one folder per source video" layouts
        exts = (".mp4", ".mkv", ".avi", ".mov", ".webm")
        vids = sorted(os.path.join(r, f) for r, _, fs in os.walk(args.input)
                      for f in fs if f.lower().endswith(exts))
        if args.skip_orig:
            vids = [v for v in vids if not os.path.basename(v).lower().startswith("orig")]
        out = args.out or "features.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["video", "key_ratio", "seconds"] + names)
            for v in vids:
                try:
                    f, info = extract_video_features(v, verbose=True)
                except Exception as e:          # keep batch running on broken files
                    print(f"!! {v}: {e}")
                    continue
                vid = os.path.relpath(v, args.input).replace(os.sep, "/")
                w.writerow([vid, f"{info['key_ratio']:.4f}", info["seconds"]]
                           + [f"{x:.6g}" for x in f])
                fh.flush()
        print(f"wrote {len(vids)} rows to {out}")
    else:
        f, info = extract_video_features(args.input, verbose=True)
        for n, x in zip(names, f):
            print(f"{n:28s} {x:.5g}")


if __name__ == "__main__":
    main()
