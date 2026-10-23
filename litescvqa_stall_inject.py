"""
LiteSCVQA - stall injection & stall-detection validation (Section 4.7)
卡顿注入与卡顿检测验证脚本

Simulates network stalls in clean screen-content videos and measures how well the
stall rule of LiteSCVQA (Eq. 7) recovers them, and how often it fires falsely on the
original (stall-free) videos, e.g. on long static slides.

A stall at frame t0 with length L is simulated the way a real-time stream behaves:
frames t0+1 .. t0+L are replaced by a copy of frame t0 (freeze), and playback resumes
at frame t0+L+1, so the content "jumps" to catch up. Stalls are only placed where the
content is moving; a freeze on a static slide is invisible and has no ground truth.

Dependencies: numpy scipy opencv-python-headless  (+ ffmpeg if --x264 is used)
Put this file next to litescvqa_features.py.

Usage
  # 1) inject stalls into every video of a folder of clean/reference videos
  python litescvqa_stall_inject.py inject refs/ -o stalled/ --per-video 3 --seed 0 [--x264 18]
  #    -> stalled/*.mp4 + stalled/ground_truth.csv
  # 2) evaluate detector: true positives on stalled/, false alarms on refs/
  python litescvqa_stall_inject.py evaluate stalled/ --refs refs/
"""
import os
import csv
import argparse
import subprocess

import numpy as np
import cv2

import litescvqa_features as lf

EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def list_videos(folder):
    return sorted(os.path.join(r, f) for r, _, fs in os.walk(folder)
                  for f in fs if f.lower().endswith(EXTS))


def read_all(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def write_video(frames, fps, out, x264_crf=None):
    h, w = frames[0].shape[:2]
    if x264_crf is None:   # OpenCV writer, no extra dependency
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
           "-crf", str(x264_crf), "-pix_fmt", "yuv420p", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


# ----------------------------------------------------------------------------
# Injection
# ----------------------------------------------------------------------------
def moving_mask(path):
    small, fps = lf.read_gray_small(path)
    d = lf.frame_differences(small)
    return d >= lf.STATIC_THR, fps


def plan_stalls(moving, fps, k, rng, min_len=2, max_sec=2.0, gap_sec=1.0):
    """Choose up to k non-overlapping (t0, L) with content moving right before and after."""
    n = len(moving)
    max_len = max(min_len, int(max_sec * fps))
    gap = int(gap_sec * fps)
    taken = np.zeros(n, bool)
    plans = []
    for _ in range(200 * k):
        if len(plans) >= k:
            break
        L = int(rng.integers(min_len, max_len + 1))
        t0 = int(rng.integers(2, n - L - 2))
        lo, hi = max(0, t0 - gap), min(n, t0 + L + 1 + gap)
        if taken[lo:hi].any():
            continue
        # content moving before the stall and moving across the resume point
        if not (moving[t0] or moving[t0 - 1]):
            continue
        if not moving[t0 + 1:t0 + L + 2].any():
            continue
        taken[lo:hi] = True
        plans.append((t0, L))
    return sorted(plans)


def inject(frames, plans):
    out = list(frames)
    for t0, L in plans:
        for j in range(1, L + 1):
            out[t0 + j] = frames[t0]
    return out


def stream_inject(src, plans, fps, out, x264_crf=None):
    """Streaming version (constant memory): frames t0+1..t0+L are replaced by frame t0."""
    freeze_at = {t0: L for t0, L in plans}
    cap = cv2.VideoCapture(src)
    writer, proc = None, None
    hold, left, t = None, 0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if left > 0:
            f, left = hold, left - 1
        elif t in freeze_at:
            hold, left = f, freeze_at[t]
        if writer is None and proc is None:
            h, w = f.shape[:2]
            if x264_crf is None:
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            else:
                cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                       "-crf", str(x264_crf), "-preset", "veryfast", "-pix_fmt", "yuv420p", out]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if writer is not None:
            writer.write(f)
        else:
            proc.stdin.write(f.tobytes())
        t += 1
    cap.release()
    if writer is not None:
        writer.release()
    if proc is not None:
        proc.stdin.close(); proc.wait()


def cmd_inject(args):
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    gt_rows = []
    for v in list_videos(args.refs):
        moving, fps = moving_mask(v)
        plans = plan_stalls(moving, fps, args.per_video, rng, max_sec=args.max_sec)
        if not plans:
            print(f"-- {v}: content too static, no stall injected")
            continue
        rel = os.path.splitext(os.path.relpath(v, args.refs))[0].replace(os.sep, "__")
        out = os.path.join(args.out, rel + "_stall.mp4")
        stream_inject(v, plans, fps, out, args.x264)
        for t0, L in plans:
            # frozen frames are t0+1 .. t0+L
            gt_rows.append([os.path.basename(out), t0 + 1, t0 + L, L, round(L / fps, 3)])
        print(f"++ {out}: {len(plans)} stalls {plans}", flush=True)
    with open(os.path.join(args.out, "ground_truth.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "start_frame", "end_frame", "length_frames", "length_sec"])
        w.writerows(gt_rows)
    print(f"ground truth: {len(gt_rows)} stalls -> {args.out}/ground_truth.csv")


# ----------------------------------------------------------------------------
# Detection (same rule as lf.temporal_features, Eq. 7) returning intervals
# ----------------------------------------------------------------------------
def detect_stalls(d, fps, use_before=True, use_jump=True, use_duration=True):
    n = len(d)
    static = d < lf.STATIC_THR
    moving = d[~static & (np.arange(n) > 0)]
    m = float(np.median(moving)) if moving.size else 0.0
    found = []
    t = 1
    while t < n:
        if static[t]:
            s = t
            while t < n and static[t]:
                t += 1
            L = t - s
            jump = d[t] if t < n else 0.0
            before = s >= 2 and not static[s - 1]
            ok = L >= 2
            if use_duration:
                ok &= L <= 2 * fps
            if use_before:
                ok &= before
            if use_jump:
                ok &= m > 0 and jump > lf.STALL_JUMP * m
            if ok:
                found.append((s, t - 1))
        t += 1
    return found


VARIANTS = {   # for ablation Table 8
    "(a) repeated frames only": dict(use_before=False, use_jump=False, use_duration=False),
    "(b) + duration":          dict(use_before=False, use_jump=False, use_duration=True),
    "(c) + motion before":     dict(use_before=True, use_jump=False, use_duration=True),
    "(d) full rule (Eq. 7)":   dict(use_before=True, use_jump=True, use_duration=True),
}


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def match(gt, det):
    """A detection is a hit if it overlaps a ground-truth stall by >= 50% of the shorter one."""
    used = set()
    tp = 0
    for g in gt:
        for i, q in enumerate(det):
            if i in used:
                continue
            if overlap(g, q) >= 0.5 * min(g[1] - g[0] + 1, q[1] - q[0] + 1):
                used.add(i)
                tp += 1
                break
    return tp, len(det) - len(used), len(gt) - tp


def diff_signal(path):
    small, fps = lf.read_gray_small(path)
    return lf.frame_differences(small), fps, len(small) / fps


def cmd_evaluate(args):
    gt = {}
    with open(os.path.join(args.stalled, "ground_truth.csv")) as fh:
        for r in csv.DictReader(fh):
            gt.setdefault(r["video"], []).append((int(r["start_frame"]), int(r["end_frame"])))
    sig = {os.path.basename(v): diff_signal(v) for v in list_videos(args.stalled)}
    ref_sig = [diff_signal(v) for v in list_videos(args.refs)] if args.refs else []
    ref_minutes = sum(s[2] for s in ref_sig) / 60 if ref_sig else 0
    rows = []
    print(f"{'variant':28s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'prec':>6s} {'recall':>6s} {'F1':>6s}"
          + ("  false/min on refs" if ref_sig else ""))
    for name, kw in VARIANTS.items():
        TP = FP = FN = 0
        for vid, g in gt.items():
            if vid not in sig:
                continue
            d, fps, _ = sig[vid]
            tp, fp, fn = match(g, detect_stalls(d, fps, **kw))
            TP, FP, FN = TP + tp, FP + fp, FN + fn
        p = TP / (TP + FP) if TP + FP else 0.0
        r = TP / (TP + FN) if TP + FN else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        far = (sum(len(detect_stalls(d, fps, **kw)) for d, fps, _ in ref_sig) / ref_minutes
               if ref_sig else float("nan"))
        rows.append([name, TP, FP, FN, round(p, 4), round(r, 4), round(f1, 4), round(far, 3)])
        print(f"{name:28s} {TP:5d} {FP:5d} {FN:5d} {p:6.3f} {r:6.3f} {f1:6.3f}"
              + (f"  {far:8.3f}" if ref_sig else ""))
    out = args.out or os.path.join(args.stalled, "stall_detection_results.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "TP", "FP", "FN", "precision", "recall", "F1", "false_alarms_per_min_refs"])
        w.writerows(rows)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description="LiteSCVQA stall injection / validation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("inject")
    a.add_argument("refs"); a.add_argument("-o", "--out", required=True)
    a.add_argument("--per-video", type=int, default=3)
    a.add_argument("--max-sec", type=float, default=2.0)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--x264", type=int, default=None, help="encode with libx264 at this CRF (needs ffmpeg)")
    b = sub.add_parser("evaluate")
    b.add_argument("stalled"); b.add_argument("--refs", default=None)
    b.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    cmd_inject(args) if args.cmd == "inject" else cmd_evaluate(args)


if __name__ == "__main__":
    main()
