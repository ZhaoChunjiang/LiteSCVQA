#!/usr/bin/env python3
"""Check that MSU-SCV feature tables align with the quality labels.

Validates metadata and the 67 predictor columns expected by the published
batch-export format. Does not reproduce video decoding or quality scores.
"""
import argparse
from glob import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def validate(feature_pattern, mos_file, expected=1600):
    paths = sorted(glob(feature_pattern))
    if not paths:
        raise ValueError(f"No feature CSV files matched: {feature_pattern}")
    frames = [pd.read_csv(p) for p in paths]
    data = pd.concat(frames, ignore_index=True)
    scores = pd.read_csv(mos_file)
    needed = {"video", "key_ratio", "seconds"}
    if not needed <= set(data.columns):
        raise ValueError(f"Feature columns missing: {sorted(needed - set(data.columns))}")
    if not {"video", "mos"} <= set(scores.columns):
        raise ValueError("MOS CSV must have video and mos columns")
    if data["video"].isna().any() or scores["video"].isna().any():
        raise ValueError("Null video IDs found")
    if data["video"].duplicated().any() or scores["video"].duplicated().any():
        raise ValueError("Duplicate video IDs found")
    predictors = [c for c in data.columns if c not in {"video", "seconds"}]
    if len(predictors) != 67:
        raise ValueError(f"Expected 67 predictors (including key_ratio), found {len(predictors)}")
    if any(not pd.api.types.is_numeric_dtype(data[c]) for c in predictors):
        raise ValueError("Non-numeric predictor columns found")
    if not np.isfinite(data[predictors].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite predictor values")
    if not np.isfinite(pd.to_numeric(scores["mos"], errors="coerce").to_numpy()).all():
        raise ValueError("Non-finite MOS scores")
    a, b = set(data["video"]), set(scores["video"])
    if a != b:
        raise ValueError(f"Unmatched files: features-only={len(a-b)}, MOS-only={len(b-a)}")
    n = len(data)
    if expected is not None and n != expected:
        raise ValueError(f"Expected {expected} rows, observed {n}")
    groups = data["video"].astype(str).str.split("/").str[0]
    counts = groups.value_counts()
    if expected == 1600 and (len(counts) != 100 or (counts != 16).any()):
        raise ValueError(f"Expected 100 source contents x 16 distortions, got {len(counts)} groups")
    print(f"PASS: {n} matched videos; {len(counts)} source contents; 67 numeric predictors")
    print(f"Source variants per content: {counts.min()}..{counts.max()}")
    print(f"Key-frame ratio: {data['key_ratio'].min():.5f}..{data['key_ratio'].max():.5f}")
    return True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, help="Quoted path glob, e.g. 'msu_feats/b_*.csv'")
    p.add_argument("--mos", required=True, help="Score CSV with video and mos columns")
    p.add_argument("--expected", type=int, default=1600, help="Row count; set 0 to allow other datasets")
    args = p.parse_args()
    try:
        validate(args.features, args.mos, None if args.expected == 0 else args.expected)
    except (ValueError, OSError, KeyError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
