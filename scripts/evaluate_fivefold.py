#!/usr/bin/env python3
"""Reproduce five-fold source-disjoint evaluation from lawful MSU-SCV feature tables.

Each of the 100 source contents appears exactly once as test data.
This program does not require raw video files when extracted feature tables
and corresponding MOS labels are already available.
"""
import argparse
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='msu_feats/b_*.csv')
    ap.add_argument('--mos', default='msu/mos.csv')
    ap.add_argument('--folds', default='protocols/five_fold_groups.json')
    ap.add_argument('--out', default='fivefold_oof_predictions.csv')
    args = ap.parse_args()
    paths = sorted(glob.glob(args.features))
    if not paths:
        raise SystemExit('No feature files matched ' + args.features)
    F = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).drop_duplicates('video')
    M = pd.read_csv(args.mos)[['video','mos']]
    D = F.merge(M, on='video', validate='one_to_one')
    if len(D) != len(F) or len(D) != len(M):
        raise SystemExit('Feature/MOS video IDs do not match one-to-one')
    D['group'] = D.video.str.split('/').str[0].astype(int)
    columns = [c for c in F.columns if c not in ('video', 'seconds')]
    if len(columns) != 67:
        raise SystemExit(f'Expected 67 input features, got {len(columns)}')
    folds = json.loads(Path(args.folds).read_text(encoding='utf-8'))['groups_by_fold']
    groups = [int(x) for fold in folds for x in fold]
    if len(groups) != len(set(groups)) or set(groups) != set(D.group.unique()):
        raise SystemExit('Fold source IDs are duplicated or do not match features')
    X = D[columns].to_numpy(float)
    y = D.mos.to_numpy(float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise SystemExit('Nonfinite feature values or MOS')
    pred = np.full(len(D), np.nan)
    for i, fold in enumerate(folds, start=1):
        test = D.group.isin(fold).to_numpy()
        model = GradientBoostingRegressor(n_estimators=300, max_depth=3,
                 learning_rate=0.05, subsample=0.8, random_state=0)
        pred[test] = model.fit(X[~test], y[~test]).predict(X[test])
        print(f'Fold {i}: {test.sum()} test videos from {len(fold)} source contents', flush=True)
    out = D[['video','group','mos']].copy()
    out['pred_quality'] = pred
    out.to_csv(args.out, index=False)
    scores = [float(spearmanr(g.pred_quality, g.mos).statistic)
              for _,g in out.groupby('group', sort=True)]
    print('Mean per-content SRCC:', round(float(np.mean(scores)), 6))
    print('Median per-content SRCC:', round(float(np.median(scores)), 6))
    print('Saved', args.out)

if __name__ == '__main__':
    main()
