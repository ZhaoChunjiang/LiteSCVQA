"""
LiteSCVQA - Step 2: quality regression + evaluation
轻量级无参考屏幕内容视频质量评价 —— 第二步：质量回归与评价

Dependencies (CPU only):  pip install numpy scipy scikit-learn pandas

Inputs
    features.csv  from litescvqa_features.py   (column "video", "key_ratio", "seconds" + 66 feature columns)
    mos.csv       subjective scores            (columns: video, mos [, group])
                  "group" = source/reference content id. Videos from the same source never
                  appear in both train and test. If omitted, group = first folder of "video"
                  (matches the MSU layout: <source_id>/<distorted>.mkv).

Modes
    python litescvqa_eval.py within  -f feats.csv -m mos.csv              # repeated content-split
    python litescvqa_eval.py within  -f feats.csv -m mos.csv --ablation   # + feature-group ablation
    python litescvqa_eval.py within  -f feats.csv -m mos.csv --regressors svr,rf,gbr,mlp
    python litescvqa_eval.py cross   -f A.csv -m A_mos.csv -F B.csv -M B_mos.csv   # train A, test B
    python litescvqa_eval.py train   -f feats.csv -m mos.csv -o model.joblib       # final model
    python litescvqa_eval.py predict -f new_feats.csv --model model.joblib          # score new videos

Protocol (standard in VQA papers)
    * N random splits (default 100), 80% sources train / 20% test
    * hyper-parameters chosen by 5-fold GroupKFold on the TRAIN part only
    * report median (and std) of SRCC, KRCC, PLCC, RMSE over splits
    * PLCC / RMSE after the 4-parameter logistic mapping (VQEG practice)
"""
import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

META_COLS = {"video", "seconds", "mos", "group"}

# Feature groups for ablation (prefix match on column names)
FEATURE_GROUPS = {
    "text":      lambda c: c.startswith("txt_"),
    "pictorial": lambda c: c.startswith("pic_"),
    "global":    lambda c: c.startswith("glob_") or c.startswith("ratio_"),
    "temporal":  lambda c: c.split("_")[0] in ("stall", "static", "motion", "switch"),
    "pool_p10p90": lambda c: c.endswith("_p10") or c.endswith("_p90"),
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load(feat_csv, mos_csv=None):
    F = pd.read_csv(feat_csv)
    if mos_csv is None:
        return F, None
    M = pd.read_csv(mos_csv)
    if not {"video", "mos"} <= set(M.columns):
        sys.exit("mos.csv needs columns: video, mos [, group]")
    D = F.merge(M, on="video", how="inner")
    if len(D) < len(M):
        print(f"warning: {len(M) - len(D)} MOS rows have no features (names must match 'video')")
    if "group" not in D.columns:
        D["group"] = D["video"].str.split("/").str[0]
        if D["group"].nunique() == len(D):
            print("warning: no 'group' column and no folders -> every video is its own group; "
                  "content may leak between train and test")
    return D, M


def feature_cols(D, drop_groups=()):
    cols = [c for c in D.columns if c not in META_COLS]
    for g in drop_groups:
        cols = [c for c in cols if not FEATURE_GROUPS[g](c)]
    return cols


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def logistic4(x, b1, b2, b3, b4):
    return (b1 - b2) / (1 + np.exp(-(x - b3) / (np.abs(b4) + 1e-9))) + b2


def evaluate(y, p):
    srcc = stats.spearmanr(y, p).correlation
    krcc = stats.kendalltau(y, p).correlation
    try:
        p0 = [y.max(), y.min(), np.mean(p), np.std(p) + 1e-6]
        popt, _ = curve_fit(logistic4, p, y, p0=p0, maxfev=20000)
        pm = logistic4(p, *popt)
    except Exception:
        pm = p
    plcc = stats.pearsonr(y, pm)[0]
    rmse = float(np.sqrt(np.mean((y - pm) ** 2)))
    return {"SRCC": srcc, "KRCC": krcc, "PLCC": plcc, "RMSE": rmse}


# ----------------------------------------------------------------------------
# Regressors (all CPU-light)
# ----------------------------------------------------------------------------
def make_model(name):
    pre = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    if name == "svr":
        est = SVR(kernel="rbf")
        grid = {"m__C": [2 ** k for k in range(-2, 9, 2)], "m__gamma": ["scale", 1e-3, 1e-2, 1e-1]}
    elif name == "rf":
        est = RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=0)
        grid = {"m__max_depth": [None, 8, 16], "m__min_samples_leaf": [1, 3]}
    elif name == "gbr":
        est = GradientBoostingRegressor(random_state=0)
        grid = {"m__n_estimators": [100, 300], "m__max_depth": [2, 3], "m__learning_rate": [0.05, 0.1]}
    elif name == "mlp":
        est = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True, random_state=0)
        grid = {"m__alpha": [1e-4, 1e-3, 1e-2]}
    else:
        raise ValueError(name)
    return Pipeline(pre + [("m", est)]), grid


def fit_tuned(name, X, y, g, inner_folds=5):
    model, grid = make_model(name)
    k = min(inner_folds, len(np.unique(g)))
    if k < 2:
        return model.fit(X, y)
    gs = GridSearchCV(model, grid, cv=GroupKFold(n_splits=k), scoring="neg_mean_squared_error", n_jobs=1)
    gs.fit(X, y, groups=g)
    return gs.best_estimator_


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def run_within(D, cols, regressor, n_splits, test_size, seed=0, verbose=True):
    X = D[cols].to_numpy(float)
    y = D["mos"].to_numpy(float)
    g = D["group"].astype(str).to_numpy()
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    rows = []
    t0 = time.time()
    for i, (tr, te) in enumerate(gss.split(X, y, g)):
        m = fit_tuned(regressor, X[tr], y[tr], g[tr])
        rows.append(evaluate(y[te], m.predict(X[te])))
        if verbose and (i + 1) % max(1, n_splits // 10) == 0:
            r = pd.DataFrame(rows).median()
            print(f"  split {i+1}/{n_splits}  median SRCC={r.SRCC:.4f} PLCC={r.PLCC:.4f}  "
                  f"({time.time()-t0:.0f}s)")
    R = pd.DataFrame(rows)
    return R.median(), R.std()


def fmt(med, std):
    return "  ".join(f"{k}={med[k]:.4f}±{std[k]:.4f}" for k in ["SRCC", "KRCC", "PLCC", "RMSE"])


def cmd_within(a):
    D, _ = load(a.features, a.mos)
    print(f"{len(D)} videos, {D['group'].nunique()} source contents")
    results = []
    for reg in a.regressors.split(","):
        cols = feature_cols(D)
        print(f"\n[{reg}] all features ({len(cols)})")
        med, std = run_within(D, cols, reg, a.splits, a.test_size)
        print(f"[{reg}] {fmt(med, std)}")
        results.append({"setting": f"{reg} / all", **med.to_dict()})
    if a.ablation:
        reg = a.regressors.split(",")[0]
        for grp in FEATURE_GROUPS:
            cols = feature_cols(D, drop_groups=[grp])
            if not cols:
                continue
            print(f"\n[{reg}] without {grp} ({len(cols)} features)")
            med, std = run_within(D, cols, reg, a.splits, a.test_size, verbose=False)
            print(f"[{reg}] w/o {grp}: {fmt(med, std)}")
            results.append({"setting": f"{reg} / w/o {grp}", **med.to_dict()})
    T = pd.DataFrame(results)
    print("\n=== Summary (median over splits) ===")
    print(T.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if a.out:
        T.to_csv(a.out, index=False)
        print(f"saved {a.out}")


def cmd_cross(a):
    A, _ = load(a.features, a.mos)
    B, _ = load(a.features_b, a.mos_b)
    cols = [c for c in feature_cols(A) if c in B.columns]
    m = fit_tuned(a.regressors.split(",")[0], A[cols].to_numpy(float), A["mos"].to_numpy(float),
                  A["group"].astype(str).to_numpy())
    r = evaluate(B["mos"].to_numpy(float), m.predict(B[cols].to_numpy(float)))
    print(f"train {a.features} -> test {a.features_b}: " +
          "  ".join(f"{k}={v:.4f}" for k, v in r.items()))


def cmd_train(a):
    import joblib
    D, _ = load(a.features, a.mos)
    cols = feature_cols(D)
    m = fit_tuned(a.regressors.split(",")[0], D[cols].to_numpy(float), D["mos"].to_numpy(float),
                  D["group"].astype(str).to_numpy())
    joblib.dump({"model": m, "cols": cols}, a.out or "litescvqa_model.joblib")
    print(f"saved model with {len(cols)} features to {a.out or 'litescvqa_model.joblib'}")


def cmd_predict(a):
    import joblib
    pack = joblib.load(a.model)
    F, _ = load(a.features)
    F["pred_quality"] = pack["model"].predict(F[pack["cols"]].to_numpy(float))
    out = a.out or "predictions.csv"
    F[["video", "pred_quality"]].to_csv(out, index=False)
    print(F[["video", "pred_quality"]].to_string(index=False))
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser(description="LiteSCVQA regression / evaluation")
    ap.add_argument("mode", choices=["within", "cross", "train", "predict"])
    ap.add_argument("-f", "--features", required=True)
    ap.add_argument("-m", "--mos")
    ap.add_argument("-F", "--features-b")
    ap.add_argument("-M", "--mos-b")
    ap.add_argument("--regressors", default="svr", help="comma list of svr,rf,gbr,mlp")
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    {"within": cmd_within, "cross": cmd_cross, "train": cmd_train, "predict": cmd_predict}[a.mode](a)


if __name__ == "__main__":
    main()
