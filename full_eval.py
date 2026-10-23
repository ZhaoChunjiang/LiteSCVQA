"""Full MSU evaluation: content-disjoint protocols, 4 regressors, feature-group ablation."""
import pandas as pd, numpy as np, glob, sys, json, time, warnings
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')

F = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('msu_feats/b_*.csv'))]).drop_duplicates('video')
M = pd.read_csv('msu/mos.csv'); D = F.merge(M, on='video'); D['group'] = D.video.str.split('/').str[0].astype(int)
ALL = [c for c in F.columns if c not in ('video', 'seconds')]
X0 = D[ALL]; y = D.mos.values; g = D.group.values; G = np.unique(g)
print(f'{len(D)} videos, {len(G)} contents, {len(ALL)} features', flush=True)
print(f'feature extraction time per video: median {D.seconds.median():.2f}s, mean {D.seconds.mean():.2f}s', flush=True)

GROUPS = {
    'text': lambda c: c.startswith('txt_') and c != 'txt_sharp_flicker',
    'pictorial': lambda c: c.startswith('pic_'),
    'global': lambda c: c.startswith('glob_') or c.startswith('ratio_') or c == 'key_ratio',
    'temporal': lambda c: c in ('txt_sharp_flicker', 'static_frac', 'motion_median', 'motion_cv', 'switch_rate'),
    'stall': lambda c: c.startswith('stall_'),
}

def L4(x, b1, b2, b3, b4): return (b1 - b2) / (1 + np.exp(-(x - b3) / (abs(b4) + 1e-6))) + b2
def plcc(p, t):
    try:
        popt, _ = curve_fit(L4, p, t, p0=[t.max(), t.min(), np.median(p), np.std(p) + 1e-3], maxfev=20000)
        return pearsonr(L4(p, *popt), t)[0]
    except Exception:
        return pearsonr(p, t)[0]

MODELS = {
    'SVR': lambda: make_pipeline(StandardScaler(), SVR(C=10, gamma='scale')),
    'RF': lambda: RandomForestRegressor(300, n_jobs=-1, random_state=0),
    'GBR': lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0),
    'MLP': lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=2000, early_stopping=True, random_state=0)),
}

def splits(n, test_frac, seed=0):
    r = np.random.default_rng(seed)
    return [r.choice(G, int(round(test_frac * len(G))), replace=False) for _ in range(n)]

def run(mk, cols, sp):
    X = D[cols].values; R = {k: [] for k in ('SRCC', 'KRCC', 'PLCC', 'pcSRCC', 'pcKRCC')}
    for te in sp:
        m = np.isin(g, te); p = mk().fit(X[~m], y[~m]).predict(X[m]); t = y[m]; gg = g[m]
        R['SRCC'].append(spearmanr(p, t)[0]); R['KRCC'].append(kendalltau(p, t)[0]); R['PLCC'].append(plcc(p, t))
        R['pcSRCC'].append(np.nanmean([spearmanr(p[gg == k], t[gg == k])[0] for k in te]))
        R['pcKRCC'].append(np.nanmean([kendalltau(p[gg == k], t[gg == k])[0] for k in te]))
    return {k: (float(np.median(v)), float(np.std(v))) for k, v in R.items()}

def show(tag, r):
    print(tag + '  ' + '  '.join(f'{k} {m:.4f}±{s:.4f}' for k, (m, s) in r.items()), flush=True)

OUT = {}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SP = splits(N, 0.2)
print(f'\n== Protocol A: {N} random content-disjoint 80/20 splits ==', flush=True)
ONLY=sys.argv[2].split(',') if len(sys.argv)>2 else list(MODELS)
for name, mk in MODELS.items():
    if name not in ONLY: continue
    t0 = time.time(); r = run(mk, ALL, SP); OUT[f'A_{name}'] = r; show(f'{name:4s}', r)
    print(f'     ({time.time() - t0:.0f}s)', flush=True)

SP30 = splits(N, 0.3, seed=1)
print(f'\n== Protocol B: {N} random 70/30 content splits (AIM-like: 30 test contents) ==', flush=True)
for name in ('GBR',):
    r = run(MODELS[name], ALL, SP30); OUT[f'B_{name}'] = r; show(f'{name:4s}', r)

SPA = SP[:30]
print('\n== Ablation (GBR, first 30 splits of protocol A): drop one group / only one group ==', flush=True)
for grp, f in GROUPS.items():
    keep = [c for c in ALL if not f(c)]; only = [c for c in ALL if f(c)]
    r1 = run(MODELS['GBR'], keep, SPA); r2 = run(MODELS['GBR'], only, SPA)
    OUT[f'abl_wo_{grp}'] = r1; OUT[f'abl_only_{grp}'] = r2
    show(f'w/o {grp:9s}({len(keep)})', r1); show(f'only {grp:8s}({len(only)})', r2)

# inference cost of the regressor
mk = MODELS['GBR']().fit(D[ALL].values, y); t0 = time.time()
for _ in range(100): mk.predict(D[ALL].values[:1])
print(f'\nGBR predict per video: {(time.time() - t0) * 10:.2f} ms', flush=True)
json.dump(OUT, open('full_eval_results_part2.json', 'w'), indent=1)
