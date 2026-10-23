"""Baselines + hybrid on the same 100 content-disjoint 80/20 splits as full_eval.py (seed 0)."""
import pandas as pd, numpy as np, glob, sys, json, time, warnings
from scipy.stats import spearmanr, kendalltau, pearsonr
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')

F = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('msu_feats/b_*.csv'))]).drop_duplicates('video')
C = pd.read_csv('msu_deep.csv').drop_duplicates('video')
M = pd.read_csv('msu/mos.csv')[['video', 'mos']]
D = F.merge(C, on='video').merge(M, on='video'); D['group'] = D.video.str.split('/').str[0].astype(int)
HC = [c for c in F.columns if c not in ('video', 'seconds')]
CNN = [c for c in C.columns if c.startswith('cg_') or c.startswith('cl_')]
BQ = [c for c in C.columns if c.startswith('brisque')]
y = D.mos.values; g = D.group.values; Gs = np.unique(g)
print(len(D), 'videos', len(Gs), 'contents', flush=True)
print(f'CNN branch time/video: median {D.sec.median():.2f}s', flush=True)

def L4(x, b1, b2, b3, b4): return (b1 - b2) / (1 + np.exp(-(x - b3) / (abs(b4) + 1e-6))) + b2
def plcc(p, t):
    try:
        po, _ = curve_fit(L4, p, t, p0=[t.max(), t.min(), np.median(p), np.std(p) + 1e-3], maxfev=20000); return pearsonr(L4(p, *po), t)[0]
    except Exception: return pearsonr(p, t)[0]

gbr = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
svr = lambda: make_pipeline(StandardScaler(), SVR(C=10, gamma='scale'))
K = 32

def pred(cfg, tr, te):
    Xh, Xc, Xb = D[HC].values, D[CNN].values, D[BQ].values
    if cfg == 'BRISQUE+SVR': return svr().fit(Xb[tr], y[tr]).predict(Xb[te])
    if cfg == 'HC-GBR': return gbr().fit(Xh[tr], y[tr]).predict(Xh[te])
    pca = make_pipeline(StandardScaler(), PCA(K, random_state=0)).fit(Xc[tr]); Z = pca.transform(Xc)
    if cfg == 'CNN+SVR': return svr().fit(Z[tr], y[tr]).predict(Z[te])
    if cfg == 'CNN+GBR': return gbr().fit(Z[tr], y[tr]).predict(Z[te])
    if cfg == 'Hybrid-early': X = np.hstack([Xh, Z]); return gbr().fit(X[tr], y[tr]).predict(X[te])
    if cfg == 'Hybrid-late':
        return 0.5 * gbr().fit(Xh[tr], y[tr]).predict(Xh[te]) + 0.5 * svr().fit(Z[tr], y[tr]).predict(Z[te])

r = np.random.default_rng(0); SP = [r.choice(np.arange(1, 101), 20, replace=False) for _ in range(100)]
# identical split generator to full_eval.py (G there = unique groups 1..100)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
cfgs = sys.argv[2].split(',') if len(sys.argv) > 2 else ['BRISQUE+SVR', 'CNN+SVR', 'CNN+GBR', 'HC-GBR', 'Hybrid-early', 'Hybrid-late']
OUT = {}
for cfg in cfgs:
    t0 = time.time(); R = {k: [] for k in ('SRCC', 'KRCC', 'PLCC', 'pcSRCC', 'pcKRCC')}
    for te_g in SP[:N]:
        m = np.isin(g, te_g); p = pred(cfg, ~m, m); t = y[m]; gg = g[m]
        R['SRCC'].append(spearmanr(p, t)[0]); R['KRCC'].append(kendalltau(p, t)[0]); R['PLCC'].append(plcc(p, t))
        R['pcSRCC'].append(np.nanmean([spearmanr(p[gg == k], t[gg == k])[0] for k in te_g]))
        R['pcKRCC'].append(np.nanmean([kendalltau(p[gg == k], t[gg == k])[0] for k in te_g]))
    OUT[cfg] = {k: (float(np.median(v)), float(np.std(v))) for k, v in R.items()}
    OUT[cfg + '_raw_SRCC'] = R['SRCC']; OUT[cfg + '_raw_pc'] = R['pcSRCC']
    print(f'{cfg:13s} ' + '  '.join(f'{k} {np.median(v):.4f}±{np.std(v):.4f}' for k, v in R.items()) + f'  ({time.time() - t0:.0f}s)', flush=True)
json.dump(OUT, open(f'hybrid_results_{N}_' + '_'.join(c.replace('+','') for c in cfgs) + '.json', 'w'))
