import pandas as pd, numpy as np, glob, json, time, pickle, warnings, resource
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings('ignore')
F = pd.concat([pd.read_csv(f) for f in sorted(glob.glob('msu_feats/b_*.csv'))]).drop_duplicates('video')
M = pd.read_csv('msu/mos.csv')[['video', 'mos']]; D = F.merge(M, on='video'); D['group'] = D.video.str.split('/').str[0].astype(int)
D['app'] = D.video.str.split('/').str[1].str.replace('.mp4', '').str.split('_').str[0]
HC = [c for c in F.columns if c not in ('video', 'seconds')]
y = D.mos.values; g = D.group.values; G = np.unique(g)
gbr = lambda: GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
r = np.random.default_rng(0); SP = [r.choice(G, 20, replace=False) for _ in range(100)]
def L4(x, b1, b2, b3, b4): return (b1 - b2) / (1 + np.exp(-(x - b3) / (abs(b4) + 1e-6))) + b2
def rmse(p, t):
    try:
        po, _ = curve_fit(L4, p, t, p0=[t.max(), t.min(), np.median(p), np.std(p) + 1e-3], maxfev=20000); q = L4(p, *po)
    except Exception: q = p
    return float(np.sqrt(np.mean((q - t) ** 2)))
X = D[HC].values; out = {'rmse': [], 'app': {}}
for te in SP:
    m = np.isin(g, te); p = gbr().fit(X[~m], y[~m]).predict(X[m]); t = y[m]
    out['rmse'].append(rmse(p, t))
    for a in np.unique(D.app[m]):
        k = (D.app[m] == a).values
        out['app'].setdefault(a, []).append(spearmanr(p[k], t[k])[0])
print('RMSE median', np.median(out['rmse']), 'std', np.std(out['rmse']), 'MOS range', y.min(), y.max(), 'std', y.std())
for a, v in sorted(out['app'].items()): print(f'{a:6s} n/split={int((D.app==a).sum()/100*20)} SRCC {np.nanmedian(v):.4f}±{np.nanstd(v):.4f}')
# mean-pooling-only ablation (30 splits)
MEAN = [c for c in HC if not (c.endswith('_p10') or c.endswith('_p90'))]
s = []; pc = []
for te in SP[:30]:
    m = np.isin(g, te); Xm = D[MEAN].values; p = gbr().fit(Xm[~m], y[~m]).predict(Xm[m]); t = y[m]; gg = g[m]
    s.append(spearmanr(p, t)[0]); pc.append(np.nanmean([spearmanr(p[gg == k], t[gg == k])[0] for k in te]))
print(f'mean-pool only ({len(MEAN)} dims): SRCC {np.median(s):.4f}±{np.std(s):.4f} pc {np.median(pc):.4f}')
mdl = gbr().fit(X, y); b = pickle.dumps(mdl)
nodes = sum(e[0].tree_.node_count for e in mdl.estimators_)
print('GBR model size KB', len(b) / 1024, 'nodes', nodes)
json.dump(out, open('extra_eval.json', 'w'))
