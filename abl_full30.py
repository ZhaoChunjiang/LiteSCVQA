import pandas as pd, numpy as np, glob, json
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import GradientBoostingRegressor
F=pd.concat([pd.read_csv(f) for f in sorted(glob.glob('msu_feats/b_*.csv'))]).drop_duplicates('video')
M=pd.read_csv('msu/mos.csv')[['video','mos']]; D=F.merge(M,on='video'); g=D.video.str.split('/').str[0].astype(int).values
HC=[c for c in F.columns if c not in('video','seconds')]; X=D[HC].values; y=D.mos.values; G=np.unique(g)
r=np.random.default_rng(0); SP=[r.choice(G,20,replace=False) for _ in range(100)]
s=[];pc=[];pl=[]
from scipy.stats import pearsonr
for te in SP[:30]:
    m=np.isin(g,te); p=GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=0).fit(X[~m],y[~m]).predict(X[m]); t=y[m]; gg=g[m]
    s.append(spearmanr(p,t)[0]); pc.append(np.nanmean([spearmanr(p[gg==k],t[gg==k])[0] for k in te]))
print(f'full 30 splits SRCC {np.median(s):.4f}±{np.std(s):.4f} pc {np.median(pc):.4f}')
h=json.load(open('hybrid_results_100_HC-GBR.json')); b=json.load(open('hybrid_results_100_BRISQUESVR_CNNSVR_CNNGBR.json')); e=json.load(open('hybrid_results_100_Hybrid-early_Hybrid-late.json'))
for k in ['raw_SRCC','raw_pc']:
  print(k,'HC>BRISQUE p',wilcoxon(h['HC-GBR_'+k],b['BRISQUE+SVR_'+k],alternative='greater').pvalue,'HC>CNN p',wilcoxon(h['HC-GBR_'+k],b['CNN+SVR_'+k],alternative='greater').pvalue,'Hyb>HC p(1-sided)',wilcoxon(e['Hybrid-early_'+k],h['HC-GBR_'+k],alternative='greater').pvalue)
