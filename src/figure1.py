#!/usr/bin/env python3
"""Data behind Figure 1: one real Roman-empire node walked through the pipeline.

GraphSAGE, seed 0, calibration/test split 0 of run_cp.py (same split and APS randomisation), alpha=0.05.
Writes figures/fig1/fig1_data.tex (macros \\Fig...). Refuses to write if a check fails.
Usage (project root): python src/figure1.py
"""
import os, sys
import numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from data import load_dataset, cal_test_split, train_adjusted_homophily
from scores import softmax, aps, norm_adj, diffuse, snr_lambda, hg_daps_delta
from conformal import split_cp, conformal_quantile

DS, MODEL, SEED, SPLIT, ALPHA, NODE = 'Roman-empire', 'SAGE', 0, 0, 0.05, 7727


def fail(m): sys.exit(f'make_fig1: CHECK FAILED: {m}')


data = load_dataset(DS); n = data.num_nodes; y = data.y.numpy(); ei = data.edge_index.numpy()
Ahat, d = norm_adj(ei, n)
ck = torch.load(os.path.join(ROOT, 'results', 'preds', DS, f'{MODEL}_s{SEED}.pt'), weights_only=False)
P = softmax(ck['logits'].numpy().astype(float)); tr, va = ck['train_idx'].numpy(), ck['val_idx'].numpy()
used = np.zeros(n, bool); used[tr] = True; used[va] = True; pool = np.nonzero(~used)[0]
rng = np.random.default_rng(1000 * SEED + SPLIT + 7 * 1000)
cal, test = cal_test_split(pool, min(1000, len(pool) // 2), SEED * 100 + SPLIT + 7 * 1000)
u = rng.random(n); S = aps(P, u)
a_hat = train_adjusted_homophily(ei, y, tr, n)
trm = np.zeros(n, bool); trm[tr] = True; me = trm[ei[0]] & trm[ei[1]]
h_e = float((y[ei[0][me]] == y[ei[1][me]]).mean()); m_ctx = int(me.sum() // 2)
lam_snr, _ = snr_lambda(S, y, va, a_hat, h_e, d)
Sd = diffuse(S, Ahat, d, 0.5); Ss = diffuse(S, Ahat, d, lam_snr)
qa = conformal_quantile(S[cal, y[cal]], ALPHA); qd = conformal_quantile(Sd[cal, y[cal]], ALPHA)
qs = conformal_quantile(Ss[cal, y[cal]], ALPHA)
v = NODE
if v not in set(test): fail('node not in test set')
nb = ei[1][ei[0] == v]; yv = int(y[v])
setA = np.nonzero(S[v] <= qa)[0]; setD = np.nonzero(Sd[v] <= qd)[0]; setS = np.nonzero(Ss[v] <= qs)[0]
if yv not in setA or yv not in setD: fail('true label not covered')
if (y[nb] == yv).any(): fail('neighbour shares the label (example should be heterophilous)')
# run-level means must match the paper's results/cp rows for this run
cp = pd.read_csv(os.path.join(ROOT, 'results', 'cp', f'{DS}.csv.gz'))
row = cp[(cp.model == MODEL) & (cp.seed == SEED) & (cp.split == SPLIT) & (cp.alpha == ALPHA)].set_index('method')
CA, CD, CS = S[test] <= qa, Sd[test] <= qd, Ss[test] <= qs
for name, C in [('APS', CA), ('DAPS', CD), ('SNR-lambda', CS)]:
    if abs(C.sum(1).mean() - row.loc[name, 'size']) > 1e-9: fail(f'{name} size mismatch')
allm = cp[(cp.model == MODEL) & (cp.alpha == ALPHA)].groupby('method')[['size', 'coverage', 'cls_min']].mean()
order = np.argsort(-P[v]); top = order[:3]
import json
meta_hadj = json.load(open(os.path.join(ROOT, 'results', 'cp', f'{DS}.meta.json')))['h_adj']
# The node ID was manually chosen for exposition, not randomly sampled.
# Count a reproducible structural pool and the stricter depicted outcome pool.
hetero = np.bincount(ei[0], weights=(y[ei[0]] != y[ei[1]]), minlength=n) == d
structural = (d[test] == 2) & hetero[test]
outcome = structural & (CA.sum(1) == 1) & CA[np.arange(len(test)), y[test]] & (CD.sum(1) >= 4)
assert structural[np.flatnonzero(test == v)[0]]
assert outcome[np.flatnonzero(test == v)[0]]
M = {
    'FigEligible': str(int(structural.sum())), 'FigOutcomeEligible': str(int(outcome.sum())),
    'FigNode': str(v), 'FigDeg': str(len(nb)), 'FigTrue': str(yv),
    'FigNbA': str(int(y[nb[0]])), 'FigNbB': str(int(y[nb[1]])),
    'FigPtop': f'{P[v, top[0]]:.3f}', 'FigPtopCls': str(int(top[0])),
    'FigStrue': f'{S[v, yv]:.2f}', 'FigNbScore': f'{(Ahat @ S)[v, yv]:.2f}', 'FigSdtrue': f'{Sd[v, yv]:.2f}',
    'FigQa': f'{qa:.3f}', 'FigQd': f'{qd:.3f}', 'FigQs': f'{qs:.3f}',
    'FigSizeA': str(len(setA)), 'FigSizeD': str(len(setD)), 'FigSizeS': str(len(setS)),
    'FigSetD': ', '.join(str(int(k)) for k in setD),
    'FigAhat': f'{a_hat:.2f}', 'FigLamSNR': f'{lam_snr[v]:.2f}', 'FigLamHG': f'{hg_daps_delta(a_hat, m_ctx):.2f}',
    'FigMeanA': f"{allm.loc['APS', 'size']:.2f}", 'FigMeanD': f"{allm.loc['DAPS', 'size']:.2f}", 'FigMeanS': f"{allm.loc['SNR-lambda', 'size']:.2f}",
    'FigInfl': f"{100 * (allm.loc['DAPS', 'size'] / allm.loc['APS', 'size'] - 1):.0f}",
    'FigCovMarg': f"{allm.loc['APS', 'coverage']:.3f}", 'FigCovCls': f"{allm.loc['APS', 'cls_min']:.2f}",
    'FigCovClsD': f"{allm.loc['DAPS', 'cls_min']:.2f}", 'FigK': str(int(y.max() + 1)),
    'FigHadj': f"{float(meta_hadj):.2f}",
}
out = os.environ.get('FIGURE1_OUTPUT', os.path.join(ROOT, 'outputs', 'figure1.json'))
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    json.dump(M, f, indent=2, sort_keys=True)
print(f'Figure 1 checks passed: {out}')
