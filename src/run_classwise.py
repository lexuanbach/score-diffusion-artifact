"""Class-conditional (label-Mondrian) conformal prediction, GCN base, alpha=0.05: the remedy for the
class-conditional failure of Sect. 7.4. Same splits and APS randomisation as run_cp.py.
Writes results/classwise/classwise.csv.gz"""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_dataset, cal_test_split, train_adjusted_homophily, DATASETS
from scores import softmax, aps, norm_adj, diffuse, snr_lambda
from conformal import split_cp, conformal_quantile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def classwise_cp(S, y, cal, test, alpha, K):
    q = np.array([conformal_quantile(S[cal[y[cal] == k], k], alpha) if (y[cal] == k).any() else np.inf for k in range(K)])
    return S[test] <= q[None, :]


rows = []
for ds in [d for d in DATASETS if d != 'ogbn-arxiv']:
    data = load_dataset(ds); n, K = data.num_nodes, data.num_classes; y = data.y.numpy(); ei = data.edge_index.numpy()
    Ahat, d = norm_adj(ei, n)
    for seed in range(5):
        ck = torch.load(os.path.join(ROOT, 'results', 'preds', ds, f'GCN_s{seed}.pt'), weights_only=False)
        P = softmax(ck['logits'].numpy().astype(float)); tr, va = ck['train_idx'].numpy(), ck['val_idx'].numpy()
        used = np.zeros(n, bool); used[tr] = True; used[va] = True; pool = np.nonzero(~used)[0]
        h_tr = train_adjusted_homophily(ei, y, tr, n)
        trm = np.zeros(n, bool); trm[tr] = True; me = trm[ei[0]] & trm[ei[1]]
        h_e = float((y[ei[0][me]] == y[ei[1][me]]).mean()) if me.any() else 0.5
        m_eff = min(1000, len(pool) // 2)
        for r in range(10):
            rng = np.random.default_rng(1000 * seed + r + 7 * 1000)
            cal, test = cal_test_split(pool, m_eff, seed * 100 + r + 7 * 1000)
            S = aps(P, rng.random(n)); lam, _ = snr_lambda(S, y, va, h_tr, h_e, d); Sd = diffuse(S, Ahat, d, lam)
            yt = y[test]
            for name, sets in [('APS', split_cp(S, y, cal, test, 0.05)[0]), ('APS-classwise', classwise_cp(S, y, cal, test, 0.05, K)),
                               ('SNR-lambda', split_cp(Sd, y, cal, test, 0.05)[0]), ('SNR-lambda-classwise', classwise_cp(Sd, y, cal, test, 0.05, K))]:
                cov = sets[np.arange(len(test)), yt]
                cls = [cov[yt == k].mean() for k in range(K) if (yt == k).sum() >= 20]
                rows.append(dict(dataset=ds, seed=seed, split=r, method=name, coverage=cov.mean(), size=sets.sum(1).mean(), cls_min=min(cls)))
    print(ds, flush=True)
os.makedirs(os.path.join(ROOT, 'results', 'classwise'), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'results', 'classwise', 'classwise.csv.gz'), index=False)
