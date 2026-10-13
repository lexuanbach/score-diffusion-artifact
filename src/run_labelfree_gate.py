"""Label-free graph-level gate with the bounded estimate of Fig. 2: lambda = 0.75 clip(soft_adj, 0, 1).
Same splits and APS randomisation as run_cp.py (alpha=0.05). Writes results/labelfree/gate.csv.gz"""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_dataset, cal_test_split, HOMOPHILOUS, HETEROPHILOUS, BINARY
from scores import softmax, aps, norm_adj, diffuse
from conformal import split_cp
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for ds in [d for d in HOMOPHILOUS + HETEROPHILOUS if d not in BINARY]:
    data = load_dataset(ds); n = data.num_nodes; y = data.y.numpy(); ei = data.edge_index.numpy()
    Ahat, d = norm_adj(ei, n); deg = d.astype(float)
    for m in ['GCN', 'GAT', 'SAGE', 'GCNII', 'MLP']:
        for seed in range(5):
            ck = torch.load(os.path.join(ROOT, 'results', 'preds', ds, f'{m}_s{seed}.pt'), weights_only=False)
            P = softmax(ck['logits'].numpy().astype(float))
            used = np.zeros(n, bool); used[ck['train_idx'].numpy()] = True; used[ck['val_idx'].numpy()] = True
            pool = np.nonzero(~used)[0]; m_eff = min(1000, len(pool) // 2)
            e = np.einsum('ij,ij->i', P[ei[0]], P[ei[1]]).mean()
            pk = (deg[:, None] * P).sum(0) / deg.sum(); c = (pk ** 2).sum(); sa = (e - c) / (1 - c)
            lam = 0.75 * float(np.clip(sa, 0, 1))
            for r in range(10):
                rng = np.random.default_rng(1000 * seed + r + 7 * 1000)
                cal, test = cal_test_split(pool, m_eff, seed * 100 + r + 7 * 1000)
                S = aps(P, rng.random(n))
                sets, _ = split_cp(diffuse(S, Ahat, d, lam), y, cal, test, 0.05)
                rows.append(dict(dataset=ds, model=m, seed=seed, split=r, method='label-free gate (bounded)', lam_sel=lam,
                                 size=sets.sum(1).mean(), coverage=sets[np.arange(len(test)), y[test]].mean()))
    print(ds, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'results', 'labelfree', 'gate.csv.gz'), index=False)
