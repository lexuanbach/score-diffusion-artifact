"""Label-free graph-level homophily of the predictions (Fig. 2): the analogue of Eq. (4) with soft
agreement <pi(u), pi(w)> as edge homophily and the predicted class shares (degree-weighted) as p_k.
Bounded above by 1. Writes results/labelfree/labelfree.csv"""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_dataset, adjusted_homophily, HOMOPHILOUS, HETEROPHILOUS, BINARY
from scores import softmax
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = []
for ds in [d for d in HOMOPHILOUS + HETEROPHILOUS if d not in BINARY]:
    data = load_dataset(ds); ei = data.edge_index.numpy(); n = data.num_nodes
    deg = np.bincount(ei[0], minlength=n).astype(float)
    hadj = adjusted_homophily(data.edge_index, data.y)
    for m in ['GCN', 'GAT', 'SAGE', 'GCNII', 'MLP']:
        for s in range(5):
            P = softmax(torch.load(os.path.join(ROOT, 'results', 'preds', ds, f'{m}_s{s}.pt'), weights_only=False)['logits'].numpy().astype(float))
            e = np.einsum('ij,ij->i', P[ei[0]], P[ei[1]]).mean()
            pk = (deg[:, None] * P).sum(0) / deg.sum(); c = (pk ** 2).sum()
            rows.append(dict(dataset=ds, model=m, seed=s, h_adj=hadj, soft_adj=(e - c) / (1 - c)))
os.makedirs(os.path.join(ROOT, 'results', 'labelfree'), exist_ok=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, 'results', 'labelfree', 'labelfree.csv'), index=False)
print(df.groupby(['dataset', 'model']).soft_adj.mean().unstack().round(2))
