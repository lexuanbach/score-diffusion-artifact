"""E0: reproduce DAPS (Zargarbashi et al., ICML 2023) Table 1 (GCN, efficiency relative to APS).

Their protocol: 20 labelled nodes per class each for training, validation, lambda tuning and
calibration; APS gets the tuning and calibration nodes together; coverage level = the
'adaptive coverage' of their Table 5; GCN with 64 hidden units, dropout 0.8, lr 0.01, wd 1e-3.
Reported DAPS - APS set size: CoraML -0.50, PubMed +0.02, CiteSeer -0.40, Amazon-Computers -0.77.
"""
import os, sys, copy
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch_geometric.datasets import CitationFull, Planetoid, Amazon
from torch_geometric.nn import GCNConv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import _clean, ROOT as DROOT
from scores import softmax, aps, norm_adj, diffuse
from conformal import split_cp, conformal_quantile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = {'CoraML': (0.940, -0.50), 'PubMed': (0.930, 0.02), 'CiteSeer': (0.945, -0.40), 'Amazon-Computers': (0.933, -0.77)}
GRID = np.round(np.linspace(0, 1, 11), 2)


def load(name):
    if name == 'CoraML': d = CitationFull(os.path.join(DROOT, 'citationfull'), 'Cora_ML')[0]
    elif name == 'CiteSeer': d = CitationFull(os.path.join(DROOT, 'citationfull'), 'CiteSeer')[0]
    elif name == 'PubMed': d = Planetoid(os.path.join(DROOT, 'planetoid'), 'PubMed')[0]
    else: d = Amazon(os.path.join(DROOT, 'amazon'), 'Computers')[0]
    d = _clean(d); d.name = name; return d


class GCN(torch.nn.Module):
    def __init__(s, f, K):
        super().__init__(); s.c1 = GCNConv(f, 64, cached=True); s.c2 = GCNConv(64, K, cached=True)
    def forward(s, x, ei):
        x = F.dropout(x, 0.8, s.training); x = F.relu(s.c1(x, ei)); return s.c2(F.dropout(x, 0.8, s.training), ei)


def per_class(y, rng, k, exclude):
    idx = []
    for c in np.unique(y):
        cand = np.setdiff1d(np.nonzero(y == c)[0], exclude)
        idx.append(rng.choice(cand, min(k, len(cand)), replace=False))
    return np.concatenate(idx)


def main(n_models=10, n_splits=50):
    torch.set_num_threads(4); rows = []
    for name, (cov_level, reported) in TARGETS.items():
        data = load(name); n, K = data.num_nodes, data.num_classes; y = data.y.numpy(); ei = data.edge_index.numpy()
        Ahat, deg = norm_adj(ei, n); alpha = 1 - cov_level
        for s in range(n_models):
            rng = np.random.default_rng(s); torch.manual_seed(s)
            tr = per_class(y, rng, 20, []); va = per_class(y, rng, 20, tr)
            model = GCN(data.x.shape[1], K); opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
            best, state, bad = -1, None, 0
            for ep in range(1000):
                model.train(); opt.zero_grad(); out = model(data.x, data.edge_index)
                F.cross_entropy(out[tr], data.y[tr]).backward(); opt.step()
                model.eval()
                with torch.no_grad(): acc = (model(data.x, data.edge_index)[va].argmax(1) == data.y[va]).float().mean().item()
                if acc > best: best, state, bad = acc, copy.deepcopy(model.state_dict()), 0
                else:
                    bad += 1
                    if bad > 100: break
            model.load_state_dict(state); model.eval()
            with torch.no_grad(): P = softmax(model(data.x, data.edge_index).numpy().astype(float))
            rest = np.setdiff1d(np.arange(n), np.r_[tr, va])
            for r in range(n_splits):
                g = np.random.default_rng(1000 * s + r)
                tune = per_class(y, g, 20, np.r_[tr, va]); cal = per_class(y, g, 20, np.r_[tr, va, tune])
                test = np.setdiff1d(rest, np.r_[tune, cal])
                S = aps(P, g.random(n))
                sa, _ = split_cp(S, y, np.r_[tune, cal], test, alpha)
                bestl = min(GRID, key=lambda l: ((diffuse(S, Ahat, deg, l)[tune] <= conformal_quantile(diffuse(S, Ahat, deg, l)[tune, y[tune]], alpha)).sum(1).mean(), l))
                sd, _ = split_cp(diffuse(S, Ahat, deg, bestl), y, cal, test, alpha)
                rows.append(dict(dataset=name, seed=s, split=r, acc=(P.argmax(1)[test] == y[test]).mean(), alpha=alpha,
                                 aps_size=sa.sum(1).mean(), daps_size=sd.sum(1).mean(), lam=bestl,
                                 aps_cov=sa[np.arange(len(test)), y[test]].mean(), daps_cov=sd[np.arange(len(test)), y[test]].mean(),
                                 reported_delta=reported))
        print(name, 'done', flush=True)
    df = pd.DataFrame(rows); os.makedirs(os.path.join(ROOT, 'results', 'e0'), exist_ok=True)
    df.to_csv(os.path.join(ROOT, 'results', 'e0', 'daps_repro.csv'), index=False)
    df['delta'] = df.daps_size - df.aps_size
    print(df.groupby('dataset')[['acc', 'aps_cov', 'daps_cov', 'aps_size', 'daps_size', 'delta', 'reported_delta', 'lam']].mean().round(3))


if __name__ == '__main__':
    main()
