"""CF-GNN (Huang et al., NeurIPS 2023), ported from the official code
(github.com/snap-stanford/conformalized-gnn, train.py, 'optimal' classification mode):

* correction model: a GNN taking the frozen base model's softmax as input, output K logits;
* loss: cross-entropy on the base model's training nodes; after epoch 1000 plus
  size_loss_weight * mean(relu(sum_k sigmoid((softmax - qhat)/tau) - target_size)) where qhat is the
  THR quantile on one half of a calibration hold-out and the size is measured on the other half;
* hold-out: half of the calibration budget (calib_fraction 0.5); the final conformal step
  calibrates APS on the remaining half;
* model selection: APS efficiency on the validation set (we evaluate it every 10 epochs on 10
  random half/half splits instead of every epoch on 100);
* hyperparameters: the published GCN-base optimum for the dataset when one exists
  (params/optimal_param_set.pkl), otherwise the published Cora-ML optimum.

Usage: python src/cfgnn.py --datasets Cora --seeds 0 1 2 3 4 --splits 10
"""
import argparse, os, sys, time, copy
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_dataset, cal_test_split
from scores import softmax, aps, norm_adj
from conformal import split_cp, conformal_quantile
from metrics import equal_count_bins
from run_cp import evaluate, feature_embedding, git_hash

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# published optimum for a GCN base model (official params/optimal_param_set.pkl, key 'GCN')
PUBLISHED = {
    'Amazon-Computers': dict(confgnn_base_model='GAT', confgnn_lr=0.1, confgnn_num_layers=1, confnn_hidden_dim=128, size_loss_weight=1, target_size=0, tau=0.1),
    'Amazon-Photo': dict(confgnn_base_model='GAT', confgnn_lr=0.001, confgnn_num_layers=4, confnn_hidden_dim=16, size_loss_weight=0.01, target_size=1, tau=0.001),
    'CiteSeer': dict(confgnn_base_model='GCN', confgnn_lr=0.001, confgnn_num_layers=3, confnn_hidden_dim=256, size_loss_weight=1, target_size=0, tau=0.001),
    'PubMed': dict(confgnn_base_model='GAT', confgnn_lr=0.1, confgnn_num_layers=3, confnn_hidden_dim=128, size_loss_weight=0.001, target_size=1, tau=0.001),
    'Cora_ML': dict(confgnn_base_model='GAT', confgnn_lr=0.0001, confgnn_num_layers=2, confnn_hidden_dim=64, size_loss_weight=0.1, target_size=1, tau=1),
}


def params_for(ds):
    if ds in PUBLISHED:
        return PUBLISHED[ds], 'published'
    return PUBLISHED['Cora_ML'], 'default(Cora-ML)'


class ConfGNN(torch.nn.Module):
    """GNN_Multi_Layer of the official code (dropout 0.5 before every layer, ReLU between)."""
    def __init__(self, K, hidden, layers, backbone):
        super().__init__()
        def conv(i, o):
            return {'GCN': lambda: GCNConv(i, o, cached=True), 'GAT': lambda: GATConv(i, o, 1),
                    'GraphSAGE': lambda: SAGEConv(i, o, 'sum'), 'SGC': lambda: SGConv(i, o)}[backbone]()
        dims = [K] + [hidden] * (layers - 1) + [K]
        self.convs = torch.nn.ModuleList([conv(a, b) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, p, ei):
        x = p
        for i, c in enumerate(self.convs):
            x = F.dropout(x, 0.5, self.training)
            x = c(x, ei)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


def aps_eff(Pv, yv, alpha, rng, reps=10):
    n = len(yv); sizes = []
    for _ in range(reps):
        perm = rng.permutation(n); c, t = perm[: n // 2], perm[n // 2:]
        S = aps(Pv, rng.random(n))
        q = conformal_quantile(S[c, yv[c]], alpha)
        sizes.append((S[t] <= q).sum(1).mean())
    return float(np.mean(sizes))


def train_cfgnn(P0, ei, y, train_idx, val_idx, holdout, alpha, hp, seed, epochs=5000, warmup=1000, eval_every=10, device='cpu'):
    torch.manual_seed(seed); g = np.random.default_rng(seed)
    K = P0.shape[1]
    model = ConfGNN(K, hp['confnn_hidden_dim'], hp['confgnn_num_layers'], hp['confgnn_base_model']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp['confgnn_lr'], weight_decay=5e-4)
    x = torch.tensor(P0, dtype=torch.float, device=device); eit = torch.tensor(ei, device=device); yt = torch.tensor(y, device=device)
    h = g.permutation(holdout); tr_cal, tr_test = h[len(h) // 2:], h[: len(h) // 2]
    tr = torch.tensor(train_idx, device=device); tc = torch.tensor(tr_cal, device=device); tt = torch.tensor(tr_test, device=device)
    n_temp = len(tr_cal); q_level = np.ceil((n_temp + 1) * (1 - alpha)) / n_temp
    best, best_out = np.inf, None
    for ep in range(1, epochs + 1):
        model.train(); opt.zero_grad()
        out = model(x, eit); sm = F.softmax(out, 1)
        loss = F.cross_entropy(out[tr], yt[tr])
        if ep > warmup:
            sc = sm[tc, yt[tc]]
            qhat = torch.quantile(sc, 1 - q_level, interpolation='higher')
            c = torch.sigmoid((sm[tt] - qhat) / hp['tau'])
            loss = loss + hp['size_loss_weight'] * torch.relu(c.sum(1) - hp['target_size']).mean()
        loss.backward(); opt.step()
        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                o = model(x, eit).cpu().numpy()
            eff = aps_eff(softmax(o[val_idx].astype(float)), y[val_idx], alpha, g)
            if eff < best:
                best, best_out = eff, o
    return softmax(best_out.astype(float)), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--model', default='GCN')
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--splits', nargs='+', type=int, default=list(range(10)))
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for ds in a.datasets:
        out_path = os.path.join(ROOT, 'results', 'cfgnn', f'{ds}_{a.model}_a{a.alpha}{a.tag}.csv.gz')
        done = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame()
        data = load_dataset(ds); n, K = data.num_nodes, data.num_classes
        ei = data.edge_index.numpy(); y = data.y.numpy(); Ahat, d = norm_adj(ei, n)
        Z = feature_embedding(data.x.numpy()); hp, src = params_for(ds)
        rows = done.to_dict('records')
        for seed in a.seeds:
            ck = torch.load(os.path.join(ROOT, 'results', 'preds', ds, f'{a.model}_s{seed}.pt'), weights_only=False)
            P = softmax(ck['logits'].numpy().astype(float))
            tr, va = ck['train_idx'].numpy(), ck['val_idx'].numpy()
            used = np.zeros(n, bool); used[tr] = True; used[va] = True
            pool = np.nonzero(~used)[0]
            m_eff = min(1000, len(pool) // 2)
            for r in a.splits:
                if len(done) and ((done.seed == seed) & (done.split == r)).any():
                    continue
                rng = np.random.default_rng(1000 * seed + r + 7 * 1000)
                cal, test = cal_test_split(pool, m_eff, seed * 100 + r + 7 * 1000)
                u = rng.random(n)
                deg_bins = equal_count_bins(d[test], 10, rng)
                hold, cal2 = cal[: m_eff // 2], cal[m_eff // 2:]
                t0 = time.time()
                Pc, eff = train_cfgnn(P, ei, y, tr, va, hold, a.alpha, hp, seed=1000 * seed + r, epochs=a.epochs, device=a.device)
                sets, _ = split_cp(aps(Pc, u), y, cal2, test, a.alpha)
                res, _, _ = evaluate(sets, y[test], a.alpha, K, deg_bins, y[test], Z, rng)
                rows.append(dict(dataset=ds, model=a.model, seed=seed, split=r, alpha=a.alpha, m_cal=m_eff,
                                 method='CF-GNN', runtime=time.time() - t0, hp_source=src, val_eff=eff,
                                 code=git_hash(), **res))
                print(ds, seed, r, f"size={res['size']:.3f} cov={res['coverage']:.3f} {time.time()-t0:.0f}s", flush=True)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                pd.DataFrame(rows).to_csv(out_path, index=False, compression='gzip')


if __name__ == '__main__':
    main()
