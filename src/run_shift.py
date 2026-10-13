"""E5: conformal prediction when calibration and test nodes are not exchangeable.

* temporal (ogbn-arxiv): base model trained on papers <= 2017; calibration nodes are held-out
  papers <= 2017, test nodes are papers >= 2019. Reference: calibration and test both >= 2019.
* degree (Amazon-ratings, Amazon-Computers): calibration nodes are drawn from the pool with probability
  proportional to (d+1)^-1 ('low->high') or (d+1) ('high->low'); test = the remaining pool.
  Reference: uniform calibration draw.
Corrections: weighted CP (Tibshirani et al. 2019) with likelihood ratios from a logistic
propensity model (calibration vs test, unlabelled), and degree-decile Mondrian CP.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_dataset, train_adjusted_homophily
from scores import softmax, aps, norm_adj, diffuse, snr_lambda
from conformal import split_cp, mondrian_cp, weighted_cp
from run_cp import degree_groups, feature_embedding, git_hash, DAPS_LAM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def propensity_weights(Fcal, Ftest, seed):
    X = np.vstack([Fcal, Ftest]); t = np.r_[np.zeros(len(Fcal)), np.ones(len(Ftest))]
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced').fit(X, t)
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-3, 1 - 1e-3)
    w = p / (1 - p)
    return w[: len(Fcal)], w[len(Fcal):]


def run(ds, scenario, model, seeds, n_splits, alpha, m=1000):
    data = load_dataset(ds); n = data.num_nodes; y = data.y.numpy(); ei = data.edge_index.numpy()
    Ahat, d = norm_adj(ei, n); groups = degree_groups(d)
    Fz = np.c_[feature_embedding(data.x.numpy(), 10), np.log1p(d)]
    rows = []
    for seed in seeds:
        tag = '_temporal' if scenario == 'temporal' else ''
        ck = torch.load(os.path.join(ROOT, 'results', 'preds', ds, f'{model}_s{seed}{tag}.pt'), weights_only=False)
        P = softmax(ck['logits'].numpy().astype(float)); tr, va = ck['train_idx'].numpy(), ck['val_idx'].numpy()
        used = np.zeros(n, bool); used[tr] = True; used[va] = True
        h_tr = train_adjusted_homophily(ei, y, tr, n)
        trm = np.zeros(n, bool); trm[tr] = True; me = trm[ei[0]] & trm[ei[1]]
        h_e = float((y[ei[0][me]] == y[ei[1][me]]).mean())
        for r in range(n_splits):
            rng = np.random.default_rng(10_000 * seed + r)
            S = aps(P, rng.random(n))
            settings = {}
            if scenario == 'temporal':
                yr = data.node_year.numpy()
                old = np.nonzero((~used) & (yr <= 2017))[0]; new = np.nonzero(yr >= 2019)[0]
                cal = rng.choice(old, m, replace=False); settings['shift'] = (cal, new)
                p = rng.permutation(new); settings['reference'] = (p[:m], p[m:])
            else:
                pool = np.nonzero(~used)[0]
                for name, wfun in [('low->high', lambda dd: 1 / (dd + 1)), ('high->low', lambda dd: dd + 1.0)]:
                    w = wfun(d[pool]); cal = rng.choice(pool, m, replace=False, p=w / w.sum())
                    settings[name] = (cal, np.setdiff1d(pool, cal))
                p = rng.permutation(pool); settings['reference'] = (p[:m], p[m:])
            lam_v, _ = snr_lambda(S, y, va, h_tr, h_e, d)
            Sha = diffuse(S, Ahat, d, lam_v)
            for split_name, (cal, test) in settings.items():
                wc, wt = propensity_weights(Fz[cal], Fz[test], seed)
                cands = {'APS': split_cp(S, y, cal, test, alpha)[0],
                         'DAPS': split_cp(diffuse(S, Ahat, d, DAPS_LAM), y, cal, test, alpha)[0],
                         'SNR-lambda': split_cp(Sha, y, cal, test, alpha)[0],
                         'SNR-lambda+weighted': weighted_cp(Sha, y, cal, test, alpha, wc, wt),
                         'SNR-lambda+Mondrian': mondrian_cp(Sha, y, cal, test, alpha, groups),
                         'DAPS+weighted': weighted_cp(diffuse(S, Ahat, d, DAPS_LAM), y, cal, test, alpha, wc, wt)}
                low = d[test] <= np.quantile(d[test], 0.1)
                for meth, sets in cands.items():
                    cov = sets[np.arange(len(test)), y[test]]
                    rows.append(dict(dataset=ds, scenario=scenario, split_name=split_name, model=model, seed=seed,
                                     split=r, alpha=alpha, method=meth, coverage=cov.mean(), size=sets.sum(1).mean(),
                                     cov_lowdeg=cov[low].mean(), lam_mean=float(lam_v.mean()), code=git_hash(),
                                     ess=wc.sum() ** 2 / (wc ** 2).sum()))
        print(ds, scenario, seed, flush=True)
    out = os.path.join(ROOT, 'results', 'shift', f'{ds}_{scenario}_{model}.csv.gz')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, compression='gzip')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--scenario', choices=['temporal', 'degree'], required=True)
    ap.add_argument('--model', default='GCN'); ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--splits', type=int, default=10); ap.add_argument('--alpha', type=float, default=0.05)
    a = ap.parse_args()
    run(a.dataset, a.scenario, a.model, a.seeds, a.splits, a.alpha)
