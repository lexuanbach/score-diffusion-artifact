"""Unit tests: split CP must reach 1-alpha on i.i.d. data; weighted CP with unit weights == split CP;
diffusion with any label-free lambda keeps validity on a random graph."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scores import aps, thr, softmax, norm_adj, diffuse
from conformal import split_cp, weighted_cp, conformal_quantile

rng = np.random.default_rng(0)
n, K, alpha = 4000, 5, 0.1
covs, covs_w, covs_d = [], [], []
for t in range(300):
    logits = rng.normal(size=(n, K)) * 2
    P = softmax(logits)
    y = np.array([rng.choice(K, p=p) for p in P[:200]] + list(rng.integers(0, K, n - 200)))
    y[200:] = (P[200:].cumsum(1) > rng.random((n - 200, 1))).argmax(1)
    S = aps(P, rng.random(n))
    perm = rng.permutation(n); cal, test = perm[:500], perm[500:]
    sets, q = split_cp(S, y, cal, test, alpha)
    covs.append(sets[np.arange(len(test)), y[test]].mean())
    sw = weighted_cp(S, y, cal, test, alpha, np.ones(len(cal)), np.ones(len(test)))
    covs_w.append(sw[np.arange(len(test)), y[test]].mean())
    ei = rng.integers(0, n, size=(2, 8 * n)); ei = np.hstack([ei, ei[::-1]])
    Ah, d = norm_adj(ei, n)
    Sd = diffuse(S, Ah, d, rng.random(n))
    sd, _ = split_cp(Sd, y, cal, test, alpha)
    covs_d.append(sd[np.arange(len(test)), y[test]].mean())
m = (500 + 1) * (1 - alpha); ub = 1 - alpha + 1 / (500 + 1)
print(f'split   cov={np.mean(covs):.4f}  (target >= {1-alpha}, <= {ub:.4f})')
print(f'weighted(unit) cov={np.mean(covs_w):.4f}')
print(f'diffused cov={np.mean(covs_d):.4f}')
assert 1 - alpha - 0.004 <= np.mean(covs) <= ub + 0.004
assert abs(np.mean(covs_w) - np.mean(covs)) < 0.01
assert 1 - alpha - 0.004 <= np.mean(covs_d) <= ub + 0.004
assert conformal_quantile(np.arange(10.), 0.01) == np.inf
print('all tests passed')
