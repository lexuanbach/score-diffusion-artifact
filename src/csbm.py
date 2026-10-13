"""E1: CSBM sweep. Set size vs diffusion strength lambda, edge homophily h, mean degree d, noise sigma.

The base classifier is a graph-agnostic multinomial logistic regression, so the undiffused
score distribution does not depend on h and every h-dependence comes from the diffusion.
Writes results/csbm/csbm.csv.gz (one row per graph x lambda x alpha).

Usage: python src/csbm.py --workers 6
"""
import argparse, os, sys, itertools, time
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scores import aps, norm_adj, diffuse
from conformal import split_cp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = np.round(np.arange(0, 1.0001, 0.05), 2)
D = [3, 5, 10, 20, 50]
SIGMA = [0.5, 1.0, 2.0]
KS = [2, 5]
SEEDS = range(10)
LAMS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
N, F, M_CAL, N_SPLITS = 5000, 10, 1000, 5


def csbm(n, K, h, dbar, sigma, rng):
    """Balanced K-class CSBM: expected h*dbar same-class and (1-h)*dbar other-class neighbours."""
    y = np.repeat(np.arange(K), n // K); y = rng.permutation(y)
    mu = np.zeros((K, F)); mu[np.arange(K), np.arange(K)] = 1.0
    if K == 2:
        mu[1] = -mu[0]
    x = mu[y] + sigma * rng.standard_normal((n, F))
    members = [np.nonzero(y == k)[0] for k in range(K)]
    edges = []
    m_same = rng.poisson(h * dbar * n / 2)              # undirected same-class edges
    m_diff = rng.poisson((1 - h) * dbar * n / 2)
    if m_same:
        k = rng.integers(0, K, m_same)
        a = np.array([members[c][i] for c, i in zip(k, rng.integers(0, n // K, m_same))])
        b = np.array([members[c][i] for c, i in zip(k, rng.integers(0, n // K, m_same))])
        edges.append(np.vstack([a, b]))
    if m_diff:
        k1 = rng.integers(0, K, m_diff); k2 = (k1 + rng.integers(1, K, m_diff)) % K
        a = np.array([members[c][i] for c, i in zip(k1, rng.integers(0, n // K, m_diff))])
        b = np.array([members[c][i] for c, i in zip(k2, rng.integers(0, n // K, m_diff))])
        edges.append(np.vstack([a, b]))
    e = np.hstack(edges) if edges else np.zeros((2, 0), int)
    e = e[:, e[0] != e[1]]
    e = np.unique(np.hstack([e, e[::-1]]), axis=1)
    return x, y, e


def one(args):
    K, h, dbar, sigma, seed = args
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(hash((K, int(h * 100), dbar, int(sigma * 10), seed)) % 2**32)
    x, y, e = csbm(N, K, h, dbar, sigma, rng)
    perm = rng.permutation(N); tr, pool = perm[:1000], perm[1000:]
    clf = LogisticRegression(max_iter=500).fit(x[tr], y[tr])
    P = clf.predict_proba(x)
    Ahat, d = norm_adj(e, N)
    h_emp = float((y[e[0]] == y[e[1]]).mean()) if e.shape[1] else np.nan
    rows = []
    for s in range(N_SPLITS):
        p2 = rng.permutation(pool); cal, test = p2[:M_CAL], p2[M_CAL:]
        S = aps(P, rng.random(N))
        for lam in LAMS:
            for alpha in (0.05, 0.1):
                sets, q = split_cp(diffuse(S, Ahat, d, lam), y, cal, test, alpha)
                cov = sets[np.arange(len(test)), y[test]]
                # coverage of the lowest-degree fifth of test nodes (Corollary on low degree)
                low = d[test] <= np.quantile(d[test], 0.2)
                rows.append(dict(K=K, h=h, dbar=dbar, sigma=sigma, seed=seed, split=s, lam=lam, alpha=alpha,
                                 size=sets.sum(1).mean(), coverage=cov.mean(), cov_low=cov[low].mean(),
                                 size_low=sets[low].sum(1).mean(), acc=(P.argmax(1)[test] == y[test]).mean(),
                                 h_emp=h_emp, deg_emp=float(d.mean())))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    grid = list(itertools.product(KS, H, D, SIGMA, SEEDS))
    if a.quick:
        grid = [g for g in grid if g[2] == 10 and g[3] == 1.0 and g[4] < 3]
    t0 = time.time(); out = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(one, grid, chunksize=4)):
            out.extend(r)
            if i % 200 == 0:
                print(f'{i}/{len(grid)} graphs, {time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(out)
    os.makedirs(os.path.join(ROOT, 'results', 'csbm'), exist_ok=True)
    df.to_csv(os.path.join(ROOT, 'results', 'csbm', 'csbm_quick.csv.gz' if a.quick else 'csbm.csv.gz'), index=False)
    print('done', len(grid), 'graphs', f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
