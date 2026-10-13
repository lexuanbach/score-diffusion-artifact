"""Crossover h* from the CSBM sweep: smallest h above which diffusion (lambda) beats lambda=0."""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def crossover(hs, delta):
    """Last sign change of delta(h) from >0 to <=0, linearly interpolated; nan if none."""
    s = np.sign(delta)
    idx = [i for i in range(len(hs) - 1) if delta[i] > 0 and delta[i + 1] <= 0]
    if not idx:
        return np.nan if (delta > 0).all() or (delta <= 0).all() else np.nan
    i = idx[-1]
    return hs[i] + (hs[i + 1] - hs[i]) * delta[i] / (delta[i] - delta[i + 1])

def main(alpha=0.05, B=1000):
    df = pd.read_csv(os.path.join(ROOT, 'results', 'csbm', 'csbm.csv.gz'))
    df = df[df.alpha == alpha]
    base = df[df.lam == 0].set_index(['K', 'h', 'dbar', 'sigma', 'seed', 'split'])['size']
    out = []
    rng = np.random.default_rng(0)
    for (K, dbar, sigma, lam), g in df[df.lam > 0].groupby(['K', 'dbar', 'sigma', 'lam']):
        g = g.set_index(['K', 'h', 'dbar', 'sigma', 'seed', 'split'])
        diff = (g['size'] - base.loc[g.index]).reset_index()
        per_seed = diff.groupby(['seed', 'h'])['size'].mean().unstack()   # seeds x h
        hs = per_seed.columns.values
        hstar = crossover(hs, per_seed.mean(0).values)
        boots = [crossover(hs, per_seed.iloc[rng.integers(0, len(per_seed), len(per_seed))].mean(0).values) for _ in range(B)]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if np.isfinite(boots).any() else (np.nan, np.nan)
        out.append(dict(K=K, dbar=dbar, sigma=sigma, lam=lam, hstar=hstar, lo=lo, hi=hi,
                        max_inflation=(diff.groupby('h')['size'].mean() / base.groupby(level='h').mean()).max()))
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(ROOT, 'results', 'csbm', 'crossover.csv'), index=False)
    return res

if __name__ == '__main__':
    res = main()
    pd.set_option('display.width', 200)
    print(res[res.lam == 0.1].pivot_table(index=['K', 'sigma'], columns='dbar', values='hstar').round(3))
    print(res[res.lam == 0.5].pivot_table(index=['K', 'sigma'], columns='dbar', values='hstar').round(3))
    print(res.groupby(['K', 'lam']).hstar.agg(['min', 'max']).round(3))


# ------------------------------------------------------------------ prediction from Proposition 1
def score_params(K, sigma, reps=5):
    """Delta (wrong-minus-true mean score), tau (true-label score s.d.) of the graph-agnostic APS score."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from csbm import csbm, N, F
    from scores import aps
    from sklearn.linear_model import LogisticRegression
    D, T = [], []
    for r in range(reps):
        rng = np.random.default_rng(r)
        x, y, _ = csbm(N, K, 0.5, 1, sigma, rng)
        clf = LogisticRegression(max_iter=500).fit(x[:1000], y[:1000])
        P = clf.predict_proba(x[1000:]); yy = y[1000:]
        S = aps(P, rng.random(len(yy)))
        st = S[np.arange(len(yy)), yy]
        wrong = np.ones_like(S, bool); wrong[np.arange(len(yy)), yy] = False
        D.append(S[wrong].mean() - st.mean()); T.append(st.std())
    return float(np.mean(D)), float(np.mean(T))


def predicted_hstar(K, Delta, tau, inv_d, lam):
    """Smallest h with a > (lam/2)(1 + rho^2 inv_d - (1-a)^2), a = (Kh-1)/(K-1), rho^2 = 1 + h(1-h)Delta^2/tau^2."""
    hs = np.linspace(1 / K, 1, 4001)
    a = (K * hs - 1) / (K - 1)
    rho2 = 1 + hs * (1 - hs) * Delta ** 2 / tau ** 2
    ok = a > lam / 2 * (1 + rho2 * inv_d - (1 - a) ** 2)
    return hs[np.argmax(ok)] if ok.any() else np.nan


def mean_inv_degree(dbar, n=5000):
    """E[1/d | d>0] for the CSBM's (Poisson) degrees."""
    from scipy.stats import poisson
    k = np.arange(1, 400); p = poisson.pmf(k, dbar)
    return float((p / k).sum() / p.sum())


def prediction_check(res):
    pars = {(K, s): score_params(K, s) for K in [2, 5] for s in [0.5, 1.0, 2.0]}
    res = res.copy()
    res['Delta'] = [pars[(K, s)][0] for K, s in zip(res.K, res.sigma)]
    res['tau'] = [pars[(K, s)][1] for K, s in zip(res.K, res.sigma)]
    res['hstar_pred'] = [predicted_hstar(K, D, t, mean_inv_degree(d), l)
                         for K, D, t, d, l in zip(res.K, res.Delta, res.tau, res.dbar, res.lam)]
    res.to_csv(os.path.join(ROOT, 'results', 'csbm', 'crossover.csv'), index=False)
    return res, pars
