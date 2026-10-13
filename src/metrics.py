"""Coverage, efficiency and conditional-coverage metrics for one calibration/test split."""
import numpy as np


def equal_count_bins(values, nbins, rng):
    """Rank-based bins with random tie-breaking: bin b holds the b-th tenth of nodes by value."""
    r = np.lexsort((rng.random(len(values)), values))
    b = np.empty(len(values), int); b[r] = (np.arange(len(values)) * nbins) // len(values)
    return b


def basic(sets, y_test):
    cov = sets[np.arange(len(y_test)), y_test]
    size = sets.sum(1)
    return cov, size


def sscv(cov, size, alpha, K, min_count=20):
    """Size-stratified coverage violation (Angelopoulos et al., 2021)."""
    edges = [0, 1, 2, 4, 7, 11, 21, 51, K + 1]
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (size >= lo) & (size < hi) & (size >= 1) if lo == 0 else (size >= lo) & (size < hi)
        if m.sum() >= min_count:
            worst = max(worst, abs(cov[m].mean() - (1 - alpha)))
    return worst


def grouped(cov, groups, alpha, min_count=20):
    vals = {g: cov[groups == g].mean() for g in np.unique(groups) if (groups == g).sum() >= min_count}
    if not vals:
        return np.nan, np.nan, vals
    dev = max(abs(v - (1 - alpha)) for v in vals.values())
    return dev, min(vals.values()), vals


def wsc(cov, Z, rng, delta=0.1, n_dirs=500, grid=20):
    """Worst-slice coverage (Cauchois et al., 2021) with a held-out split.

    Directions and slab endpoints (quantile grid of the projection) are chosen on
    one half of the test nodes; the selected slab's coverage is reported on the other.
    """
    n = len(cov)
    perm = rng.permutation(n); A, B = perm[: n // 2], perm[n // 2:]
    V = rng.standard_normal((n_dirs, Z.shape[1])); V /= np.linalg.norm(V, axis=1, keepdims=True)
    PA = Z[A] @ V.T                                   # (nA, n_dirs)
    order = np.argsort(PA, axis=0)
    cs = np.vstack([np.zeros(n_dirs), np.cumsum(cov[A][order], axis=0)])   # (nA+1, n_dirs)
    idx = np.round(np.linspace(0, len(A), grid + 1)).astype(int)
    best, arg = np.inf, None
    for i in range(grid):
        for j in range(i + 1, grid + 1):
            if (idx[j] - idx[i]) < delta * len(A):
                continue
            c = (cs[idx[j]] - cs[idx[i]]) / (idx[j] - idx[i])
            k = int(np.argmin(c))
            if c[k] < best:
                best, arg = c[k], (k, i, j)
    k, i, j = arg
    srt = np.sort(PA[:, k])
    lo, hi = srt[idx[i]], srt[min(idx[j], len(A)) - 1]
    pb = Z[B] @ V[k]
    m = (pb >= lo) & (pb <= hi)
    return cov[B][m].mean() if m.sum() else np.nan
