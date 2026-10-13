"""Split, Mondrian, weighted and neighbourhood (NAPS) conformal calibration."""
import numpy as np
import scipy.sparse as sp


def conformal_quantile(cal_scores, alpha):
    """ceil((m+1)(1-alpha))-th smallest calibration score; +inf when that rank exceeds m."""
    m = len(cal_scores)
    k = int(np.ceil((m + 1) * (1 - alpha)))
    if k > m:
        return np.inf
    return np.partition(cal_scores, k - 1)[k - 1]


def split_cp(S, y, cal, test, alpha):
    """Returns the boolean prediction-set matrix for test nodes and the threshold."""
    q = conformal_quantile(S[cal, y[cal]], alpha)
    return S[test] <= q, q


def mondrian_cp(S, y, cal, test, alpha, groups):
    """Class-agnostic Mondrian CP: one quantile per group (e.g. degree bin)."""
    sets = np.zeros((len(test), S.shape[1]), dtype=bool)
    for g in np.unique(groups[test]):
        cg = cal[groups[cal] == g]
        tg = groups[test] == g
        q = conformal_quantile(S[cg, y[cg]], alpha) if len(cg) else np.inf
        sets[tg] = S[test[tg]] <= q
    return sets


def weighted_cp(S, y, cal, test, alpha, w_cal, w_test):
    """Tibshirani et al. (2019) weighted split CP with likelihood-ratio weights.

    For each test node, the threshold is the (1-alpha) quantile of the weighted
    distribution putting mass w_i on calibration score i and w_test on +inf.
    """
    sc = S[cal, y[cal]]
    o = np.argsort(sc); sc = sc[o]; wc = w_cal[o]
    cw = np.cumsum(wc)
    tot = cw[-1] + w_test                      # (n_test,)
    # smallest index j with cw[j] / tot >= 1 - alpha
    target = (1 - alpha) * tot
    j = np.searchsorted(cw, target, side='left')
    q = np.where(j < len(sc), sc[np.minimum(j, len(sc) - 1)], np.inf)
    return S[test] <= q[:, None]


def naps_cp(S, y, cal, test, alpha, A2):
    """NAPS (Clarkson, 2023): calibrate each test node on the calibration nodes in its
    k-hop neighbourhood (uniform weights). A2 is the k-hop reachability matrix (csr, bool).
    Nodes with fewer than ceil(1/alpha)-1 neighbouring calibration nodes are not applicable;
    they receive the full label set and are flagged. Vectorised: calibration nodes are
    sorted by score, so the k-th stored column of a row gives the k-th smallest score.
    """
    n, K = S.shape
    sc = S[cal, y[cal]]
    o = np.argsort(sc, kind='stable'); cal_s = cal[o]; sc_s = sc[o]
    M = A2[test][:, cal_s].tocsr(); M.sort_indices()
    cnt = np.diff(M.indptr)
    k = np.ceil((cnt + 1) * (1 - alpha)).astype(int)
    ok = (k <= cnt) & (cnt > 0)
    q = np.full(len(test), np.inf)
    q[ok] = sc_s[M.indices[M.indptr[:-1][ok] + k[ok] - 1]]
    sets = S[test] <= q[:, None]
    return sets, ok


def khop_reach(edge_index, n, k=2):
    A = sp.csr_matrix((np.ones(edge_index.shape[1], dtype=bool), (edge_index[0], edge_index[1])), shape=(n, n))
    R = A.copy(); P = A.copy()
    for _ in range(k - 1):
        P = (P @ A).astype(bool); R = (R + P).astype(bool)
    R.setdiag(False); R.eliminate_zeros()
    return R.tocsr()
