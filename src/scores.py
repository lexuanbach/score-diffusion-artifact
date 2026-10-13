"""Non-conformity scores (lower = more conforming) and graph score transformations.

All functions take/return numpy arrays of shape (n, K).
"""
import numpy as np
import scipy.sparse as sp


def softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def thr(P):
    """Sadinle et al. (2019): 1 - pi_k."""
    return 1.0 - P


def aps(P, u):
    """Romano et al. (2020), randomised: mass of strictly more likely classes + u * pi_k.

    u has shape (n,) (one uniform per node, Romano's original) or (n, K).
    """
    order = np.argsort(-P, axis=1, kind='stable')
    Ps = np.take_along_axis(P, order, 1)
    cum_before = np.cumsum(Ps, 1) - Ps
    S = np.empty_like(P)
    np.put_along_axis(S, order, cum_before, 1)
    uu = u[:, None] if u.ndim == 1 else u
    return S + uu * P


def raps(P, u, lam=0.01, k_reg=2):
    """Angelopoulos et al. (2021): APS + lam * (rank - k_reg)_+ with 1-based rank."""
    rank = np.argsort(np.argsort(-P, axis=1, kind='stable'), axis=1) + 1
    return aps(P, u) + lam * np.maximum(rank - k_reg, 0)


def norm_adj(edge_index, n):
    """Row-normalised adjacency D^{-1}A (no self loops). Isolated rows are left empty."""
    A = sp.csr_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(n, n))
    d = np.asarray(A.sum(1)).ravel()
    Dinv = sp.diags(np.where(d > 0, 1.0 / np.maximum(d, 1), 0.0))
    return (Dinv @ A).tocsr(), d


def neighbour_mean(Ahat, d, S):
    """(D^{-1} A) S, with isolated nodes keeping their own score (nothing to diffuse)."""
    M = Ahat @ S
    iso = d == 0
    if iso.any():
        M[iso] = S[iso]
    return M


def diffuse(S, Ahat, d, lam):
    """DAPS (Zargarbashi et al., 2023): (1 - lam) S + lam D^{-1} A S; lam scalar or per-node (n,)."""
    lam = np.asarray(lam, dtype=float)
    lam = lam[:, None] if lam.ndim == 1 else lam
    return (1 - lam) * S + lam * neighbour_mean(Ahat, d, S)


def diffuse_2hop(S, Ahat, d, lam):
    """2-hop variant: neighbour term replaced by the mean of 1-hop and 2-hop averages."""
    M1 = neighbour_mean(Ahat, d, S)
    M2 = neighbour_mean(Ahat, d, M1)
    lam = np.asarray(lam, dtype=float)
    lam = lam[:, None] if lam.ndim == 1 else lam
    return (1 - lam) * S + lam * 0.5 * (M1 + M2)


def soft_homophily(P, edge_index, n):
    """h_v = mean_{w in N(v)} <pi(v), pi(w)>; isolated nodes get 0 (PLAN Sect. 3.2)."""
    src, dst = edge_index
    dots = np.einsum('ij,ij->i', P[src], P[dst])
    num = np.bincount(src, weights=dots, minlength=n)
    d = np.bincount(src, minlength=n)
    return np.where(d > 0, num / np.maximum(d, 1), 0.0)


def hard_homophily(P, edge_index, n):
    yhat = P.argmax(1); src, dst = edge_index
    same = (yhat[src] == yhat[dst]).astype(float)
    num = np.bincount(src, weights=same, minlength=n)
    d = np.bincount(src, minlength=n)
    return np.where(d > 0, num / np.maximum(d, 1), 0.0)


def ha_lambda(h, h0=0.5, lam_max=1.0):
    """HA-CP per-node mixing weight lam_v = lam_max * clip((h_v - h0) / (1 - h0), 0, 1)."""
    return lam_max * np.clip((h - h0) / (1 - h0), 0.0, 1.0)


def soft_adjusted_global(P, edge_index):
    """Label-free graph-level homophily of the predictions (ablation only): adjusted-homophily
    analogue with <pi(u), pi(w)> as edge agreement and the predicted class marginal as chance."""
    src, dst = edge_index
    e = np.einsum('ij,ij->i', P[src], P[dst]).mean()
    pbar = P.mean(0); chance = (pbar ** 2).sum(); self_ = (P * P).sum(1).mean()
    return float((e - chance) / max(self_ - chance, 1e-9))


# ---------------------------------------------------------------- HeAD-CP (Lam & Anh, arXiv 2607.25273)
def headcp_signed(S, P, edge_index, Ahat, d, gmax=0.5):
    """Signed-gamma variant: gamma_v = gmax (2 h_v - 1), h_v = Laplace-smoothed pseudo-label agreement."""
    n = P.shape[0]; yhat = P.argmax(1); src, dst = edge_index
    agree = np.bincount(src, weights=(yhat[src] == yhat[dst]).astype(float), minlength=n)
    h = (agree + 1) / (d + 2)
    return diffuse(S, Ahat, d, gmax * (2 * h - 1))


def headcp_edge(S, P, edge_index, d, eta=0.5):
    """Edge-compatibility variant: s + eta/|N| sum_u rho(u,v) (s_u - s_v), rho = (<p_u,p_v> - 1/K)/(1 - 1/K)."""
    n, K = P.shape; src, dst = edge_index
    rho = (np.einsum('ij,ij->i', P[src], P[dst]) - 1 / K) / (1 - 1 / K)
    W = sp.csr_matrix((rho, (src, dst)), shape=(n, n))
    rsum = np.asarray(W.sum(1)).ravel()
    inv = np.where(d > 0, 1 / np.maximum(d, 1), 0.0)
    return S + eta * inv[:, None] * (W @ S - rsum[:, None] * S)


def headcp_v3(S, P, edge_index, Ahat, d, gbase=0.5, gvar=0.5):
    """DAPS-with-correction variant: gamma_v = gbase + gvar * h~_v * conf_v."""
    n, K = P.shape
    h_t = (soft_homophily(P, edge_index, n) - 1 / K) / (1 - 1 / K)
    conf = (P.max(1) - 1 / K) / (1 - 1 / K)
    return diffuse(S, Ahat, d, gbase + gvar * h_t * conf)


def hg_daps_delta(h_adj_ctx, m, dmax=0.5, m0=100):
    """HG-DAPS gate (Long et al., arXiv 2610.08564, Eq. 3): delta = dmax clip(h_adj * m/(m+m0), 0, 1)."""
    if m == 0:
        return 0.0
    return dmax * float(np.clip(h_adj_ctx * m / (m + m0), 0, 1))


def snr_lambda(S, y, val_idx, a_hat, h_edge_hat, d, use_degree=True, use_rho=True):
    """SNR-optimal weight (Proposition 1): lambda_v = a+ / (a+ + rho^2 / d_v).

    a_hat, h_edge_hat: adjusted / edge homophily of training edges; Delta, tau: APS score separation and
    true-label s.d. on validation nodes. Returns (lambda vector, rho^2)."""
    K = S.shape[1]
    st = S[val_idx, y[val_idx]]
    wm = np.ones((len(val_idx), K), bool); wm[np.arange(len(val_idx)), y[val_idx]] = False
    Dl, tau = S[val_idx][wm].mean() - st.mean(), st.std()
    rho2 = 1 + h_edge_hat * (1 - h_edge_hat) * Dl ** 2 / tau ** 2 if use_rho else 1.0
    a = max(a_hat, 0.0)
    dd = np.maximum(d, 1) if use_degree else np.full_like(d, max(d[d > 0].mean(), 1), dtype=float)
    lam = np.where(d > 0, a / (a + rho2 / dd + 1e-12), 0.0)
    return lam, rho2
