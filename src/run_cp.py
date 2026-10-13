"""Conformal stage (E2, E3, E4, E6, E7): every method on cached base-classifier outputs.

For each (dataset, model, seed) and each of 10 calibration/test resamples, all methods
share the same split and the same APS randomisation, so comparisons are paired.
Writes one row per (dataset, model, seed, split, alpha, method) to results/cp/<dataset>.csv.gz

Usage: python src/run_cp.py --datasets Cora --models GCN --seeds 0 1 2 3 4 --splits 10
"""
import argparse, os, sys, time, json, subprocess
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_dataset, cal_test_split, adjusted_homophily, edge_homophily, local_label_homophily, BINARY, train_adjusted_homophily
from scores import (softmax, thr, aps, raps, norm_adj, diffuse, diffuse_2hop, soft_homophily,
                    hard_homophily, ha_lambda, neighbour_mean, soft_adjusted_global,
                    headcp_signed, headcp_edge, headcp_v3, hg_daps_delta, snr_lambda)
from conformal import split_cp, mondrian_cp, naps_cp, khop_reach, conformal_quantile
from metrics import basic, sscv, grouped, wsc, equal_count_bins

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAM_GRID = np.round(np.linspace(0, 1, 11), 2)
DAPS_LAM = 0.5          # DAPS paper: optimum "around 0.5 for most datasets and models"
H0, LAM_MAX = 0.0, 0.75  # HA-CP constants, frozen on Cora + Roman-empire (logs/FREEZE.md); never tuned per dataset


def git_hash():
    try:
        return subprocess.check_output(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        return 'nogit'


def degree_groups(d, nbins=10):
    """Mondrian strata: degree deciles over all nodes, duplicate edges merged."""
    cuts = np.unique(np.quantile(d, np.linspace(0, 1, nbins + 1)[1:-1]))
    return np.digitize(d, cuts, right=True)


def feature_embedding(x, dim=20, seed=0):
    from sklearn.decomposition import TruncatedSVD
    x = x - x.mean(0)
    if x.shape[1] > dim:
        x = TruncatedSVD(dim, random_state=seed).fit_transform(x)
    return x / (x.std(0) + 1e-9)


WSC_METHODS = {'APS', 'DAPS', 'SNR-lambda', 'HG-DAPS', 'Mondrian-deg', 'SNR-lambda+Mondrian', 'DAPS-tuned', 'CF-GNN', 'HeAD-CP[signed]'}


def evaluate(sets, y_test, alpha, K, deg_bins, cls, Z, rng, full=True, with_wsc=True):
    cov, size = basic(sets, y_test)
    r = {'coverage': cov.mean(), 'size': size.mean(), 'singleton': (size == 1).mean(), 'empty': (size == 0).mean()}
    if not full:
        return r, cov, size
    r['sscv'] = sscv(cov, size, alpha, K)
    dev, mn, vals = grouped(cov, deg_bins, alpha)
    r['deg_dev'], r['deg_min'] = dev, mn
    for b in range(10):
        r[f'cov_dec{b+1}'] = vals.get(b, np.nan)
        r[f'size_dec{b+1}'] = size[deg_bins == b].mean() if (deg_bins == b).any() else np.nan
    dev, mn, _ = grouped(cov, cls, alpha)
    r['cls_dev'], r['cls_min'] = dev, mn
    r['wsc'] = wsc(cov.astype(float), Z, rng, n_dirs=200) if with_wsc else np.nan
    return r, cov, size


def run_dataset(ds, models, seeds, n_splits, alphas, cal_sizes, out_path, extra_cal_sizes=False):
    data = load_dataset(ds)
    n, K = data.num_nodes, data.num_classes
    ei = data.edge_index.numpy(); y = data.y.numpy()
    Ahat, d = norm_adj(ei, n)
    Z = feature_embedding(data.x.numpy())
    h_label = local_label_homophily(data.edge_index, data.y, n).numpy()
    groups = degree_groups(d)
    A2 = khop_reach(ei, n, 2) if n <= 60000 else None
    meta = {'dataset': ds, 'n': n, 'K': K, 'E_und': ei.shape[1] // 2, 'f': int(data.x.shape[1]),
            'h_edge': edge_homophily(data.edge_index, data.y), 'h_adj': adjusted_homophily(data.edge_index, data.y),
            'deg_mean': float(d.mean()), 'deg_median': float(np.median(d)), 'isolated': int((d == 0).sum())}
    rows, code = [], git_hash()
    for model in models:
        for seed in seeds:
            pth = os.path.join(ROOT, 'results', 'preds', ds, f'{model}_s{seed}.pt')
            if not os.path.exists(pth):
                print('missing', pth, flush=True); continue
            ck = torch.load(pth, weights_only=False)
            P = softmax(ck['logits'].numpy().astype(np.float64))
            used = np.zeros(n, bool); used[ck['train_idx'].numpy()] = True; used[ck['val_idx'].numpy()] = True
            pool = np.nonzero(~used)[0]
            acc = (P.argmax(1)[pool] == y[pool]).mean()
            t0 = time.time()
            h_tr = train_adjusted_homophily(ei, y, ck['train_idx'].numpy(), n)
            lam_ha = LAM_MAX * float(np.clip((h_tr - H0) / (1 - H0), 0, 1))
            trm = np.zeros(n, bool); trm[ck['train_idx'].numpy()] = True
            m_ctx = int((trm[ei[0]] & trm[ei[1]]).sum() // 2)
            lam_hg = hg_daps_delta(h_tr, m_ctx)
            t_hacp_prep = time.time() - t0
            h_soft = soft_homophily(P, ei, n)
            h_hard = hard_homophily(P, ei, n)
            g_soft = soft_adjusted_global(P, ei)
            h_adj_all = meta['h_adj']
            variants_lam = {
                'HA-CP[h0=0.25]': LAM_MAX * float(np.clip((h_tr - 0.25) / 0.75, 0, 1)),
                'HA-CP[lmax=0.5]': 0.5 * float(np.clip(h_tr, 0, 1)),
                'HA-CP[lmax=1]': 1.0 * float(np.clip(h_tr, 0, 1)),
                'HA-CP[all-label h_adj]': LAM_MAX * float(np.clip(h_adj_all, 0, 1)),
                'HA-CP[global soft]': LAM_MAX * float(np.clip(g_soft, 0, 1)),
                'HA-CP[per-node soft]': ha_lambda(h_soft, 0.5, 1.0),
                'HA-CP[per-node hard]': ha_lambda(h_hard, 0.5, 1.0),
                'HA-CP[per-node label-oracle]': ha_lambda(h_label, 0.5, 1.0),
            }
            va_idx = ck['val_idx'].numpy()
            h_edge_tr = float((y[ei[0][trm[ei[0]] & trm[ei[1]]]] == y[ei[1][trm[ei[0]] & trm[ei[1]]]]).mean()) if m_ctx else 0.5
            base = dict(dataset=ds, model=model, seed=seed, acc=acc, code=code,
                        h_adj_train=h_tr, lam_hacp=lam_ha, lam_hg=lam_hg, m_ctx=m_ctx, g_soft=g_soft, h_soft_mean=h_soft.mean(),
                        h_soft_label_corr=np.corrcoef(h_soft[d > 0], h_label[d > 0])[0, 1])
            for m_cal in cal_sizes:
                m_eff = min(m_cal, len(pool) // 2)
                for r in range(n_splits):
                    rng = np.random.default_rng(1000 * seed + r + 7 * m_cal)
                    cal, test = cal_test_split(pool, m_eff, seed * 100 + r + 7 * m_cal)
                    tune, cal2 = cal[: m_eff // 2], cal[m_eff // 2:]
                    u = rng.random(n)
                    S = aps(P, u)
                    t0 = time.time()
                    lam_snr, rho2 = snr_lambda(S, y, va_idx, h_tr, h_edge_tr, d)
                    t_snr = time.time() - t0
                    lam_snr_g, _ = snr_lambda(S, y, va_idx, h_tr, h_edge_tr, d, use_degree=False)
                    lam_snr_r, _ = snr_lambda(S, y, va_idx, h_tr, h_edge_tr, d, use_rho=False)
                    lam_snr_o, _ = snr_lambda(S, y, va_idx, h_adj_all, h_edge_tr, d)
                    deg_bins = equal_count_bins(d[test], 10, rng)
                    yt = y[test]
                    full_eval = (m_cal == cal_sizes[0])
                    for alpha in alphas:
                        def emit(method, sets, extra=None, full=True, t=np.nan):
                            res, _, _ = evaluate(sets, yt, alpha, K, deg_bins, yt, Z, rng, full=full and full_eval, with_wsc=method in WSC_METHODS)
                            row = dict(base, split=r, alpha=alpha, m_cal=m_eff, method=method, runtime=t, **res)
                            if extra: row.update(extra)
                            rows.append(row)
                        t0 = time.time(); sets, _ = split_cp(S, y, cal, test, alpha); emit('APS', sets, t=time.time() - t0)
                        if not full_eval:   # E7: calibration-size sweep, APS/DAPS/HA-CP only
                            emit('DAPS', split_cp(diffuse(S, Ahat, d, DAPS_LAM), y, cal, test, alpha)[0], full=False)
                            emit('HA-CP', split_cp(diffuse(S, Ahat, d, lam_ha), y, cal, test, alpha)[0], full=False)
                            emit('SNR-lambda', split_cp(diffuse(S, Ahat, d, lam_snr), y, cal, test, alpha)[0], full=False)
                            emit('DAPS-tuned', split_cp(diffuse(S, Ahat, d, 0.5), y, cal2, test, alpha)[0], full=False)
                            continue
                        emit('THR', split_cp(thr(P), y, cal, test, alpha)[0])
                        emit('RAPS', split_cp(raps(P, u), y, cal, test, alpha)[0])
                        t0 = time.time(); sets, _ = split_cp(diffuse(S, Ahat, d, DAPS_LAM), y, cal, test, alpha)
                        emit('DAPS', sets, t=time.time() - t0)
                        # DAPS with lambda tuned on half the calibration labels (DAPS paper, Sect. 6 / App. B)
                        t0 = time.time(); best = (np.inf, 0.0)
                        for lam in LAM_GRID:
                            Sl = diffuse(S, Ahat, d, lam)
                            q = conformal_quantile(Sl[tune, y[tune]], alpha)
                            sz = (Sl[tune] <= q).sum(1).mean()
                            if sz < best[0] - 1e-12: best = (sz, lam)
                        sets, _ = split_cp(diffuse(S, Ahat, d, best[1]), y, cal2, test, alpha)
                        emit('DAPS-tuned', sets, {'lam_sel': best[1]}, t=time.time() - t0)
                        # Retrospective oracle: minimizes test set sizes (no test labels used in selection).
                        sizes = {}
                        for lam in LAM_GRID:
                            st, _ = split_cp(diffuse(S, Ahat, d, lam), y, cal, test, alpha)
                            sizes[lam] = st.sum(1).mean()
                        lam_star = min(sizes, key=lambda k: (sizes[k], k))
                        st, _ = split_cp(diffuse(S, Ahat, d, lam_star), y, cal, test, alpha)
                        emit('DAPS-oracle', st, dict({'lam_sel': lam_star}, **{f'size_lam{l:.1f}': v for l, v in sizes.items()}), full=False)
                        t0 = time.time(); sets, _ = split_cp(diffuse(S, Ahat, d, lam_ha), y, cal, test, alpha)
                        emit('HA-CP', sets, {'lam_sel': lam_ha}, t=time.time() - t0 + t_hacp_prep)
                        for name, lam in variants_lam.items():
                            emit(name, split_cp(diffuse(S, Ahat, d, lam), y, cal, test, alpha)[0], full=False)
                        emit('HA-CP[2hop]', split_cp(diffuse_2hop(S, Ahat, d, lam_ha), y, cal, test, alpha)[0], full=False)
                        # lambda tuned on the validation labels (already spent on early stopping; disjoint from cal/test)
                        bv = (np.inf, 0.0)
                        for lam in LAM_GRID:
                            Sl = diffuse(S, Ahat, d, lam)
                            q = conformal_quantile(Sl[va_idx, y[va_idx]], alpha)
                            sz = (Sl[va_idx] <= q).sum(1).mean()
                            if sz < bv[0] - 1e-12: bv = (sz, lam)
                        emit('DAPS-valtuned', split_cp(diffuse(S, Ahat, d, bv[1]), y, cal, test, alpha)[0], {'lam_sel': bv[1]}, full=False)
                        emit('DAPS[2hop]', split_cp(diffuse_2hop(S, Ahat, d, DAPS_LAM), y, cal, test, alpha)[0], full=False)
                        emit('SNR-lambda', split_cp(diffuse(S, Ahat, d, lam_snr), y, cal, test, alpha)[0],
                             {'lam_sel': float(lam_snr.mean()), 'rho2': rho2}, t=t_snr)
                        emit('SNR-lambda[global]', split_cp(diffuse(S, Ahat, d, lam_snr_g), y, cal, test, alpha)[0], full=False)
                        emit('SNR-lambda[rho=1]', split_cp(diffuse(S, Ahat, d, lam_snr_r), y, cal, test, alpha)[0], full=False)
                        emit('SNR-lambda[all-label a]', split_cp(diffuse(S, Ahat, d, lam_snr_o), y, cal, test, alpha)[0], full=False)
                        emit('SNR-lambda[2hop]', split_cp(diffuse_2hop(S, Ahat, d, lam_snr), y, cal, test, alpha)[0], full=False)
                        emit('SNR-lambda+Mondrian', mondrian_cp(diffuse(S, Ahat, d, lam_snr), y, cal, test, alpha, groups))
                        emit('HG-DAPS', split_cp(diffuse(S, Ahat, d, lam_hg), y, cal, test, alpha)[0], {'lam_sel': lam_hg})
                        emit('HeAD-CP[signed]', split_cp(headcp_signed(S, P, ei, Ahat, d), y, cal, test, alpha)[0])
                        emit('HeAD-CP[edge]', split_cp(headcp_edge(S, P, ei, d), y, cal, test, alpha)[0], full=False)
                        emit('HeAD-CP[v3]', split_cp(headcp_v3(S, P, ei, Ahat, d), y, cal, test, alpha)[0], full=False)
                        emit('Mondrian-deg', mondrian_cp(S, y, cal, test, alpha, groups))
                        emit('HA-CP+Mondrian', mondrian_cp(diffuse(S, Ahat, d, lam_ha), y, cal, test, alpha, groups))
                        if A2 is not None:
                            sets, ok = naps_cp(S, y, cal, test, alpha, A2)
                            res = {'naps_applicable': ok.mean(),
                                   'naps_size_applicable': sets[ok].sum(1).mean() if ok.any() else np.nan,
                                   'naps_cov_applicable': sets[ok][np.arange(ok.sum()), yt[ok]].mean() if ok.any() else np.nan}
                            emit('NAPS', sets, res, full=False)
            print(f'{ds} {model} s{seed} acc={acc:.3f} rows={len(rows)}', flush=True)
    df = pd.DataFrame(rows)
    for k, v in meta.items():
        df[k] = v
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False, compression='gzip')
    with open(out_path.replace('.csv.gz', '.meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--models', nargs='+', default=['GCN', 'GAT', 'SAGE', 'GCNII', 'MLP'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--splits', type=int, default=10)
    ap.add_argument('--alphas', nargs='+', type=float, default=[0.05, 0.1])
    ap.add_argument('--cal-sizes', nargs='+', type=int, default=[1000])
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    for ds in a.datasets:
        out = os.path.join(ROOT, 'results', 'cp', f'{ds}{a.tag}.csv.gz')
        run_dataset(ds, a.models, a.seeds, a.splits, a.alphas, a.cal_sizes, out)


if __name__ == '__main__':
    main()
