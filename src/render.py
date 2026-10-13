"""Render every number, table and figure-data file of the paper from results/.

Outputs (never edit by hand) in paper/generated/:
  numbers.tex      \\nm{key} macros used in the prose
  tab_*.tex        table bodies
  *.dat            pgfplots data for the figures
Usage: python src/render.py
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.environ.get('ARTIFACT_RENDER_DIR', os.path.join(ROOT, 'outputs', 'intermediate'))
os.makedirs(GEN, exist_ok=True)
RNG = np.random.default_rng(0)
NUM = {}
HOMO = ['Amazon-Photo', 'Cora', 'PubMed', 'Amazon-Computers', 'CiteSeer', 'CoraFull']
HET_MC = ['Amazon-ratings', 'Squirrel-filtered', 'Chameleon-filtered', 'Roman-empire']
BIN = ['Tolokers', 'Questions', 'Minesweeper']
MC = HOMO + HET_MC
SHORT = {'Amazon-Photo': 'Photo', 'Amazon-Computers': 'Computers', 'Amazon-ratings': 'Amazon-ratings',
         'Squirrel-filtered': 'Squirrel-f', 'Chameleon-filtered': 'Chameleon-f', 'Roman-empire': 'Roman-empire',
         'Minesweeper': 'Minesweeper', 'Tolokers': 'Tolokers', 'Questions': 'Questions', 'CoraFull': 'CoraFull',
         'Cora': 'Cora', 'PubMed': 'PubMed', 'CiteSeer': 'CiteSeer'}
KEY = {d: d.replace('-', '').replace('filtered', 'f') for d in SHORT}


def nm(key, val, fmt='{:.2f}'):
    v = val if isinstance(val, str) else fmt.format(val)
    if v.startswith('-') and len(v) > 1 and (v[1].isdigit() or v[1] == '.'):
        v = '$-$' + v[1:]
    elif v.startswith('+') and len(v) > 1 and v[1].isdigit():
        v = '$+$' + v[1:]
    NUM[key] = v


def pct(x): return f'{100 * x:.1f}'


def boot_hw(x, B=1000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return np.nan
    means = x[RNG.integers(0, len(x), (B, len(x)))].mean(1)
    lo, hi = np.percentile(means, [2.5, 97.5]); return (hi - lo) / 2


def f3(x):
    if not np.isfinite(x): return '--'
    s = f'{abs(x):.3f}'.replace('0.', '.', 1)
    return ('$-$' if x < 0 else '') + s


def f2(x): return f'{x:.2f}' if np.isfinite(x) else '--'


# ------------------------------------------------------------------ loading
def load_cp():
    files = [f for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'cp', '*.csv.gz'))) if '_cal' not in f]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cf = sorted(glob.glob(os.path.join(ROOT, 'results', 'cfgnn', '*.csv.gz')))
    cfg = pd.concat([pd.read_csv(f) for f in cf], ignore_index=True) if cf else pd.DataFrame()
    if len(cfg):
        cfg = cfg.drop_duplicates(subset=['dataset', 'model', 'seed', 'split', 'alpha'])
    lf = os.path.join(ROOT, 'results', 'labelfree', 'gate.csv.gz')
    if os.path.exists(lf):
        g = pd.read_csv(lf); g['alpha'] = 0.05; g['method'] = 'LF-gate-bounded'
        df = pd.concat([df, g], ignore_index=True)
    return df, cfg


def meta_table(df):
    m = df.groupby('dataset').first()[['n', 'E_und', 'f', 'K', 'h_edge', 'h_adj', 'deg_mean']]
    lam = df[(df.method == 'APS') & (df.model == 'GCN')].groupby('dataset')[['h_adj_train']].mean()
    return m.join(lam).sort_values('h_adj', ascending=False)


# ------------------------------------------------------------------ Table: datasets
def tab_datasets(meta):
    """One readable six-column table with explicit benchmark groups."""
    def cells(ds, r):
        dag = '\\textdagger' if ds in BIN else ''
        return (f"{SHORT.get(ds, ds)}{dag} & {int(r.n):,} & {int(r.E_und):,} & {int(r.K)} & "
                f"{f3(r.h_adj)} & {f3(r.h_adj_train)}").replace(',', '\\,')
    homo = [(ds, r) for ds, r in meta.iterrows() if r.h_adj > 0.3]
    het = [(ds, r) for ds, r in meta.iterrows() if r.h_adj <= 0.3]
    lines = []
    for label, group in [('Homophilous benchmarks', homo), ('Heterophilous benchmarks', het)]:
        if lines: lines.append('\\midrule')
        if lines: lines.append('\\tablegroup{6}{' + label + '}')
        for ds, r in group:
            lines.append(cells(ds, r) + ' \\\\')
    for ds, r in meta.iterrows():
        nm(f'hadj{KEY[ds]}', r.h_adj, '{:.2f}'); nm(f'ahat{KEY[ds]}', r.h_adj_train, '{:.2f}')
    open(os.path.join(GEN, 'tab_datasets_rows.tex'), 'w').write('\n'.join(lines) + '\n')
    nm('nDatasets', str(len(meta)))


# ------------------------------------------------------------------ Table: main (GCN)
MAIN = [('APS', '\\aps'), ('DAPS', '\\daps'), ('DAPS-tuned', '\\daps-cal'), ('CF-GNN', '\\cfgnn'),
        ('HeAD-CP[signed]', '\\textsc{head}'), ('HG-DAPS', '\\textsc{hg}'), ('SNR-lambda', 'SNR-$\\lambda$'),
        ('DAPS-oracle', 'oracle')]


def tab_main(df, cfg, model='GCN', alpha=0.05):
    d = df[(df.model == model) & (df.alpha == alpha)]
    if len(cfg):
        d = pd.concat([d, cfg[(cfg.model == model) & (cfg.alpha == alpha)]], ignore_index=True)
    rows, hw_max, covs = [], 0.0, []
    piv_all = d.pivot_table(index=['dataset', 'seed', 'split'], columns='method', values='size')
    if len(cfg):
        nm('cfgnnRuns', str(int(cfg.groupby('dataset').size().min())))
    for ds in MC:
        sub = d[d.dataset == ds]
        if sub.empty: continue
        means = {m: sub[sub.method == m]['size'].mean() for m, _ in MAIN}
        valid = {m: v for m, v in means.items() if np.isfinite(v) and m != 'DAPS-oracle'}
        best = min(valid, key=valid.get)
        pv = piv_all.loc[ds]
        bold = set()
        for m in valid:
            if m == best: bold.add(m); continue
            diff = (pv[m] - pv[best]).dropna()
            if (diff == 0).all(): bold.add(m); continue
            try:
                if wilcoxon(diff).pvalue >= 0.01: bold.add(m)
            except ValueError:
                bold.add(m)
        cells = []
        for m, _ in MAIN:
            v = means[m]
            if not np.isfinite(v): cells.append('--'); continue
            s = sub[sub.method == m]
            if m != 'DAPS-oracle':
                hw_max = max(hw_max, boot_hw(s['size']))
                covs.append(s.coverage.mean())
            txt = f'{v:.2f}'
            cells.append(f'\\best{{{txt}}}' if m in bold else txt)
        rows.append(f"{SHORT[ds]} & {f3(sub.h_adj.iloc[0])} & " + ' & '.join(cells) + ' \\\\')
        if ds == HOMO[-1]: rows.append('\\midrule')
    open(os.path.join(GEN, 'tab_main_rows.tex'), 'w').write('\n'.join(rows) + '\n')
    nm('mainCIhw', hw_max, '{:.2f}')
    nm('mainCovMin', min(covs), '{:.3f}'); nm('mainCovMax', max(covs), '{:.3f}')
    # inflation of DAPS over APS, GCN
    shr = {}
    for ds in HOMO:
        sub = d[d.dataset == ds]
        if len(sub): shr[ds] = 1 - sub[sub.method == 'DAPS']['size'].mean() / sub[sub.method == 'APS']['size'].mean()
    if shr:
        k = max(shr, key=shr.get); nm('dapsShrinkMaxHomo', pct(shr[k]), '{}'); nm('dapsShrinkMaxHomoData', SHORT[k], '{}')
    for ds in MC:
        sub = d[d.dataset == ds]
        if sub.empty: continue
        a, b = sub[sub.method == 'APS'].set_index(['seed', 'split'])['size'], sub[sub.method == 'DAPS'].set_index(['seed', 'split'])['size']
        nm(f'dapsInfl{KEY[ds]}', pct(b.mean() / a.mean() - 1), '{}')
        try:
            p = wilcoxon(a.loc[b.index], b).pvalue
        except ValueError:
            p = 1.0
        allm = df[(df.dataset == ds) & (df.alpha == alpha)].groupby(['model', 'method'])['size'].mean().unstack()
        infl = allm['DAPS'] / allm['APS'] - 1
        nm(f'dapsInflMax{KEY[ds]}', pct(infl.max()), '{}'); nm(f'dapsInflMaxModel{KEY[ds]}', infl.idxmax(), '{}')
        nm(f'dapsP{KEY[ds]}', f'{p:.1e}' if p < 0.001 else f'{p:.3f}', '{}')


# ------------------------------------------------------------------ Table: selector regret (all models)
SELECTORS = [
    ('APS', '\\aps\\ ($\\lambda=0$)', 'none'),
    ('DAPS', '\\daps\\ ($\\lambda=0.5$)', 'none'),
    ('Random-grid', 'uniform random grid weight', 'none'),
    ('HA-CP[per-node soft]', 'soft agreement, node-wise', 'none'),
    ('HeAD-CP[signed]', '\\textsc{head-cp} signed', 'none'),
    ('HeAD-CP[edge]', '\\textsc{head-cp} edge', 'none'),
    ('HeAD-CP[v3]', '\\textsc{head-cp} v3', 'none'),
    ('HA-CP[global soft]', 'soft gate, self-normalised', 'none'),
    ('LF-gate-bounded', 'soft gate, Eq.~\\eqref{eq:adjhom} analogue', 'none'),
    ('DAPS-tuned', '\\daps, tuned on calibration', '$m/2$ cal.'),
    ('DAPS-valtuned', '\\daps, tuned on validation', 'val.'),
    ('HG-DAPS', '\\textsc{hg-daps}', 'train'),
    ('HA-CP', 'label gate, $0.75\\,\\mathrm{clip}(\\hat a)$', 'train'),
    ('SNR-lambda', 'SNR-$\\lambda$ \\eqref{eq:snrrule}', 'train+val'),
]


def regret_frame(df, alpha=0.05):
    d = df[(df.alpha == alpha) & (df.dataset.isin(MC))]
    piv = d.pivot_table(index=['dataset', 'model', 'seed', 'split'], columns='method', values='size')
    # Exact expected size under a uniform draw from the 11 stored weights.
    oracle = d[d.method == 'DAPS-oracle'].set_index(['dataset', 'model', 'seed', 'split'])
    grid = [f'size_lam{x:.1f}' for x in np.linspace(0, 1, 11)]
    assert oracle[grid].notna().all().all()
    assert np.allclose(oracle[grid].min(axis=1), oracle['size'])
    piv['Random-grid'] = oracle[grid].mean(axis=1)
    reg = piv.div(piv['DAPS-oracle'], axis=0) - 1
    cell = reg.groupby(level=['dataset', 'model']).mean()
    return cell


def tab_regret(df, alpha=0.05):
    cell = regret_frame(df, alpha)
    homo = cell.index.get_level_values(0).isin(HOMO)
    rows = []
    best_worst = min(cell[m].max() for m, _, _ in SELECTORS if m in cell)
    for m, label, lab in SELECTORS:
        if m not in cell: continue
        c = cell[m]
        within = (c <= 0.02).mean()
        worst = c.max()
        w = f'{pct(worst)}'
        if np.isclose(worst, best_worst): w = f'\\best{{{w}}}'
        group = {'APS': 'Fixed / random', 'HA-CP[per-node soft]': 'Label-free: node-wise',
                 'HA-CP[global soft]': 'Label-free: graph-level', 'DAPS-tuned': 'Label-based'}
        if m in group and m != 'APS': rows.append('\\tablegroup{7}{' + group[m] + '}')
        label = {'Random-grid': 'Uniform random grid',
                 'HA-CP[per-node soft]': 'Soft agreement',
                 'HA-CP[global soft]': 'Self-normalised gate',
                 'LF-gate-bounded': 'Bounded gate (Eq.~\\eqref{eq:adjhom})',
                 'DAPS-tuned': '\\daps-cal', 'DAPS-valtuned': '\\daps-val',
                 'HA-CP': 'Label gate'}.get(m, label)
        ci = boot_hw(c)
        rows.append(f"{label} & {lab} & ${pct(c.mean())} \\pm {pct(ci)}$ & {pct(c[homo].mean())} & {pct(c[~homo].mean())} & {w} & {100 * within:.0f} \\\\")
        if m == 'LF-gate-bounded': rows.append('\\midrule')
        if m in ['Random-grid', 'HeAD-CP[v3]']: rows.append('\\midrule')
        key = m.replace('[', '').replace(']', '').replace('-', '').replace(' ', '')
        nm(f'regMean{key}', pct(c.mean()), '{}'); nm(f'regWorst{key}', pct(worst), '{}')
        nm(f'regHomo{key}', pct(c[homo].mean()), '{}'); nm(f'regHet{key}', pct(c[~homo].mean()), '{}')
        nm(f'regWithin{key}', f'{100 * within:.0f}', '{}')
    open(os.path.join(GEN, 'tab_regret_rows.tex'), 'w').write('\n'.join(rows) + '\n')
    nm('nCells', str(len(cell)))
    for m in ['HeAD-CP[signed]', 'HA-CP[per-node soft]', 'APS', 'DAPS', 'HA-CP[global soft]', 'HG-DAPS', 'SNR-lambda', 'LF-gate-bounded']:
        if m in cell:
            ds, mo = cell[m].idxmax()
            key = m.replace('[', '').replace(']', '').replace('-', '').replace(' ', '')
            nm(f'regWorstCell{key}', f'{SHORT[ds]}/{mo}', '{}')
    # heterophilous cells where a selector beats the oracle (negative regret)
    het = ~homo
    mlp = cell.index.get_level_values(1) == 'MLP'
    for m in ['SNR-lambda', 'SNR-lambda[global]', 'DAPS-valtuned']:
        key = m.replace('[', '').replace(']', '').replace('-', '')
        nm(f'regMLPworst{key}', pct(cell.loc[mlp, m].max()), '{}'); nm(f'regGNNworst{key}', pct(cell.loc[~mlp, m].max()), '{}')
        nm(f'regMLPmean{key}', pct(cell.loc[mlp, m].mean()), '{}'); nm(f'regGNNmean{key}', pct(cell.loc[~mlp, m].mean()), '{}')
    nm('minRegretSNR', pct(cell['SNR-lambda'].min()), '{}'); nm('minRegretHead', pct(cell['HeAD-CP[signed]'].min()), '{}')
    nm('headSignedHetBelowOracle', f"{int((cell.loc[het, 'HeAD-CP[signed]'] < 0).sum())}", '{}')
    nm('nHetCells', str(int(het.sum())))
    # ablations of SNR-lambda
    for m in ['SNR-lambda[global]', 'SNR-lambda[rho=1]', 'SNR-lambda[all-label a]', 'SNR-lambda[2hop]', 'DAPS[2hop]',
              'HA-CP[lmax=0.5]', 'HA-CP[lmax=1]', 'HA-CP[h0=0.25]', 'HA-CP[all-label h_adj]', 'HA-CP[per-node hard]',
              'HA-CP[per-node label-oracle]']:
        if m in cell:
            key = m.replace('[', '').replace(']', '').replace('-', '').replace(' ', '').replace('=', '').replace('.', '').replace('_', '')
            nm(f'regMean{key}', pct(cell[m].mean()), '{}'); nm(f'regWorst{key}', pct(cell[m].max()), '{}')
    return cell


def revision_summaries(df, cfg):
    rows = []
    for alpha in (0.05, 0.1):
        c = regret_frame(df, alpha)
        for m, _, _ in SELECTORS:
            if m not in c: continue
            rows.append(dict(alpha=alpha, method=m, cells=int(c[m].count()),
                             mean_pct=100*c[m].mean(), worst_pct=100*c[m].max(),
                             within_two_pct=100*(c[m] <= .02).mean()))
        if alpha == .1:
            for m in ['DAPS-valtuned', 'SNR-lambda']:
                key=m.replace('-', '')
                nm('alphaTenMean'+key, pct(c[m].mean()), '{}')
                nm('alphaTenWorst'+key, pct(c[m].max()), '{}')
            g=df[(df.alpha==alpha)&(df.dataset=='Roman-empire')].groupby(['model','method'])['size'].mean().unstack()
            infl=100*(g.DAPS/g.APS-1)
            nm('alphaTenRomanMin', infl.min(), '{:.1f}')
            nm('alphaTenRomanMax', infl.max(), '{:.1f}')
    pd.DataFrame(rows).to_csv(os.path.join(GEN, 'revision_sensitivity.csv'), index=False)
    # Confirm complete pairing keys, not merely matching row counts.
    keys=['dataset','model','seed','split','alpha']
    base=df[(df.method=='APS')&(df.alpha==.05)&(df.model=='GCN')&df.dataset.isin(MC)]
    assert set(map(tuple, cfg[keys].to_numpy())) == set(map(tuple, base[keys].to_numpy()))


# ------------------------------------------------------------------ label-free failure diagnostics
def label_free(df):
    lf = os.path.join(ROOT, 'results', 'labelfree', 'labelfree.csv')
    if os.path.exists(lf):
        b = pd.read_csv(lf).groupby(['dataset', 'model'])[['soft_adj', 'h_adj']].mean().reset_index()
        with open(os.path.join(GEN, 'fig_labelfree.dat'), 'w') as f:
            f.write('hadj gsoft gnn\n')
            for _, r in b.iterrows():
                f.write(f"{r.h_adj:.4f} {r.soft_adj:.4f} {0 if r.model == 'MLP' else 1}\n")
        bc = b.set_index(['dataset', 'model']).soft_adj
        nm('lfCoraMLP', bc[('Cora', 'MLP')], '{:.2f}')
        nm('lfCoraGNNmin', min(bc[('Cora', m)] for m in ['GCN', 'GAT', 'SAGE', 'GCNII']), '{:.2f}')
        nm('lfCoraGNNmax', max(bc[('Cora', m)] for m in ['GCN', 'GAT', 'SAGE', 'GCNII']), '{:.2f}')
        hom = b[b.h_adj > 0.3]
        nm('lfHomoMLPgap', (hom[hom.model == 'MLP'].h_adj - hom[hom.model == 'MLP'].soft_adj).mean(), '{:.2f}')
        nm('lfRomanGCN', bc[('Roman-empire', 'GCN')], '{:.2f}')
    acc = df[(df.method == 'APS') & (df.alpha == 0.05)].groupby(['dataset', 'model']).acc.mean()
    for (ds, mo), v in acc.items():
        nm(f'acc{KEY[ds]}{mo}', v, '{:.2f}')
    g = df[(df.method == 'APS') & (df.alpha == 0.05) & df.dataset.isin(MC)].groupby(['dataset', 'model'])[['g_soft', 'h_adj', 'h_soft_label_corr', 'acc']].mean().reset_index()
    het = g[g.h_adj < 0.2]
    nm('gsoftHetGNNmean', het[het.model != 'MLP'].g_soft.mean(), '{:.2f}')
    nm('gsoftHetMLPmean', het[het.model == 'MLP'].g_soft.mean(), '{:.2f}')
    nm('gsoftHetGNNmax', het[het.model != 'MLP'].g_soft.max(), '{:.2f}')
    r = g[g.model == 'GCN'].set_index('dataset')
    for ds in ['Roman-empire', 'Amazon-ratings', 'Cora']:
        if ds in r.index: nm(f'gsoftGCN{KEY[ds]}', r.loc[ds, 'g_soft'], '{:.2f}')
    nm('nodeCorrHomo', g[g.h_adj > 0.3].h_soft_label_corr.mean(), '{:.2f}')
    nm('nodeCorrHet', g[g.h_adj < 0.2].h_soft_label_corr.mean(), '{:.2f}')


# ------------------------------------------------------------------ conditional coverage
COND = ['APS', 'DAPS', 'SNR-lambda']


def tab_cond(df, model='GCN', alpha=0.05):
    d = df[(df.model == model) & (df.alpha == alpha)]
    rows = []
    panels = {"class": [], "slice": []}
    order = ['Cora', 'Amazon-Photo', 'CoraFull', 'Amazon-ratings', 'Squirrel-filtered', 'Roman-empire'] + BIN
    for ds in order:
        sub = d[d.dataset == ds]
        if sub.empty: continue
        cells = []
        for m in COND:
            s = sub[sub.method == m]
            cells += [f3(s.coverage.mean()), f3(s.deg_min.mean()), f3(s.cls_min.mean()), f3(s.wsc.mean())]
        dag = '\\textdagger' if ds in BIN else ''
        rows.append(f"{SHORT[ds]}{dag} & " + ' & '.join(cells) + ' \\\\')
        for panel, indices in [('class', [0, 4, 8, 2, 6, 10]), ('slice', [1, 5, 9, 3, 7, 11])]:
            panels[panel].append(f"{SHORT[ds]}{dag} & " + ' & '.join(cells[i] for i in indices) + ' \\\\')
            if ds == 'CoraFull': panels[panel].append('\\midrule')
        if ds == 'CoraFull': rows.append('\\midrule')
    open(os.path.join(GEN, 'tab_cond_rows.tex'), 'w').write('\n'.join(rows) + '\n')
    for panel, lines in panels.items():
        open(os.path.join(GEN, f'tab_cond_{panel}_rows.tex'), 'w').write('\n'.join(lines) + '\n')
    a = d[d.method == 'APS']
    het = a[a.dataset.isin(HET_MC + BIN)]; hom = a[a.dataset.isin(HOMO)]
    nm('clsMinHet', het.groupby('dataset').cls_min.mean().min(), '{:.2f}')
    nm('clsMinHetData', SHORT[het.groupby('dataset').cls_min.mean().idxmin()], '{}')
    nm('clsMinHomo', hom.groupby('dataset').cls_min.mean().min(), '{:.2f}')
    nm('degMinHet', het.groupby('dataset').deg_min.mean().min(), '{:.2f}')
    nm('degMinHomo', hom.groupby('dataset').deg_min.mean().min(), '{:.2f}')
    dd = d[d.method.isin(['DAPS', 'SNR-lambda']) & d.dataset.isin(HOMO)].groupby('dataset').deg_min.mean()
    nm('degMinHomoDiff', dd.min(), '{:.2f}'); nm('degMinHomoDiffData', SHORT[dd.idxmin()], '{}')
    nap = df[(df.method == 'NAPS') & (df.model == model) & (df.alpha == alpha) & df.dataset.isin(MC)].groupby('dataset').naps_applicable.mean()
    nm('napsAppMin', pct(nap.min()), '{}'); nm('napsAppMax', pct(nap.max()), '{}')
    # degree-decile curves for the figure
    for ds in ['Cora', 'Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Questions', 'Tolokers']:
        sub = d[d.dataset == ds]
        if sub.empty: continue
        with open(os.path.join(GEN, f'fig_deg_{KEY[ds]}.dat'), 'w') as f:
            f.write('dec ' + ' '.join(m.replace('-', '') for m in COND + ['Mondrian-deg']) + '\n')
            for b in range(1, 11):
                vals = [sub[sub.method == m][f'cov_dec{b}'].mean() for m in COND + ['Mondrian-deg']]
                f.write(f'{b} ' + ' '.join(f'{v:.4f}' for v in vals) + '\n')
    # lowest decile, all methods, all datasets
    low = d[d.method.isin(COND + ['Mondrian-deg'])].groupby(['dataset', 'method']).cov_dec1.mean().unstack()
    for ds in low.index:
        for m in low.columns:
            nm(f'lowdec{KEY[ds]}{m.replace("-", "")}', low.loc[ds, m], '{:.3f}')


# ------------------------------------------------------------------ CSBM
def csbm():
    p = os.path.join(ROOT, 'results', 'csbm', 'crossover.csv')
    if not os.path.exists(p): return
    res = pd.read_csv(p)
    raw = pd.read_csv(os.path.join(ROOT, 'results', 'csbm', 'csbm.csv.gz'))
    raw = raw[raw.alpha == 0.05]
    # (a) set size vs h for K=2, dbar=10, sigma=1
    for K in (2, 5):
        g = raw[(raw.K == K) & (raw.dbar == 10) & (raw.sigma == 1.0)].groupby(['h', 'lam'])['size'].mean().unstack()
        g.columns = [f'l{int(100 * c)}' for c in g.columns]
        g.to_csv(os.path.join(GEN, f'fig_csbm_K{K}.dat'), sep=' ', index_label='h')
    # (b) crossover vs degree, K=2, sigma=1
    for lam in (0.1, 0.5):
        s = res[(res.K == 2) & (res.sigma == 1.0) & (res.lam == lam)].sort_values('dbar')
        s[['dbar', 'hstar', 'lo', 'hi', 'hstar_pred']].assign(elo=lambda x: x.hstar - x.lo, ehi=lambda x: x.hi - x.hstar) \
            .to_csv(os.path.join(GEN, f'fig_hstar_l{int(100 * lam)}.dat'), sep=' ', index=False)
    ok = res.dropna(subset=['hstar', 'hstar_pred'])
    for K in (2, 5):
        o = ok[ok.K == K]
        nm(f'predCorrK{K}', np.corrcoef(o.hstar, o.hstar_pred)[0, 1], '{:.2f}')
        nm(f'predMaeK{K}', (o.hstar - o.hstar_pred).abs().mean(), '{:.3f}')
        nm(f'predAboveK{K}', f'{100 * (o.hstar >= o.hstar_pred - 0.005).mean():.0f}', '{}')
    k2 = res[res.K == 2]
    nm('hstarMinK2', k2.hstar.min(), '{:.2f}'); nm('hstarMaxK2', k2.hstar.max(), '{:.2f}')
    nm('nAboveHalf', str(int((k2.hstar > 0.5).sum()))); nm('nConfigsK2', str(int(k2.hstar.notna().sum())))
    k5 = res[res.K == 5]
    nm('nAboveFifth', str(int((k5.hstar > 0.2).sum()))); nm('nConfigsK5', str(int(k5.hstar.notna().sum())))
    # no improvement at h <= 1/2 (K=2): share of (config, h<=0.5) cells where diffusion lowers size by > 1%
    base = raw[raw.lam == 0].groupby(['K', 'dbar', 'sigma', 'h'])['size'].mean()
    dif = raw[raw.lam > 0].groupby(['K', 'dbar', 'sigma', 'lam', 'h'])['size'].mean()
    rel = dif / base.reindex(dif.index.droplevel('lam')).values - 1
    rel = rel.reset_index()
    het2 = rel[(rel.K == 2) & (rel.h <= 0.5)]
    nm('shareHelpHet', f'{100 * (het2['size'] < -0.01).mean():.1f}', '{}')
    nm('maxInflK2', pct(rel[rel.K == 2]['size'].max()), '{}')
    nm('nGraphs', f"{raw.groupby(['K', 'h', 'dbar', 'sigma', 'seed']).ngroups:,}".replace(',', '\\,'), '{}')
    e = raw[raw.lam == 0]
    cc = raw.groupby(['K', 'dbar', 'sigma', 'lam']).coverage.mean()
    nm('csbmCovCfgMin', cc.min(), '{:.3f}'); nm('csbmCovCfgMax', cc.max(), '{:.3f}')
    a = raw[(raw.K == 2) & (raw.dbar == 10) & (raw.sigma == 1.0) & (raw.h == 0)].groupby('lam')['size'].mean()
    nm('figAinflHzero', pct(a[1.0] / a[0.0] - 1), '{}')
    for K in (2, 5):
        o = ok[ok.K == K]
        nm(f'predAboveRawK{K}', f'{100 * (o.hstar >= o.hstar_pred).mean():.0f}', '{}')


# ------------------------------------------------------------------ shift
def shift():
    fs = sorted(glob.glob(os.path.join(ROOT, 'results', 'shift', '*.csv.gz')))
    if not fs: return
    d = pd.concat([pd.read_csv(f) for f in fs])
    ess = d[d.split_name != 'reference'].groupby(['dataset', 'split_name']).ess.mean()
    nm('shiftEssMin', ess.min(), '{:.0f}'); nm('shiftEssMax', ess.max(), '{:.0f}')
    meths = ['APS', 'DAPS', 'SNR-lambda', 'SNR-lambda+weighted', 'SNR-lambda+Mondrian']
    rows = []
    for (ds, sc), g in d.groupby(['dataset', 'scenario'], sort=False):
        for sn in [s for s in g.split_name.unique() if s != 'reference'] + ['reference']:
            gg = g[g.split_name == sn]
            label = {'shift': '$\\le$2017 $\\to$ $\\ge$2019', 'reference': 'uniform (reference)'}.get(sn, sn.replace('->', '$\\to$'))
            cells = [f3(gg[gg.method == m].coverage.mean()) for m in meths]
            rows.append(f"{SHORT.get(ds, ds)} & {label} & " + ' & '.join(cells) + ' \\\\')
        rows.append('\\midrule')
    rows = rows[:-1]
    open(os.path.join(GEN, 'tab_shift_rows.tex'), 'w').write('\n'.join(rows) + '\n')
    for (ds, sn, m), v in d.groupby(['dataset', 'split_name', 'method']).coverage.mean().items():
        nm(f'shift{KEY.get(ds, ds.replace("-", ""))}{sn.replace("->", "to").replace("-", "")}{m.replace("-", "").replace("+", "")}', v, '{:.3f}')


# ------------------------------------------------------------------ calibration size (E7)
def calsize():
    fs = sorted(glob.glob(os.path.join(ROOT, 'results', 'cp', '*_cal.csv.gz')))
    if not fs: return
    d = pd.concat([pd.read_csv(f) for f in fs]); d = d[(d.alpha == 0.05) & (d.method == 'APS')]
    g = d.groupby(['dataset', 'm_cal']).coverage.std().unstack(0)
    g.columns = [KEY[c] for c in g.columns]
    g.to_csv(os.path.join(GEN, 'fig_calsize.dat'), sep=' ', index_label='m')
    for ds, s in d.groupby('dataset'):
        for m, v in s.groupby('m_cal').coverage.std().items():
            nm(f'calsd{KEY[ds]}m{m}', 100 * v, '{:.1f}')


# ------------------------------------------------------------------ class-conditional remedy
def classwise():
    p = os.path.join(ROOT, 'results', 'classwise', 'classwise.csv.gz')
    if not os.path.exists(p): return
    d = pd.read_csv(p).groupby(['dataset', 'method'])[['coverage', 'size', 'cls_min']].mean()
    for (ds, m), r in d.iterrows():
        k = KEY[ds] + m.replace('-', '')
        nm(f'cw{k}cls', r.cls_min, '{:.3f}'); nm(f'cw{k}size', r['size'], '{:.2f}')
    het = [x for x in HET_MC + BIN if x in d.index.get_level_values(0)]
    nm('cwHetClsMin', d.loc[(het, 'APS-classwise'), 'cls_min'].min(), '{:.2f}')


# ------------------------------------------------------------------ E0
def e0():
    p = os.path.join(ROOT, 'results', 'e0', 'daps_repro.csv')
    if not os.path.exists(p): return
    d = pd.read_csv(p); d['delta'] = d.daps_size - d.aps_size
    g = d.groupby('dataset').agg(delta=('delta', 'mean'), rep=('reported_delta', 'first'), aps=('aps_size', 'mean'),
                                 cov=('daps_cov', 'mean'), acc=('acc', 'mean'))
    for ds, r in g.iterrows():
        k = ds.replace('-', '')
        nm(f'eZero{k}', r.delta, '{:+.2f}'); nm(f'eZeroRep{k}', r.rep, '{:+.2f}')
    q = os.path.join(ROOT, 'results', 'e0', 'cfgnn_official_coraml.csv')
    if os.path.exists(q):
        c = pd.read_csv(q).mean()
        nm('eZeroCFbase', c.base_aps_size, '{:.2f}'); nm('eZeroCF', c.cfgnn_size, '{:.2f}')
    rows = [f"{ds} & {r.acc:.3f} & {r.aps:.2f} & {r.delta:+.2f} & {r.rep:+.2f} \\\\" for ds, r in g.iterrows()]
    open(os.path.join(GEN, 'tab_e0_rows.tex'), 'w').write('\n'.join(rows) + '\n')


def write_numbers():
    with open(os.path.join(GEN, 'numbers.tex'), 'w') as f:
        f.write('% generated by src/render.py -- do not edit\n')
        f.write('\\makeatletter\\def\\nm#1{\\@ifundefined{nm@#1}{\\textbf{??#1??}}{\\csname nm@#1\\endcsname}}\\makeatother\n')
        for k, v in sorted(NUM.items()):
            f.write(f'\\expandafter\\def\\csname nm@{k}\\endcsname{{{v}}}\n')


if __name__ == '__main__':
    df, cfg = load_cp()
    meta = meta_table(df)
    tab_datasets(meta)
    tab_main(df, cfg)
    tab_regret(df)
    revision_summaries(df, cfg)
    label_free(df)
    tab_cond(df)
    csbm(); shift(); calsize(); e0(); classwise()
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load_dataset
    yq = load_dataset('Questions').y.numpy()
    nm('questionsMinority', f'{100 * min(np.bincount(yq)) / len(yq):.0f}', '{}')
    write_numbers()
    print(f'{len(NUM)} numbers written; datasets: {sorted(df.dataset.unique())}')
