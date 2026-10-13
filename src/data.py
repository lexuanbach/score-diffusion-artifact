"""Dataset loaders, homophily measures and split logic.

Every graph is returned as an undirected, self-loop-free PyG Data object with
fields x, y, edge_index, num_classes and, where the source provides them,
train/val/test masks (unused: we draw our own splits, see make_split).
"""
import os
import numpy as np
import torch
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

HOMOPHILOUS = ['Amazon-Photo', 'Cora', 'PubMed', 'Amazon-Computers', 'CiteSeer', 'CoraFull']
HETEROPHILOUS = ['Amazon-ratings', 'Tolokers', 'Chameleon-filtered', 'Squirrel-filtered',
                 'Questions', 'Minesweeper', 'Roman-empire']
SHIFT = ['ogbn-arxiv']
DATASETS = HOMOPHILOUS + HETEROPHILOUS + SHIFT
BINARY = {'Tolokers', 'Questions', 'Minesweeper'}

PLATONOV_URL = 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data'


def _clean(data):
    ei, _ = remove_self_loops(data.edge_index)
    ei = to_undirected(ei, num_nodes=data.num_nodes)
    data.edge_index = coalesce(ei, num_nodes=data.num_nodes)
    data.y = data.y.view(-1).long()
    data.x = data.x.float()
    data.num_classes = int(data.y.max()) + 1
    return data


def _load_npz_platonov(name):
    """Filtered Chameleon/Squirrel from Platonov et al. (ICLR 2023)."""
    from torch_geometric.data import Data
    fname = {'Chameleon-filtered': 'chameleon_filtered.npz',
             'Squirrel-filtered': 'squirrel_filtered.npz'}[name]
    path = os.path.join(ROOT, 'platonov', fname)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(f'{PLATONOV_URL}/{fname}', path)
    z = np.load(path)
    return Data(x=torch.tensor(z['node_features']), y=torch.tensor(z['node_labels']),
                edge_index=torch.tensor(z['edges']).t().contiguous())


def load_dataset(name):
    import torch_geometric.datasets as D
    if name in ('Cora', 'CiteSeer', 'PubMed'):
        data = D.Planetoid(os.path.join(ROOT, 'planetoid'), name)[0]
    elif name == 'Amazon-Photo':
        data = D.Amazon(os.path.join(ROOT, 'amazon'), 'Photo')[0]
    elif name == 'Amazon-Computers':
        data = D.Amazon(os.path.join(ROOT, 'amazon'), 'Computers')[0]
    elif name == 'CoraFull':
        data = D.CoraFull(os.path.join(ROOT, 'corafull'))[0]
    elif name in ('Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions'):
        data = D.HeterophilousGraphDataset(os.path.join(ROOT, 'hetero'), name)[0]
    elif name in ('Chameleon-filtered', 'Squirrel-filtered'):
        data = _load_npz_platonov(name)
    elif name == 'Flickr':
        data = D.Flickr(os.path.join(ROOT, 'flickr'))[0]
    elif name == 'ogbn-arxiv':
        data = _load_arxiv()
    else:
        raise KeyError(name)
    data = _clean(data)
    data.name = name
    return data


def _load_arxiv():
    """ogbn-arxiv without the ogb dataset class (it needs interactive confirmation)."""
    from torch_geometric.data import Data
    import pandas as pd
    base = os.path.join(ROOT, 'ogbn_arxiv')
    raw = os.path.join(base, 'arxiv', 'raw')
    if not os.path.exists(os.path.join(raw, 'edge.csv.gz')):
        os.makedirs(base, exist_ok=True)
        import urllib.request, zipfile
        zp = os.path.join(base, 'arxiv.zip')
        urllib.request.urlretrieve('http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip', zp)
        zipfile.ZipFile(zp).extractall(base)
    edge = pd.read_csv(os.path.join(raw, 'edge.csv.gz'), header=None).values.T
    x = pd.read_csv(os.path.join(raw, 'node-feat.csv.gz'), header=None).values
    y = pd.read_csv(os.path.join(raw, 'node-label.csv.gz'), header=None).values[:, 0]
    year = pd.read_csv(os.path.join(raw, 'node_year.csv.gz'), header=None).values[:, 0]
    data = Data(x=torch.tensor(x, dtype=torch.float), y=torch.tensor(y),
                edge_index=torch.tensor(edge, dtype=torch.long))
    data.node_year = torch.tensor(year)
    return data


# --------------------------------------------------------------------- homophily
def degrees(edge_index, n):
    return torch.bincount(edge_index[0], minlength=n)


def edge_homophily(edge_index, y):
    return float((y[edge_index[0]] == y[edge_index[1]]).float().mean())


def adjusted_homophily(edge_index, y):
    """Platonov et al. (2023), Eq. (adjusted homophily) with degree-weighted class shares."""
    n = y.numel(); K = int(y.max()) + 1
    d = degrees(edge_index, n).float()
    pk = torch.zeros(K).index_add_(0, y, d) / d.sum()
    he = edge_homophily(edge_index, y)
    s = float((pk ** 2).sum())
    return (he - s) / (1 - s)


def local_label_homophily(edge_index, y, n):
    """Per-node fraction of same-label neighbours (the label oracle; never used for sets)."""
    same = (y[edge_index[0]] == y[edge_index[1]]).float()
    num = torch.zeros(n).index_add_(0, edge_index[0], same)
    d = degrees(edge_index, n).float()
    return torch.where(d > 0, num / d.clamp(min=1), torch.zeros(n))


# ------------------------------------------------------------------------ splits
def model_split(n, seed, train_frac=0.2, val_frac=0.1, labelled=None):
    """Fixed per model seed: train / val for the base classifier, rest = conformal pool.

    20/10 follows the DAPS evaluation (stratification is unnecessary at these sizes).
    """
    g = np.random.default_rng(10_000 + seed)
    idx = np.arange(n) if labelled is None else np.asarray(labelled)
    idx = g.permutation(idx)
    ntr = int(round(train_frac * len(idx))); nva = int(round(val_frac * len(idx)))
    return idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]


def cal_test_split(pool, n_cal, seed):
    g = np.random.default_rng(20_000 + seed)
    p = g.permutation(pool)
    return p[:n_cal], p[n_cal:]


def train_adjusted_homophily(edge_index, y, train_idx, n):
    """Adjusted homophily of the subgraph induced by training nodes, using training labels only.
    This is the only label information HA-CP uses; V_train is disjoint from V_cal and V_test."""
    edge_index = np.asarray(edge_index); y = np.asarray(y)
    intr = np.zeros(n, bool); intr[train_idx] = True
    m = intr[edge_index[0]] & intr[edge_index[1]]
    if m.sum() < 2:
        return 0.0
    e = edge_index[:, m]
    K = int(y.max()) + 1
    d = np.bincount(e[0], minlength=n).astype(float)
    pk = np.bincount(y, weights=d, minlength=K) / d.sum()
    he = float((y[e[0]] == y[e[1]]).mean()); s = float((pk ** 2).sum())
    return (he - s) / (1 - s)
