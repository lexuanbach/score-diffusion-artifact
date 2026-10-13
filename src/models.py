"""Base classifiers with one fit() interface. Hyperparameters are fixed across datasets (PLAN Sect. 5)."""
import copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GCN2Conv

MODELS = ['GCN', 'GAT', 'SAGE', 'GCNII', 'MLP']
WIDE = {'Roman-empire', 'CoraFull'}          # hidden 256 instead of 64


class MLP(nn.Module):
    def __init__(s, f, h, K, p):
        super().__init__(); s.l1 = nn.Linear(f, h); s.l2 = nn.Linear(h, K); s.p = p
    def forward(s, x, ei):
        x = F.dropout(x, s.p, s.training); x = F.relu(s.l1(x))
        return s.l2(F.dropout(x, s.p, s.training))


class GCN(nn.Module):
    def __init__(s, f, h, K, p):
        super().__init__(); s.c1 = GCNConv(f, h, cached=True); s.c2 = GCNConv(h, K, cached=True); s.p = p
    def forward(s, x, ei):
        x = F.dropout(x, s.p, s.training); x = F.relu(s.c1(x, ei))
        return s.c2(F.dropout(x, s.p, s.training), ei)


class GAT(nn.Module):
    def __init__(s, f, h, K, p, heads=8):
        super().__init__()
        s.c1 = GATConv(f, h // heads, heads=heads, dropout=p)
        s.c2 = GATConv(h, K, heads=1, dropout=p); s.p = p
    def forward(s, x, ei):
        x = F.dropout(x, s.p, s.training); x = F.elu(s.c1(x, ei))
        return s.c2(F.dropout(x, s.p, s.training), ei)


class SAGE(nn.Module):
    """Mean aggregation; SAGEConv keeps separate ego (root) and neighbour weights."""
    def __init__(s, f, h, K, p):
        super().__init__(); s.c1 = SAGEConv(f, h, aggr='mean', root_weight=True)
        s.c2 = SAGEConv(h, K, aggr='mean', root_weight=True); s.p = p
    def forward(s, x, ei):
        x = F.dropout(x, s.p, s.training); x = F.relu(s.c1(x, ei))
        return s.c2(F.dropout(x, s.p, s.training), ei)


class GCNII(nn.Module):
    """Chen et al. (2020); 8 propagation layers, alpha=0.1, theta=0.5."""
    def __init__(s, f, h, K, p, L=8, alpha=0.1, theta=0.5):
        super().__init__(); s.inp = nn.Linear(f, h); s.out = nn.Linear(h, K); s.p = p
        s.convs = nn.ModuleList([GCN2Conv(h, alpha, theta, layer=i + 1, cached=True) for i in range(L)])
    def forward(s, x, ei):
        x = F.dropout(x, s.p, s.training); x = x0 = F.relu(s.inp(x))
        for c in s.convs:
            x = F.dropout(x, s.p, s.training); x = F.relu(c(x, x0, ei))
        return s.out(F.dropout(x, s.p, s.training))


def build(name, f, K, dataset):
    h = 256 if dataset in WIDE else 64
    return {'GCN': GCN, 'GAT': GAT, 'SAGE': SAGE, 'GCNII': GCNII, 'MLP': MLP}[name](f, h, K, 0.5)


def fit(model_name, data, train_idx, val_idx, seed, epochs=500, patience=100, device='cpu', lr=0.01, wd=5e-4):
    """Train with early stopping on validation accuracy; return softmax probabilities for all nodes."""
    torch.manual_seed(seed)
    K = data.num_classes
    model = build(model_name, data.x.shape[1], K, data.name).to(device)
    x, ei, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    tr = torch.as_tensor(train_idx, device=device); va = torch.as_tensor(val_idx, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad, t0 = -1, None, 0, time.time()
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = model(x, ei); loss = F.cross_entropy(out[tr], y[tr]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(x, ei)[va].argmax(1) == y[va]).float().mean().item()
        if acc > best:
            best, best_state, bad = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits = model(x, ei).cpu()
    return logits, {'val_acc': best, 'epochs': ep + 1, 'train_sec': time.time() - t0}
