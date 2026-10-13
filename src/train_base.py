"""Train base classifiers and cache logits: results/preds/<dataset>/<model>_s<seed>.pt

Usage: python src/train_base.py --datasets Cora Roman-empire --models GCN MLP --seeds 0 1 2 3 4
"""
import argparse, os, sys, json, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_dataset, model_split
from models import fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--models', nargs='+', default=['GCN', 'GAT', 'SAGE', 'GCNII', 'MLP'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--split', default='frac', choices=['frac', 'temporal'])
    ap.add_argument('--threads', type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for ds in a.datasets:
        data = load_dataset(ds)
        for m in a.models:
            for s in a.seeds:
                tag = '' if a.split == 'frac' else '_temporal'
                out = os.path.join(ROOT, 'results', 'preds', ds, f'{m}_s{s}{tag}.pt')
                if os.path.exists(out):
                    continue
                os.makedirs(os.path.dirname(out), exist_ok=True)
                if a.split == 'temporal':   # ogbn-arxiv: train/val on papers <= 2017 only
                    old = torch.nonzero(data.node_year <= 2017).view(-1).numpy()
                    tr, va, _ = model_split(data.num_nodes, s, labelled=old, train_frac=0.6, val_frac=0.2)
                else:
                    tr, va, _ = model_split(data.num_nodes, s)
                logits, info = fit(m, data, tr, va, seed=s, device=a.device)
                test_mask = torch.ones(data.num_nodes, dtype=torch.bool); test_mask[tr] = False; test_mask[va] = False
                acc = (logits.argmax(1)[test_mask] == data.y[test_mask]).float().mean().item()
                info.update(dataset=ds, model=m, seed=s, pool_acc=acc, split=a.split)
                torch.save({'logits': logits, 'train_idx': torch.as_tensor(tr), 'val_idx': torch.as_tensor(va), 'info': info}, out)
                print(json.dumps(info), flush=True)

if __name__ == '__main__':
    main()
