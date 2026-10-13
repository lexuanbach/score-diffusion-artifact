"""Download every dataset used in the study into data/ (run once)."""
import sys, traceback
sys.path.insert(0, 'src')
from data import load_dataset, DATASETS

for name in DATASETS:
    try:
        d = load_dataset(name)
        print(f'{name:16s} n={d.num_nodes:7d} K={d.num_classes:3d} f={d.x.shape[1]:5d} E_und={d.edge_index.shape[1]//2}', flush=True)
    except Exception:
        print('FAILED', name, flush=True); traceback.print_exc()
