# Data and cached predictions

Release `v1.0.0` contains `data.tar.gz` (about 383 MiB) and `predictions.tar.gz` (about 328 MiB). `bundles.json` pins their byte counts and SHA-256 digests. `scripts/fetch_bundles.py` verifies the archives before extraction. `docs/data-files.json` and `docs/predictions-files.json` inventory individual files. The data archive contains 89 raw/processed files; the prediction archive contains 330 PyTorch logit/split files (13 graphs × 5 classifiers × 5 seeds, plus 5 temporal arxiv GCN files).

## Sources

| Graphs | Source and loader |
|---|---|
| Cora, CiteSeer, PubMed | [Planetoid](https://github.com/kimiyoung/planetoid), PyG `Planetoid` |
| Amazon-Photo, Amazon-Computers, CoraFull | [GNN benchmark data](https://github.com/shchur/gnn-benchmark), PyG `Amazon` and `CoraFull` |
| Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions | [Heterophilous graphs](https://github.com/yandex-research/heterophilous-graphs), PyG `HeterophilousGraphDataset` |
| Filtered Chameleon and Squirrel | Same heterophilous-graphs repository, filtered NPZ files |
| ogbn-arxiv | [Open Graph Benchmark node datasets](https://ogb.stanford.edu/docs/nodeprop/), direct `arxiv.zip` loader |
| Cora-ML and citation-full CiteSeer for E0 | GNN benchmark source through PyG `CitationFull` |

These public benchmark datasets are third-party materials. Retain upstream attribution and consult each source's applicable data terms before other redistribution or uses. The release preserves the downloaded and processed snapshots used for this study, rather than claiming ownership of the source graphs.

`src/data.py` removes self loops, converts graphs to undirected form and coalesces duplicate edges. Main-experiment train/validation splits use 20%/10% of nodes. Dataset-provided masks are retained in some source objects but not used for the study's random splits. Binary graphs enter coverage diagnostics, not multi-class efficiency comparisons. `src/e0_daps.py` follows its own published-protocol splits. Arxiv temporal training uses papers dated at most 2017.

If the release assets are unavailable, `python src/download.py` downloads the 14 main/shift datasets from upstream. E0 additionally downloads its citation-full data on first use. Upstream files may change, so downloaded replacements are not assumed to match the release hashes. The snapshot bundle is the supported route for matching stored results.

Cached `.pt` files store logits, train/validation indices and training metadata, not full model parameter checkpoints. They support rerunning conformal prediction. To obtain new model predictions, use the retraining commands. Only load the trusted, checksum-verified release files because PyTorch/PyG processed caches may use Python serialization.
