# Third-party sources and provenance

Original research code is by Xuan-Bach Le. This package does not assign a blanket license to the underlying datasets, upstream baseline implementations or dependencies. Their respective source terms apply.

- CF-GNN: Huang, Jin, Candès and Leskovec, *Uncertainty Quantification over Graph with Conformalized Graph Neural Networks*, NeurIPS 2023. `src/cfgnn.py` is a port based on [the official implementation](https://github.com/snap-stanford/conformalized-gnn). The file documents changes to validation frequency and hyperparameter transfer. The upstream repository did not contain a top-level LICENSE file at the inspected commit `3564a3d6d9bd4ff69e36c57672f8272c5fb0ff39`. We do not invent a license for it.
- DAPS: Zargarbashi, Antonelli and Bojchevski, *Conformal Prediction Sets for Graph Neural Networks*, ICML 2023. Scalar diffusion and the E0 protocol reproduction are implemented in `src/scores.py` and `src/e0_daps.py`.
- HeAD-CP and HG-DAPS: the corresponding functions in `src/scores.py` implement the rules cited in the article (arXiv:2607.25273 and arXiv:2610.08564). Published constants are documented in source.
- APS, split conformal, weighted conformal and conditional-coverage diagnostics are implemented for this study. Their equations and references are given in the supplement and accompanying article.
- PyTorch, PyTorch Geometric, NumPy, SciPy, pandas, scikit-learn and Matplotlib are installed as dependencies, not vendored. Consult their distributions for license notices.
- Dataset provenance is listed separately in DATA.md. No decorative photograph or third-party figure icons are distributed in this artifact.

Historical design choices: node-wise and label-gate constants were explored on Cora and Roman-empire before the remaining graphs. SNR-lambda and its estimators were added after all graphs had been inspected. The recorded old design notes do not establish a blind design for that later rule. This package preserves the final implementations and stored results without presenting those choices as preregistered.
