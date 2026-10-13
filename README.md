# When Does Score Diffusion Help Graph Conformal Prediction?

Reproducibility artifact for Xuan-Bach Le's paper. This repository contains experiment code, per-run measurements, reference numerical outputs, and the supplementary PDF. Data and cached classifier predictions are attached to release `v1.0.0` and downloaded by the script below. The manuscript PDF and LaTeX sources are intentionally not included.

## Quick start

Use Python 3.13. The pinned environment below was validated on macOS ARM64. Linux/CUDA training uses the same scripts, but installing the appropriate PyTorch CUDA build may require a platform-specific environment.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/fetch_bundles.py
python scripts/verify.py --bundles
python src/test_conformal.py
python scripts/reproduce.py --figures
```

The last command rebuilds the reported numbers, tables, and figure evidence in `outputs/`. It checks all 530 stored numerical values, table CSVs and figure-data files against `expected/`, and verifies the selected Figure 1 node against the original run-level measurements. No LaTeX installation is needed. For summaries only, use `python scripts/fetch_bundles.py --only data` and omit `--figures` (Figure 1 needs predictions).

**Three reproduction levels:**

1. **From stored measurements:** `scripts/reproduce.py --figures` regenerates numerical tables, Figure 2, and Figure 1 evidence. Figure 1's evidence charts use the same values, but do not recreate the manuscript's decorative layout or photo.
2. **From stored predictions:** rerun conformal calibration and selectors using all 330 saved logit/split files. This avoids retraining and preserves the original base predictions.
3. **From raw data:** rerun classifier training and experiments with the supplied seeds and settings. GPU training is recommended. Floating-point and training differences across hardware/software can change results. Exact numerical matching is asserted for level 1, and was checked for one Cora/GCN conformal run at level 2. We have not rerun the entire training suite while packaging this release.

A small level-2 check, without overwriting reference results:

```bash
python scripts/run_stage.py --workdir runs/smoke run_cp.py --datasets Cora --models GCN --seeds 0 --splits 1 --alphas 0.05
```

See [REPRODUCE.md](REPRODUCE.md) for all commands and the experiment-to-output map, [DATA.md](DATA.md) for data provenance, [docs/VALIDATION.md](docs/VALIDATION.md) for checks actually run, and [supplement.pdf](supplement.pdf) for proofs, pseudocode and metric definitions.

## Contents

| Path | Contents |
|---|---|
| `src/` | Dataset loading, model training, conformal selectors, simulations, evaluation and numerical renderer |
| `scripts/` | Bundle download/checksums, isolated experiment execution and portable exports |
| `configs/experiments.json` | Dataset/model lists, seeds, split counts and main settings |
| `results/` | Original per-run CSV/JSON measurements (predictions are release assets) |
| `expected/` | Snapshot of reported values and plot/table data used for regression checks |
| `docs/` | File hashes, environment snapshot, provenance and validation notes |
| `supplement.pdf` | Supplement only, with main-paper references retained as printed section/equation numbers |

Reference results are read-only inputs by convention. Use `scripts/run_stage.py` for new experiments. Public release data contain research benchmark graphs and model outputs, not author credentials, server settings, review files or repository history.

## Attribution

If you use this artifact, cite *When Does Score Diffusion Help Graph Conformal Prediction?* by Xuan-Bach Le. See [THIRD_PARTY.md](THIRD_PARTY.md) for dataset and baseline sources. Upstream data and dependencies retain their own terms. No blanket relicensing of third-party materials is implied.
