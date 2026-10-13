# Reproduction guide

Run commands from the artifact root in the Python environment described in README.md. Run `python scripts/fetch_bundles.py` first. Allow roughly 4 GB for the checkout, unpacked data/predictions and downloaded archives, plus space for new runs. Large graph training can require substantially more RAM/VRAM. The release does not promise a hardware-independent runtime or bit-for-bit retraining.

## A. Rebuild the paper's numerical evidence

```bash
python scripts/verify.py --bundles
python src/test_conformal.py
python scripts/reproduce.py --figures
```

Outputs include:

| Paper item | Export | Measurement sources |
|---|---|---|
| Table 1: datasets | `tab_datasets_rows.csv` | `results/cp/*.csv.gz` metadata |
| Table 2: main set sizes | `tab_main_rows.csv` | conformal results and CF-GNN measurements |
| Table 3: selector regret | `tab_regret_rows.csv` | conformal, label-free gate and retrospective scalar-grid results |
| Table 4: coverage | `tab_cond_class_rows.csv`, `tab_cond_slice_rows.csv` | conformal coverage diagnostics |
| Figure 1 | `figure1.json`, `figure1-evidence.pdf` | Roman-empire GraphSAGE seed 0, split 0, selected node 7727 |
| Figure 2 | `figure2.pdf`, `figure2.png` and `fig_*.dat` | CSBM and label-free homophily results |
| Supplement shift table | `tab_shift_rows.csv` | `results/shift/` |
| Textual numerical claims | `numbers.csv`, `numbers.json` | all stored experiment families |
| Alpha sensitivity | `revision_sensitivity.csv` | primary and alpha=0.1 experiments |

The CSV tables keep the paper's column order and contain body rows rather than headers. Table 1: dataset, nodes, edges, classes, adjusted homophily, training-edge estimate. Table 2: dataset, homophily, APS, DAPS, DAPS-cal, CF-GNN, signed HeAD, HG, SNR, oracle. Table 3: selector, label access, mean, homophilous mean, heterophilous mean, worst, share within 2%. Coverage exports contain dataset followed by three methods for each of the two diagnostics. Formatting such as bold best entries is not part of the numeric comparison.

The original renderer writes temporary table markup inside a temporary directory. The wrapper exports CSV/JSON/data files and removes that directory. No manuscript or LaTeX source is distributed, and no TeX compiler is used. `numbers.json` preserves the original renderer's string formatting for exact comparison; `numbers.csv` is the readable version.

## B. Rerun from cached predictions

`run_stage.py` copies the code to a separate workspace and links the bundled data and cached logits. New result files stay in that workspace, leaving the release measurements untouched.

```bash
python scripts/run_stage.py --workdir runs/smoke run_cp.py --datasets Cora --models GCN --seeds 0 --splits 1 --alphas 0.05
python scripts/experiments.py --workdir runs/full --device cpu
```

The second command prints the complete experiment sequence. Add `--execute` to run it. This includes the full conformal grid, calibration-size sweep, CSBM simulation and crossover analysis, CF-GNN correction training, label-free diagnostics, class-wise calibration, distribution shifts and DAPS protocol reproduction. It is a long-running workload even with cached base predictions.

Individual examples:

```bash
python scripts/run_stage.py --workdir runs/full run_cp.py --datasets Cora Roman-empire
python scripts/run_stage.py --workdir runs/full csbm.py --workers 4
python scripts/run_stage.py --workdir runs/full analyze_csbm.py
python scripts/run_stage.py --workdir runs/full cfgnn.py --datasets Cora --device cpu
python scripts/run_stage.py --workdir runs/full labelfree.py
python scripts/run_stage.py --workdir runs/full run_labelfree_gate.py
python scripts/run_stage.py --workdir runs/full run_classwise.py
python scripts/run_stage.py --workdir runs/full run_shift.py --dataset ogbn-arxiv --scenario temporal
python scripts/run_stage.py --workdir runs/full run_shift.py --dataset Amazon-Computers --scenario degree
python scripts/run_stage.py --workdir runs/full run_shift.py --dataset Amazon-ratings --scenario degree
python scripts/run_stage.py --workdir runs/full e0_daps.py
```

`run_cp.py` has 5 seeds, 10 splits, alpha values 0.05 and 0.1 by default. `cfgnn.py --splits` accepts explicit split indices, not a count. Its default is indices 0 through 9. `csbm.py --quick` is a development subset, not the paper's 6,300-graph sweep.

## C. Retrain base classifiers

```bash
python scripts/experiments.py --workdir runs/retrained --device cuda --workers 4 --retrain
```

Review the printed commands, then add `--execute`. On a CPU-only machine use `--device cpu`. `--retrain` creates a new prediction directory. Existing checkpoints in that workspace are skipped by `train_base.py`, so use a fresh workspace for a clean rerun. Training configurations are in `src/models.py`, split rules in `src/data.py`, selector constants in `src/scores.py` and `src/run_cp.py`. No private server or external account is required.

The publication results combined laptop CPU and H100 GPU runs. The laptop environment is recorded in `docs/environment-macos.lock.txt`; the historical server record names PyTorch 2.4 and PyG 2.8 but does not preserve a complete server lockfile. We cannot claim bitwise reproducible GPU training across platforms.

## Official CF-GNN diagnostic

The main benchmark uses the supplied `src/cfgnn.py` port and its documented published hyperparameters. A separate official-code Cora-ML diagnostic is preserved in `results/e0/cfgnn_official_coraml.csv`. Its historical source checkout was not hashed. The following pinned upstream checkout provides a concrete rerun recipe, but is not claimed to reconstruct the unidentified historical commit:

```bash
git clone https://github.com/snap-stanford/conformalized-gnn.git runs/official-cfgnn
git -C runs/official-cfgnn checkout 3564a3d6d9bd4ff69e36c57672f8272c5fb0ff39
cd runs/official-cfgnn
python -m pip install -r requirements.txt
mkdir -p model
python train.py --dataset Cora_ML_CF --model GCN --optimal --device cpu --num_runs 3 --alpha 0.1
```

Follow upstream installation requirements if extra PyG extensions are needed. The `--wandb` flag is omitted. This optional upstream diagnostic is not executed by `scripts/experiments.py`, and was not rerun during packaging.

## Interpreting differences

Bootstrap means use fixed RNG seeds. Rendered reference numbers must match exactly in the validated environment. Floating-point plots/tables use tight tolerances. Retrained neural classifiers can vary with PyTorch/CUDA versions and reduction ordering, so compare coverage/set-size summaries rather than expecting identical model logits. The supplementary PDF specifies the statistical assumptions and metric definitions. The artifact does not turn the retrospective oracle into a deployable selector or replace the paper's stated limitations.
