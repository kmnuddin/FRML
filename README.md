# FRML — Federated Ridge Meta-Learning

Reference implementation and simulation harness for **FRML**, a closed-form, communication-efficient federated aggregation protocol for federated active learning (FAL). FRML was developed for IVD-level classification on lumbar-spine axial MRI but the code is dataset-agnostic — any tabular set of pre-computed embeddings with categorical labels can drive a simulation.

<p align="center">
  <img src="docs/FRML Overview.png" alt="FRML protocol overview" width="850"/>
</p>
<p align="center"><em><b>Figure 1.</b> Overview of the FRML protocol. <b>Left:</b> clients train a local committee on L<sub>i</sub>, build OOF meta-features, and transmit only the sufficient statistics S<sub>XX</sub>, S<sub>XY</sub> together with sample moments (n, μ, σ); the server solves a closed-form ridge model and broadcasts <b>W</b>. <b>Right:</b> each client uses committee uncertainty to actively select the most informative unlabeled samples for expert annotation under a fixed per-round budget B.</em></p>

The repository is organised as a CPU-only Python project. No GPUs, no distributed runtime — clients are simulated sequentially within a single process so that an entire federated run finishes in a few minutes on a laptop.

```
src/fel_ivd_federated/
    loop/        # Simulation runners (federated, centralized, baselines)
    meta/        # FRML ridge meta-learner and meta-feature construction
    models/      # Local model wrappers (RF, GBT, SVM, LR, MLP) + FL baselines
    selection/   # Seed selection (k-coreset, k-means)
    utils/       # Data IO, Dirichlet partitioning, calibration metrics
tools/
    run_fel.py           # Entry point for an FRML run
    run_centralized.py   # Pooled-data oracle baseline
    run_baseline.py      # FedAvg / FedProx / FedNova baselines
fel_sweeps.py            # Multi-run experiment driver
Analysis.ipynb           # Notebook for plotting and aggregating sweep logs
```

## Installation

```bash
git clone <repo>
cd FRML
pip install numpy pandas scikit-learn pyyaml
# optional, only needed if your embeddings are .pt/.pth files:
pip install torch
```

Python ≥ 3.10 is recommended.

## How the FRML protocol works

FRML replaces parameter averaging with **closed-form ridge regression over sufficient statistics of client-side predictive outputs**. The global model is *exactly* what would be obtained by pooling every client's out-of-fold meta-features in one place — there is no client drift and the per-round payload is a small fixed-size matrix pair regardless of the local model size.

A round proceeds in two stages.

**Stage 1 — Client-side acquisition loop.** Each client `c` holds a private pool of pre-computed FM embeddings indexed into a labeled set `L_c` and an unlabeled pool `U_c`. The labeled set is initialised by diversity-based seeding on the embedding space (k-coreset farthest-first or k-means). A local model `f_c` (RF by default, but any classifier in `models/model_wrapper.py` works) is fit on `L_c`. Acquisition scores are then computed on `U_c` using the local committee — entropy, margin, least-confident, or BALD — and the top-B samples are forwarded to an annotator. The new labels expand `L_c`, and the local model is refit. The server is *not* involved in acquisition; this keeps unlabeled data fully local and avoids extra communication rounds.

**Stage 2 — Server-side aggregation via sufficient statistics.** After local training, each client constructs **meta-features** `X_c` from out-of-fold (OOF) predictions on `L_c`. For tree ensembles these are per-tree probability vectors aggregated into mean probability, per-class variance, JS-disagreement, variation ratio, predictive entropy, and top-2 margin (`meta/summary_features.py`). For non-tree models the meta-features collapse to the model's `predict_proba` output plus entropy and top-2 margin. The client then z-scores `X_c`, appends a bias column, and sends only the second-order sufficient statistics

```
S_xx_c = X_c.T @ X_c
S_xy_c = X_c.T @ Y_c     (Y_c is the one-hot label matrix)
```

together with the local sample count `n_c`, per-feature mean `μ_c`, and variance `σ²_c`. Per-sample meta-features and labels are never transmitted.

The server aggregates additively, pools the moments into global `(μ, σ)`, and solves the ridge normal equations in closed form:

```
S_xx = Σ_c S_xx_c
S_xy = Σ_c S_xy_c
(S_xx + λ I) W = S_xy
```

`W` is the global meta-learner. Because both `S_xx` and `S_xy` are sums, the closed-form solution recovers exactly the model that pooling all clients' OOF features centrally would produce — aggregation is **lossless**.

**Inference.** For a new sample, the client computes meta-features locally, standardises with the broadcast `(μ, σ)`, and applies a softmax over `X @ W` to obtain `p_meta`. The server never sees raw inputs or per-sample meta-features.

The driver loop in `loop/sim_runner.py` runs steps 1–2 for a configurable number of rounds, logging per-round metrics (macro-F1, balanced accuracy, AUC, ECE, Brier) on a fixed stratified holdout into a JSONL file under `logging.out_dir`.

## How to organize new data for simulation

A simulation is fully described by **one CSV per client** plus **one YAML config**. Each CSV row corresponds to one sample (one axial slice in our setting), and the embeddings themselves live as separate files on disk — referenced by path from the CSV. This keeps the CSV small and lets you re-use the same embeddings across runs.

### 1. Per-client CSV

Each client gets its own CSV (e.g. `data/embed_clientA.csv`). The required columns are:

| Column | Type | What it holds |
|---|---|---|
| `emb` | path | Path to a `.npy`, `.npz`, `.pt`, or `.pth` file containing a 1-D embedding vector for this row. All rows must have the same dimensionality. |
| `axial_path` | path / string | Identifier for the source slice. Used as a grouping key — all augmentations of the same source share this value. |
| `ivd_level` | string | Class label (e.g. `L1_L2`, `L2_L3`, ...). Categorical, treated as strings. |
| `id` | int / string | Sample identifier (not used by the protocol itself, but kept for traceability). |

Column names are configurable through the `data:` block in the YAML (`label_col`, `id_col`, `image_col`, `emb_col`) — the names above are the defaults.

**Optional augmentation columns** (only needed if you want test-time or train-time augmentation):

| Column | What it holds |
|---|---|
| `is_aug` | `0` for the canonical/original row of a slice, `1` for augmented copies. |
| `aug_idx` | Integer ordering within the augmented copies (used to deterministically take the first K). |

If `is_aug` is absent, the first row encountered per `axial_path` is treated as the original and the rest as augmentations.

**Embedding files.** Store one embedding per file. Supported formats:

- `.npy` — `np.save` of a 1-D float array
- `.npz` — first key, or one of `emb` / `embedding` / `arr_0`
- `.pt` / `.pth` — a tensor, or a dict with key `emb` / `embedding`

A minimal CSV looks like this:

```csv
id,axial_path,emb,ivd_level,is_aug,aug_idx
P0001_s07,/data/clientA/P0001/slice07.png,/data/clientA/emb/P0001_s07.npy,L4_L5,0,0
P0001_s07,/data/clientA/P0001/slice07.png,/data/clientA/emb/P0001_s07_a1.npy,L4_L5,1,1
P0002_s11,/data/clientA/P0002/slice11.png,/data/clientA/emb/P0002_s11.npy,L3_L4,0,0
...
```

### 2. YAML config

Point the runner at your client CSVs and pick simulation hyperparameters. The keys understood by `run_fel.py` are listed below — anything omitted falls back to the default shown.

```yaml
# ---- data ----
data:
  rsna_csv:     data/embed_clientA.csv     # one csv per client; key is the client id
  mendeley_csv: data/embed_clientB.csv     # add more keys for more clients
  label_col: ivd_level
  id_col:    id
  image_col: axial_path
  emb_col:   emb

# ---- evaluation ----
holdout:
  test_frac: 0.30        # stratified per-client holdout, fixed across rounds

# ---- AL loop ----
seed_k: 200              # initial labeled budget per client
rounds: 60
seeding:
  method: k_coreset      # or "kmeans"
al:
  method: margin         # margin | entropy | least_confident | bald | qbc
  batch_B: 200
  per_class_min: 1

# ---- local model ----
rf:
  n_estimators: 100
  max_depth: null
  class_weight: balanced_subsample

# ---- FRML meta-learner ----
meta:
  lambda: 0.01
  oof_folds: 5
  refresh_every: 1
  feature_groups: mean_p,var_p,disagreement,uncertainty   # any comma-separated subset

# ---- logging ----
random_state: 42
logging:
  out_dir: runs/my_first_run
```

The client names in `data:` are arbitrary — any key ending in `_csv` becomes a federated client. Two are shown above (RSNA, Mendeley) but you can list ten if you want.

### 3. Run it

```bash
PYTHONPATH=src python tools/run_fel.py --config my_config.yaml
```

Per-round metrics stream to `runs/my_first_run/sim_log.jsonl`. The same config can be passed to `run_centralized.py` to produce the pooled-data oracle, or to `run_baseline.py` (with an additional `baseline_method:` key) to run FedAvg / FedProx / FedNova on the same data and holdout.

### 4. Optional extensions

The runner understands a few additional top-level keys for advanced experiments:

- `partition:` — split each real client into `n_clients` simulated sub-clients via Dirichlet (`alpha` controls non-IID severity) or uniform. Useful for scalability and heterogeneity studies without collecting more data.
- `heterogeneous_rf:` — a list of RF hyperparameter dicts assigned round-robin across clients (e.g. weak/strong forests in the same federation).
- `client_model_types:` — a list of model strings (`rf`, `gbt`, `logistic`, `svm`, `mlp`) assigned round-robin. Lets you federate genuinely different architectures, which is the regime where parameter averaging breaks down but FRML still works.
- `emb_noise_stds:` — per-client Gaussian noise injected on embeddings to simulate scanner or domain shift.
- `augment.train_n_per_sample: K` — at training time, include up to K augmented rows per labeled original (requires the `is_aug`/`aug_idx` columns).

## Running experiment sweeps

`fel_sweeps.py` builds derived YAMLs from a base config and runs the entire suite of experiments reported in the paper.

```bash
python fel_sweeps.py --base-config configs/example.yaml --out-root runs/sweeps --study 0
```

The `--study` flag selects which experiment to run. The Studies 0–10 correspond to: evaluation framework + centralized oracle (0), AL strategy comparison (1), seed/budget sweeps (2/3), label noise (4), Dirichlet scalability (5), meta-feature ablation (6), heterogeneous RFs (7), embedding-noise robustness (8), mixed-architecture federation (9), and gradient-based FL baselines (10). Each run writes its own derived YAML and JSONL log under `runs/sweeps/<study>/<tag>/`.

`Analysis.ipynb` walks every run directory under a sweep root, parses the JSONL logs, and reproduces the figures in the paper.

## Reproducing the paper

The default configuration in the snippet above matches the protocol described in the paper: 60 rounds, batch size B=200, seed pool of 200 selected by k-coreset, λ=0.01, 30% stratified holdout. Running studies 0–10 with that base config reproduces the federation-gap, communication-cost, ablation, scalability, and heterogeneity results reported in Tables 1–4 and Figures 3–7 of the manuscript.

## Citation

Comming shortly.
