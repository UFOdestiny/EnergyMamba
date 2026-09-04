# EnergyMamba

**An Uncertainty-Aware Graph-Enhanced Selective State Space Model for Energy Consumption Prediction**

[![Venue](https://img.shields.io/badge/ACM%20SIGKDD-2026-blue)](https://dl.acm.org/doi/10.1145/3770855.3818841)
[![arXiv](https://img.shields.io/badge/arXiv-2606.00506-b31b1b)](https://arxiv.org/abs/2606.00506)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3770855.3818841-orange)](https://doi.org/10.1145/3770855.3818841)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Energy consumption prediction is usually treated as pure time-series forecasting, which ignores the spatial dependencies between regions, and it rarely comes with usable uncertainty estimates — exactly what is needed when demand goes abnormal (heat waves, storms, outages). EnergyMamba addresses both:

1. **GE-Mamba** — a Graph-Enhanced Selective State Space Model that injects spatial context learned from the grid topology into the *selectivity* of a Mamba scan, so space and time are modelled jointly rather than in sequence.
2. **AS-CQR** — Adaptive Sequential Conformalized Quantile Regression, which normalizes the nonconformity score by the local interval width and updates the correction online, keeping intervals valid under distribution shift.

On four large-scale real-world datasets from Florida, New York and California, EnergyMamba improves prediction accuracy by ~5% and uncertainty quantification by ~6% over 15 state-of-the-art baselines.

## Method

### GE-Mamba (`src/flow/energymamba/em_model.py`)

Internal tensor convention is `(B, N, T, D)` — batch, node, time, hidden — so the graph convolution mixes across `N` at every timestep while the selective SSM scans along `T` per node.

| Piece | Detail |
| --- | --- |
| **GCN spatial context** | `Z_t = GELU( Ã H_t W )` with the pre-normalized adjacency `Ã = D̃^{-1/2}(A+I)D̃^{-1/2}`, applied independently at each timestep. |
| **Spatial-conditioned selective scan** (`GEMambaSSM`) | A Mamba selective state-space scan whose input-dependent `(Δ, B, C)` are conditioned on the spatial context `z`, so neighbourhood information decides what the state keeps and forgets. Built on `mamba_ssm`'s fused `selective_scan_fn`. |
| **GE-Mamba block** | Residual block: RMSNorm → GCN → bidirectional spatial-conditioned scan → dropout → residual. |
| **U-Net backbone** | `unet_depth` encoder stages with temporal downsampling, a bottleneck, and matching decoder stages with skip connections — multi-scale temporal patterns at `num_layers` blocks per stage. |
| **Output head** | A single `output_proj` sized by `output_dim`, which makes the model a drop-in quantile regressor (see below). |

`forward(x)` takes `(B, seq_len, N, F)` and returns `(B, horizon, N, output_dim)`.

### AS-CQR (`src/flow/energymamba/ACQR_engine.py`)

Under `--cqr`, the runner records the true feature count `F` and widens `output_dim` to `3F`, so the model's own projection emits three ordered channels per feature — `q_mid = c0`, `q_lo = c0 - softplus(c1)`, `q_hi = c0 + softplus(c2)` (no quantile crossing) — trained with the pinball loss. `ACQR_Engine` subclasses the shared `CQR_Engine` and replaces only calibration and interval construction, with three changes over vanilla CQR (Romano et al., 2019):

1. **Locally adaptive nonconformity** — the score is normalized by the predicted width, making it scale-invariant across nodes, horizons and regimes:
   `ε_t = max(q_lo − y, y − q_hi) / (q_hi − q_lo + δ)`
2. **Sliding-window correction** — instead of one fixed `Q` from the validation split, `Q_t` is the `(1 − α̃_t)` empirical quantile of the most recent `m` scores, and the interval is widened *multiplicatively* by the raw width:
   `Ĉ(X_t) = [q_lo − Q_t·w_t , q_hi + Q_t·w_t]`, `w_t = q_hi − q_lo`
3. **Online feedback** — the effective miscoverage is updated from the realised coverage after every step, so long-run coverage converges to `1 − α`:
   `α̃_{t+1} = α̃_t + γ ( α − 1{ Y_t ∉ Ĉ(X_t) } )`

The validation split seeds the sliding window, so the very first test steps already carry a meaningful correction; from there the window slides over the test stream in temporal order. Calibration and metrics run in the original (inverse-transformed) data space, and `Q` is persisted in the checkpoint.

`--cqr no` (the default) runs the plain point model on the shared `BaseEngine`; `--cqr horizon` / `--cqr global` switches in `ACQR_Engine`. Metrics under `--cqr`: `Quantile`, `MAE`, `MAPE`, `RMSE`, `MPIW`, `IS`, `COV`, `F1`, `TZR`, `KL`, `CRPS`.

## Repository layout

```
EnergyMamba/
  src/flow/energymamba/
    em_model.py         GE-Mamba: GCN + spatial-conditioned selective SSM + U-Net
    ACQR_engine.py      Adaptive Sequential CQR calibration engine
    main.py             Entry point: model args, adjacency setup, run_experiment()
  base/                 Shared framework
    runner.py           run_experiment(): the single experiment driver
    model.py            BaseModel contract
    engine.py           Training / validation / test loop, checkpointing, export
    CQR_engine.py       Conformalized Quantile Regression engine (ACQR's parent)
    metrics.py          Point, distributional and interval metrics
    efficiency.py       Hardware info, memory, inference time, FLOPs
  utils/
    args.py             Common CLI arguments, path config, set_seed
    dataloader.py       Dataset / DataLoader, dataset registry lookup
    generate.py         Raw array -> his.npz / info.json / split indices
    registry.yaml       Dataset name -> data & adjacency paths
    graph_algo.py       Adjacency normalizations
    get_adj_mat.py      Build an adjacency matrix from geographic shapefiles
    log.py              Logger
    res.py              Result collection / comparison CLI
  jobs/train.sh         Slurm submission script
```

## Installation

```bash
conda create -n st python=3.10 -y
conda activate st

# PyTorch (CUDA 12.8 build)
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

`mamba_ssm` needs a CUDA toolchain matching your PyTorch build; install it after PyTorch.

## Data

Point the framework at your data root (defaults to `<repo>/datasets`):

```bash
export POPST_DATA=/path/to/datasets      # where datasets live
export POPST_RESULT=/path/to/result      # where logs & checkpoints go
```

Each dataset folder has this layout, and every entry is produced by `utils/generate.py`:

```
<POPST_DATA>/<dataset>/
  <adj_name>.npy        Adjacency matrix (N x N)
  <years>/
    his.npz             Normalized data + scaler parameters
    info.json           Shape, scaler, split sizes, seq_length_x / seq_length_y
    meta.json           Scaler parameters and raw data shape
    idx_{train,val,test,all}.npy   Split sample indices
```

Generate it from a raw array of shape `(T, N, F)` and register it in `utils/registry.yaml`:

```bash
python utils/generate.py --data_path /path/to/raw.npy --dataset my_energy --years 2018 --fmt NDT
```

```yaml
my_energy:
  data: my_energy
  adj: my_energy/adj.npy
```

`N` is read from `info.json` at runtime; `seq_len` / `horizon` / `input_dim` / `output_dim` are auto-filled from the same file unless given on the command line.

## Usage

```bash
# Point prediction
python src/flow/energymamba/main.py --dataset chicago_15min --years 2018

# Uncertainty-aware: AS-CQR with a per-horizon correction, 90% target coverage
python src/flow/energymamba/main.py --dataset chicago_15min --years 2018 \
    --cqr horizon --quantile_alpha 0.1

# A single shared correction instead
python src/flow/energymamba/main.py --dataset chicago_15min --cqr global

# Test from a checkpoint (the calibrated Q is restored with it)
python src/flow/energymamba/main.py --dataset chicago_15min --cqr horizon \
    --mode test --model_path /path/to/EnergyMamba_CQR_<timestamp>.pt

# Export prediction archives
python src/flow/energymamba/main.py --dataset chicago_15min --mode test --export
```

Slurm:

```bash
sbatch jobs/train.sh
EXTRA="--cqr horizon" DATASETS="chicago_15min" sbatch jobs/train.sh
```

Compare runs:

```bash
python utils/res.py --path result/MyExperiment
python utils/res.py --log result/MyExperiment/EnergyMamba_CQR/chicago_15min/<timestamp>.log
```

Results land in `result/<proj>/EnergyMamba/<dataset>/<timestamp>.log` (`EnergyMamba_CQR` under `--cqr`) next to the matching `.pt` checkpoint.

## Arguments

### Model (GE-Mamba)

| Argument | Default | Description |
| --- | --- | --- |
| `--d_model` | `64` | Hidden width |
| `--num_layers` | `2` | GE-Mamba blocks per U-Net stage (`K`) |
| `--unet_depth` | `2` | Encoder / decoder stages (`S`) |
| `--d_state` | `16` | SSM state dimension |
| `--expand` | `2` | Mamba inner expansion factor |
| `--d_conv` | `4` | Depthwise causal-conv width inside the SSM |
| `--dropout` | `0.1` | Dropout |

### AS-CQR (used only with `--cqr`)

| Argument | Default | Description |
| --- | --- | --- |
| `--cqr` | `no` | `no` (point model), `horizon` (one correction per forecast step), `global` (single shared correction) |
| `--quantile_alpha` | `0.1` | Target miscoverage `α`; intervals target `1 − α` coverage |
| `--acqr_window` | `100` | `m` — recent nonconformity scores kept in the sliding window |
| `--acqr_gamma` | `0.005` | `γ` — online miscoverage feedback rate |
| `--acqr_delta` | `1e-6` | `δ` — stability floor in the width-normalized denominator |

### Training

| Argument | Default | Description |
| --- | --- | --- |
| `--bs` | `64` | Batch size |
| `--max_epochs` | `2000` | Maximum epochs |
| `--patience` | `30` | Early-stopping patience on validation loss |
| `--lrate` | `1e-3` | Learning rate (AdamW) |
| `--wdecay` | `5e-4` | Weight decay |
| `--clip_grad_norm` | `1.0` | Gradient-norm clipping |
| `--step_size` | `200` | StepLR decay interval |
| `--gamma` | `0.95` | StepLR decay factor |
| `--seed` | `2025` | Random seed |

### Data & system

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `chicago_15min` | Dataset name (must exist in `registry.yaml`) |
| `--years` | `2018` | Data sub-folder |
| `--seq_len` / `--horizon` | auto | Input length / forecast steps, auto-filled from `info.json` |
| `--input_dim` / `--output_dim` | auto | Feature counts, auto-filled from `info.json` |
| `--no_normalize` | -- | Disable MinMax normalization (on by default) |
| `--device` | `cuda` | Device |
| `--mode` | `train` | `train` or `test` |
| `--model_path` | -- | Checkpoint to load in test mode |
| `--export` | off | Save prediction archives with the final evaluation |
| `--proj` | -- | Sub-folder name for grouping results |

## Implementation details

Experiments were run on a Linux server with an NVIDIA A100 GPU. This release is pinned to PyTorch 2.8.0 / CUDA 12.8 and `mamba_ssm` 2.2.6.

## Citation

```bibtex
@inproceedings{yu2026energymamba,
  title     = {EnergyMamba: An Uncertainty-Aware Graph-Enhanced Selective State Space Model for Energy Consumption Prediction},
  author    = {Yu, Dahai and Xu, Rongchao and Jiang, Lin and Wang, Guang},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages     = {12727--12738},
  year      = {2026},
  doi       = {10.1145/3770855.3818841}
}
```

## Related work

- [POPST](https://github.com/UFOdestiny/POPST) — the unified spatiotemporal benchmarking framework this release is extracted from (~30 flow models, ~17 OD models, shared conformal-prediction engines)
- [TrustEnergy](https://github.com/UFOdestiny/TrustEnergy) (AAAI 2026) — memory-augmented spatiotemporal GNN with sequential CQR
- [HealthMamba](https://github.com/UFOdestiny/HealthMamba) (IJCAI 2026) — graph state space model with three-mechanism uncertainty quantification
- [UQGNN](https://github.com/UFOdestiny/UQGNN) (SIGSPATIAL 2025) — multivariate Gaussian spatiotemporal prediction

## Acknowledgements

The selective scan builds on [Mamba](https://github.com/state-spaces/mamba); the U-Net arrangement follows [U-Mamba](https://github.com/bowang-lab/U-Mamba). Baselines used in the paper are implemented in [POPST](https://github.com/UFOdestiny/POPST) on the same runner and metric pipeline.

## License

Released under the [MIT License](LICENSE).
