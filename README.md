# Modeling Data Preparation Pipeline

This repository contains the scripts used to turn `dump.yaml` into a reproducible machine learning dataset for predicting 6×6 beam covariance matrices.

The model takes 38 accelerator input parameters and directly outputs the 6×6 covariance matrix of the beam phase space `(x, px, y, py, z, pz)`. Internally the network predicts 21 lower-triangular Cholesky factors, then reconstructs `C = L @ Lᵀ`. Training is performed in normalized covariance space (per-element mean/std from the training set), with either MSE or L1 loss.

## Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate modeling
```

## End-to-End Workflow

The pipeline has nine stages:

1. Convert `dump.yaml` to CSV
2. Remove rows without `particles_241`
3. Create Cholesky covariance targets from particle files
4. Drop non-feature and non-target columns to build the final dataset
5. Split the dataset into train, validation, and test CSVs
6. (Optional) Create an interpolation holdout from the training set
7. Train the covariance MLP
8. Analyze model performance
9. (Optional) Plot input parameter distributions

All commands below assume you are running them from the repository root.

## 1. Convert YAML to CSV

Convert `dump.yaml` into a flat CSV table:

```bash
python create_csv_from_yaml.py dump.yaml dump.csv
```

Expected output:

- Input: `dump.yaml`
- Output: `dump.csv`

Optional file embedding is supported, but it is not part of the current covariance-target workflow.

## 2. Clean the CSV

Keep only rows where `particles_241` is present:

```bash
python cleanup.py dump.csv dump-particles_241-not-null.csv --column particles_241
```

Expected output:

- Input: `dump.csv`
- Output: `dump-particles_241-not-null.csv`
- Filter: rows with non-null `particles_241`

## 3. Create Cholesky Covariance Targets

Read each `particles_241` OpenPMD particle file, compute the 6D covariance matrix over `(x, px, y, py, z, pz)`, apply a Cholesky decomposition, and flatten the lower triangle into 21 target columns:

```bash
python create_cov_targets_from_particles.py \
  dump-particles_241-not-null.csv \
  cov-targets.csv \
  --progress-every 200 \
  --drop-failed
```

Expected output:

- Input: `dump-particles_241-not-null.csv`
- Output: `cov-targets.csv`
- Added columns: `cov_chol_0` through `cov_chol_20`
- Status column: `cov_target_status`

Notes:

- `--drop-failed` removes rows where covariance extraction failed.
- The script keeps all 21 lower-triangular Cholesky elements for a fixed-width target vector.

## 4. Create the Final Dataset

Remove metadata, file-path, scalarized, emittance, sigma, and other non-training columns:

```bash
python create_dataset.py cov-targets.csv dataset.csv
```

Expected output:

- Input: `cov-targets.csv`
- Output: `dataset.csv`

Current dataset structure:

- 38 feature columns
- 21 target columns: `cov_chol_0` through `cov_chol_20`
- Total: 59 columns

### Feature and Target Columns

Feature columns:

- `CQ10121:b1_gradient`
- `GUNF:rf_field_scale`
- `GUNF:theta0_deg`
- `L0AF_phase:theta0_deg`
- `L0AF_scale:rf_field_scale`
- `L0BF_phase:theta0_deg`
- `L0BF_scale:rf_field_scale`
- `QA10361`
- `QA10371`
- `QE10425`
- `QE10441`
- `QE10511`
- `QE10525`
- `SOL10111:solenoid_field_scale`
- `SQ10122:b1_gradient`
- `bmad_L0BF_phase:theta0_deg`
- `bmad_L0BF_scale:rf_field_scale`
- `bmad_QA10361`
- `bmad_QA10371`
- `bmad_QE10425`
- `bmad_QE10441`
- `bmad_QE10511`
- `bmad_QE10525`
- `bmad_bmad_phase_offset`
- `distgen:VCC`
- `distgen:t_dist:sigma_t:value`
- `distgen:total_charge:value`
- `impact_CQ10121:b1_gradient`
- `impact_GUNF:rf_field_scale`
- `impact_GUNF:theta0_deg`
- `impact_L0AF_phase:theta0_deg`
- `impact_L0AF_scale:rf_field_scale`
- `impact_SOL10111:solenoid_field_scale`
- `impact_SQ10122:b1_gradient`
- `impact_VCC_Cal`
- `impact_distgen:t_dist:sigma_t:value`
- `impact_distgen:total_charge:value`
- `n_particles_241`

Target columns:

- `cov_chol_0`
- `cov_chol_1`
- `cov_chol_2`
- `cov_chol_3`
- `cov_chol_4`
- `cov_chol_5`
- `cov_chol_6`
- `cov_chol_7`
- `cov_chol_8`
- `cov_chol_9`
- `cov_chol_10`
- `cov_chol_11`
- `cov_chol_12`
- `cov_chol_13`
- `cov_chol_14`
- `cov_chol_15`
- `cov_chol_16`
- `cov_chol_17`
- `cov_chol_18`
- `cov_chol_19`
- `cov_chol_20`

Useful option:

```bash
python create_dataset.py cov-targets.csv dataset.csv --drop-null-rows
```

## 5. Split into Train, Validation, and Test Sets

Split the prepared dataset into reproducible train/validation/test files:

```bash
python split_dataset.py dataset.csv
```

Default behavior:

- Train fraction: `0.70`
- Validation fraction: `0.15`
- Test fraction: `0.15`
- Shuffle: enabled
- Seed: `42`

Generated files:

- `dataset-train.csv`
- `dataset-val.csv`
- `dataset-test.csv`

Example with custom ratios:

```bash
python split_dataset.py \
  dataset.csv \
  --train-fraction 0.8 \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --seed 42
```

## 6. Train the Model

The model outputs the full 6×6 covariance matrix directly. Targets are the 21 lower-triangular Cholesky factors (`cov_chol_0..20`); the network reconstructs `C = L @ Lᵀ` internally. Loss is computed in per-element normalized covariance space.

### Quick start (recommended settings)

```bash
python train.py \
  --loss-space cov \
  --cov-loss l1 \
  --epochs 200 \
  --patience 40 \
  --batch-size 256 \
  --lr 1e-3 \
  --output-dir model-output-cov-l1
```

### With staged fine-tuning (best results)

After base training the script runs additional fine-tuning stages with progressively smaller batch sizes to improve performance on difficult examples:

```bash
python train.py \
  --loss-space cov \
  --cov-loss l1 \
  --epochs 200 \
  --patience 40 \
  --batch-size 256 \
  --lr 1e-3 \
  --finetune-batch-sizes 32 8 2 \
  --finetune-epochs-per-stage 300 \
  --finetune-lr 1e-4 \
  --finetune-lr-decay 0.5 \
  --finetune-plateau-patience 5 \
  --finetune-min-lr 1e-6 \
  --output-dir model-output-cov-l1-ft
```

### Key training options

| Flag | Default | Description |
|---|---|---|
| `--cov-loss` | `mse` | Covariance-space objective: `mse` or `l1`. L1 gives significantly better MAPE (~11% vs ~26%) |
| `--epochs` | `200` | Max training epochs |
| `--patience` | `20` | Early-stopping patience in epochs (0 = disabled) |
| `--batch-size` | `256` | Mini-batch size |
| `--lr` | `1e-3` | Initial learning rate (halved via `ReduceLROnPlateau` after 10 stagnant epochs) |
| `--finetune-batch-sizes` | `None` | Space-separated batch sizes for staged fine-tuning, e.g. `32 8 2` |
| `--finetune-epochs-per-stage` | `0` | Epochs per fine-tuning stage |
| `--finetune-lr` | `1e-4` | Initial LR for fine-tuning |
| `--finetune-lr-decay` | `0.5` | LR multiplier applied after each stage |
| `--finetune-min-lr` | `1e-6` | Minimum LR floor during fine-tuning |
| `--output-dir` | `model-output` | Directory for checkpoint and transformer files |

### Generated files in `--output-dir`

- `model.pt` — best-val-loss model weights
- `input_transformers.pt` — input feature scaler (`x_mean`, `x_std`) and feature column names
- `output_transformers.pt` — Cholesky target scaler (`y_mean`, `y_std`) and target column names
- `covariance_transformers.pt` — per-element covariance normalizers (`cov_mean`, `cov_std`, `cov_labels`)
- `training_history.csv` — per-epoch train/val loss

### Model architecture

38 inputs → 21 Cholesky outputs → 6×6 covariance output:

```
Linear(38 → 100), ELU
Linear(100 → 200), ELU, Dropout(0.05)
Linear(200 → 200), ELU, Dropout(0.05)
Linear(200 → 300), ELU, Dropout(0.05)
Linear(300 → 300), ELU, Dropout(0.05)
Linear(300 → 200), ELU, Dropout(0.05)
Linear(200 → 100), ELU, Dropout(0.05)
Linear(100 → 100), ELU
Linear(100 → 100), ELU
Linear(100 → 21)  → build L (lower-triangular) → C = L @ Lᵀ
```

### Normalization

Loss is computed in normalized covariance space. Before training, per-element mean (μᵢⱼ) and standard deviation (σᵢⱼ) are computed from the training-set covariance matrices for all 36 elements. The normalized MSE/L1 loss is:

```
L = mean_{i,j} loss_fn( (Ĉᵢⱼ - μᵢⱼ) / σᵢⱼ, (Cᵢⱼ - μᵢⱼ) / σᵢⱼ )
```

This correctly handles the block-diagonal structure of the covariance matrix: elements like σ²_{pz} (~10⁸) and cross-plane off-diagonals (~10⁻⁹) each contribute equally to the loss regardless of their raw magnitude.

## 7. Analyze Model Performance

Use `analyze_covariance.py` for all models trained with the current `train.py`. This script evaluates the model on the test set and produces several plots.

```bash
python analyze_covariance.py \
  --model-dir model-output-cov-l1-ft \
  --output-dir analysis-output-cov-l1-ft \
  --agreement-csv dataset-train.csv
```

### Key options

| Flag | Default | Description |
|---|---|---|
| `--model-dir` | `model-output` | Directory containing `model.pt` and transformer files |
| `--test-csv` | `dataset-test.csv` | CSV for evaluation metrics |
| `--output-dir` | `analysis-output-covariance` | Directory for plots and CSV results |
| `--agreement-csv` | `dataset-train.csv` | CSV for per-sample agreement overlay plots |
| `--agreement-max-samples` | `1000` | Max samples shown in overlay/dot plots |
| `--agreement-zoom-low-q` | `5` | Lower percentile for y-axis zoom in dot plot |
| `--agreement-zoom-high-q` | `95` | Upper percentile for y-axis zoom in dot plot |
| `--skip-scatter` | `False` | Skip scatter (pred vs true) plots |
| `--skip-sorted` | `False` | Skip sorted-by-magnitude overlay plots |

### Generated files in `--output-dir`

- `test_metrics.csv` — per-element MAE, RMSE, normalized RMSE, MAPE for all 36 covariance entries
- `training_curve.png` — train/val loss vs epoch
- `mae_per_element.png` — bar chart of MAE per covariance element
- `mape_per_element.png` — bar chart of MAPE per covariance element (most interpretable)
- `per_sample_agreement_overlay.png` — predicted vs true as overlaid lines per element
- `per_sample_agreement_zoomed_dots.png` — same but as dots with percentile y-zoom
- `scatter_pred_vs_true.png` — predicted vs true scatter with R² per element
- `sorted_by_magnitude_overlay.png` — samples sorted by |true| with predicted overlaid
- `covariance_histograms.png` — distribution of target vs predicted per element (6×6 grid)
- `covariance_std_heatmap.png` — log10(std) heatmap confirming block-diagonal structure

> **Note**: `analyze.py` is the old Cholesky-output analysis script and is **incompatible** with models trained with the current `train.py`. Always use `analyze_covariance.py` for current models.

## 8. (Optional) Create an Interpolation Holdout

To test how well the model interpolates, exclude a contiguous block of samples from the middle of the training set when sorted by a chosen input parameter:

```bash
python create_interpolation_holdout.py dataset-train.csv \
  --sort-column distgen:t_dist:sigma_t:value \
  --holdout-size 30 \
  --center-fraction 0.5 \
  --expand-equal-boundaries \
  --prefix dataset-train-sigmat-gap30
```

This produces:
- `dataset-train-sigmat-gap30-train.csv` — reduced training set (gap removed)
- `dataset-train-sigmat-gap30-secondary-test.csv` — the 30 held-out interpolation samples
- `dataset-train-sigmat-gap30-summary.csv` — gap range and row-count summary

### Key options

| Flag | Default | Description |
|---|---|---|
| `--sort-column` | required | Feature to sort on before extracting the holdout block |
| `--holdout-size` | `30` | Number of rows to remove |
| `--center-fraction` | `0.5` | Position of the block centre in sorted order (0=bottom, 1=top) |
| `--expand-equal-boundaries` | off | Expand block to include all rows with the same value as the block edges, ensuring a true feature gap |

Then train and evaluate on the gap:

```bash
python train.py \
  --train-csv dataset-train-sigmat-gap30-train.csv \
  --val-csv dataset-val.csv \
  --test-csv dataset-train-sigmat-gap30-secondary-test.csv \
  --loss-space cov --cov-loss l1 \
  --epochs 200 --patience 40 --batch-size 256 --lr 1e-3 \
  --finetune-batch-sizes 32 8 2 --finetune-epochs-per-stage 300 \
  --finetune-lr 1e-4 --finetune-lr-decay 0.5 \
  --finetune-plateau-patience 5 --finetune-min-lr 1e-6 \
  --output-dir model-output-interp-sigmat-gap30-l1

python analyze_covariance.py \
  --model-dir model-output-interp-sigmat-gap30-l1 \
  --test-csv dataset-train-sigmat-gap30-secondary-test.csv \
  --agreement-csv dataset-train-sigmat-gap30-train.csv \
  --output-dir analysis-output-interp-sigmat-gap30-l1
```

> **Choosing `--sort-column`**: Features with many unique values produce the most meaningful interpolation gaps. Good candidates in this dataset: `distgen:t_dist:sigma_t:value` (~6233 unique), `L0AF_phase:theta0_deg` (~6259 unique), `distgen:total_charge:value` (~6266 unique). Avoid `SOL10111:solenoid_field_scale` — it is concentrated around a single median value.

## 9. (Optional) Plot Input Parameter Distributions

```bash
python plot_input_histograms.py \
  --train-csv dataset-train.csv \
  --output-dir analysis-input-histograms
```

Generates:
- `input_parameter_histograms.png` — 6×7 histogram grid of all 38 input features with mean/std annotations
- `input_parameter_stats.csv` — full statistics table (mean, std, min, max, median, q25, q75)

## Reproducible Command Sequence

Run the full pipeline in this order:

```bash
# Data preparation
python create_csv_from_yaml.py dump.yaml dump.csv
python cleanup.py dump.csv dump-particles_241-not-null.csv --column particles_241
python create_cov_targets_from_particles.py dump-particles_241-not-null.csv cov-targets.csv --progress-every 200 --drop-failed
python create_dataset.py cov-targets.csv dataset.csv
python split_dataset.py dataset.csv

# Training (with fine-tuning, L1 loss)
python train.py \
  --loss-space cov --cov-loss l1 \
  --epochs 200 --patience 40 --batch-size 256 --lr 1e-3 \
  --finetune-batch-sizes 32 8 2 --finetune-epochs-per-stage 300 \
  --finetune-lr 1e-4 --finetune-lr-decay 0.5 \
  --finetune-plateau-patience 5 --finetune-min-lr 1e-6 \
  --output-dir model-output-cov-l1-ft

# Analysis
python analyze_covariance.py \
  --model-dir model-output-cov-l1-ft \
  --output-dir analysis-output-cov-l1-ft \
  --agreement-csv dataset-train.csv
```

## Outputs Produced by the Pipeline

- `dump.csv` — flat CSV converted from `dump.yaml`
- `dump-particles_241-not-null.csv` — cleaned CSV with valid particle paths
- `cov-targets.csv` — cleaned CSV plus 21 Cholesky target columns
- `dataset.csv` — final ML dataset with feature and target columns only
- `dataset-train.csv`, `dataset-val.csv`, `dataset-test.csv` — reproducible splits
- `<model-dir>/model.pt` — trained model weights (best validation loss)
- `<model-dir>/input_transformers.pt` — input feature scaler and column names
- `<model-dir>/output_transformers.pt` — Cholesky target scaler and column names
- `<model-dir>/covariance_transformers.pt` — per-element covariance normalizers
- `<model-dir>/training_history.csv` — per-epoch loss history
- `<analysis-dir>/test_metrics.csv` — per-element MAE/RMSE/MAPE on the test set
- `<analysis-dir>/*.png` — training curve, error plots, agreement plots

## Scripts Reference

| Script | Purpose |
|---|---|
| `create_csv_from_yaml.py` | Convert `dump.yaml` to flat CSV |
| `cleanup.py` | Filter rows missing `particles_241` |
| `create_cov_targets_from_particles.py` | Compute Cholesky targets from OpenPMD particle files |
| `create_dataset.py` | Drop non-training columns to produce final dataset |
| `split_dataset.py` | Split dataset into train/val/test CSVs |
| `train.py` | Train covariance MLP; supports L1/MSE loss and staged fine-tuning |
| `analyze_covariance.py` | Evaluate a trained model; generates metrics and all agreement/diagnostic plots |
| `plot_input_histograms.py` | Plot distributions of all 38 input parameters from any split CSV |
| `create_interpolation_holdout.py` | Carve out a sorted contiguous block from training data as a secondary interpolation test set |