# Modeling Data Preparation Pipeline

This repository contains the scripts used to turn `dump.yaml` into a reproducible machine learning dataset for predicting 6x6 beam covariance matrices.

The current model is trained on 7 approved simulator-space input parameters and predicts the 6x6 covariance matrix of the beam phase space `(x, px, y, py, z, pz)`. Internally the network predicts 21 lower-triangular Cholesky factors, reconstructs `C = L @ L^T`, and computes loss in normalized covariance space using per-element statistics from the training split. The inference entrypoint also supports machine-facing PV inputs by mapping them into simulator parameter space before normalization and model evaluation.

## Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate modeling
```

## End-to-End Workflow

The current workflow has 10 stages:

1. Convert `dump.yaml` to CSV
2. Remove rows without `particles_241`
3. Create Cholesky covariance targets from particle files
4. Keep the approved model inputs and target columns
5. Split the dataset into train, validation, and test CSVs
6. Train the covariance MLP
7. Analyze model performance
8. Run inference from simulator or machine inputs
9. (Optional) Create an interpolation holdout
10. (Optional) Plot input parameter distributions

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

Keep only the approved model inputs and covariance targets:

```bash
python create_dataset.py cov-targets.csv dataset.csv
```

Expected output:

- Input: `cov-targets.csv`
- Output: `dataset.csv`

Current dataset structure:

- 7 feature columns
- 21 target columns: `cov_chol_0` through `cov_chol_20`
- Total: 28 columns

### Feature and Target Columns

Feature columns:

- `CQ10121:b1_gradient`
- `GUNF:rf_field_scale`
- `GUNF:theta0_deg`
- `SOL10111:solenoid_field_scale`
- `SQ10122:b1_gradient`
- `distgen:t_dist:sigma_t:value`
- `distgen:total_charge:value`

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

The model consumes the 7 simulator-space inputs above and outputs a full 6x6 covariance matrix. Targets are the 21 lower-triangular Cholesky factors (`cov_chol_0..20`); the network reconstructs `C = L @ L^T` internally. Loss is computed in per-element normalized covariance space.

### Quick start

```bash
python train.py \
  --cov-loss l1 \
  --epochs 200 \
  --patience 40 \
  --batch-size 256 \
  --lr 1e-3 \
  --output-dir model-output-cov-l1
```

### With staged fine-tuning

```bash
python train.py \
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
| `--cov-loss` | `mse` | Covariance-space objective: `mse` or `l1` |
| `--epochs` | `200` | Max training epochs |
| `--patience` | `20` | Early-stopping patience in epochs (`0` disables) |
| `--batch-size` | `256` | Mini-batch size |
| `--lr` | `1e-3` | Initial learning rate |
| `--finetune-batch-sizes` | `None` | Space-separated batch sizes for staged fine-tuning, for example `32 8 2` |
| `--finetune-epochs-per-stage` | `0` | Epochs per fine-tuning stage |
| `--finetune-lr` | `1e-4` | Initial LR for fine-tuning |
| `--finetune-lr-decay` | `0.5` | LR multiplier applied after each stage |
| `--finetune-plateau-patience` | `5` | Plateau patience during fine-tuning |
| `--finetune-min-lr` | `1e-6` | Minimum LR floor during fine-tuning |
| `--output-dir` | `model-output` | Directory for checkpoints and transformer files |

### Generated files in `--output-dir`

- `model.pt` - best validation checkpoint
- `input_transformers.pt` - input feature scaler (`x_mean`, `x_std`) and `feature_cols`
- `output_transformers.pt` - Cholesky target scaler (`y_mean`, `y_std`) and `target_cols`
- `covariance_transformers.pt` - per-element covariance normalizers (`cov_mean`, `cov_std`, `cov_labels`)
- `training_history.csv` - per-epoch train and validation loss

### Model architecture

7 inputs -> 21 Cholesky outputs -> 6x6 covariance output:

```text
Linear(7 -> 100), ELU
Linear(100 -> 200), ELU, Dropout(0.05)
Linear(200 -> 200), ELU, Dropout(0.05)
Linear(200 -> 300), ELU, Dropout(0.05)
Linear(300 -> 300), ELU, Dropout(0.05)
Linear(300 -> 200), ELU, Dropout(0.05)
Linear(200 -> 100), ELU, Dropout(0.05)
Linear(100 -> 100), ELU
Linear(100 -> 100), ELU
Linear(100 -> 21) -> build L (lower triangular) -> C = L @ L^T
```

### Normalization

Before training, the code computes:

- `x_mean`, `x_std` from the training split inputs
- `y_mean`, `y_std` from the training split Cholesky targets
- `cov_mean`, `cov_std` from the reconstructed training split covariance matrices

The loss is then applied in normalized covariance space so each covariance element contributes on a comparable scale.

## 7. Analyze Model Performance

Use `analyze_covariance.py` for models trained with the current `train.py`.

```bash
python analyze_covariance.py \
  --model-dir model-output-cov-l1-ft \
  --output-dir analysis-output-cov-l1-ft \
  --agreement-csv dataset-train.csv
```

Generated files in `--output-dir` include:

- `test_metrics.csv` - per-element MAE, RMSE, normalized RMSE, and MAPE for all 36 covariance entries
- `training_curve.png` - train and validation loss vs epoch
- `mae_per_element.png`
- `mape_per_element.png`
- `per_sample_agreement_overlay.png`
- `per_sample_agreement_zoomed_dots.png`
- `scatter_pred_vs_true.png`
- `sorted_by_magnitude_overlay.png`
- `covariance_histograms.png`
- `covariance_std_heatmap.png`

## 8. Run Inference

Use `infer_covariance.py` to run the trained model on either simulator-space inputs or machine-facing PV inputs.

### Supported input schemas

Simulator-space columns must match the 7 training features exactly:

- `CQ10121:b1_gradient`
- `GUNF:rf_field_scale`
- `GUNF:theta0_deg`
- `SOL10111:solenoid_field_scale`
- `SQ10122:b1_gradient`
- `distgen:t_dist:sigma_t:value`
- `distgen:total_charge:value`

Machine-facing columns are derived from `pv_mapping.py` in the same feature order:

- `QUAD:IN10:121:BACT`
- `KLYS:LI10:21:AMPL`
- `KLYS:LI10:21:PHAS`
- `SOLN:IN10:121:BACT`
- `QUAD:IN10:122:BACT`
- `distgen:t_dist:sigma_t:value`
- `TORO:IN10:591:TMIT_PC`

The sixth input currently has no machine PV alias, so its machine-input name remains `distgen:t_dist:sigma_t:value`.

### Input-space selection

`infer_covariance.py` supports `--input-space auto|sim|pv`:

- `auto` detects the schema from the CSV headers and prefers the PV schema when both are present.
- `sim` requires the 7 simulator columns.
- `pv` requires the 7 machine-facing columns listed above.

### Inference logic

The inference path is:

1. Load `model.pt` and `input_transformers.pt` from `--model-dir`.
2. Read the input CSV and resolve whether it is simulator-space or machine-space input.
3. If the CSV is machine-space, map PV values into simulator parameter space using the affine rules in `pv_mapping.py`.
4. Normalize the simulator-space inputs with the saved training `x_mean` and `x_std`.
5. Run the surrogate model to produce a `(N, 6, 6)` covariance prediction array.
6. Export a `lume-torch` wrapper, run the same machine inputs through that wrapper, and assert that the direct and wrapper predictions match.
7. Save both a flat CSV and a NumPy array of the predictions.

### Example: simulator-space inference

```bash
python infer_covariance.py \
  --model-dir model-output-cov-l1 \
  --input-csv dataset-test.csv \
  --input-space sim \
  --output-dir inference-sim
```

### Example: machine-PV inference

```bash
python infer_covariance.py \
  --model-dir model-output-cov-l1 \
  --input-csv machine-input-test.csv \
  --input-space pv \
  --output-dir inference-machine-test
```

### Generated outputs

- `predicted_covariances.csv` - one row per sample containing:
  - `sample_index`
  - the machine-space inputs used for evaluation
  - `sim_<feature>` columns containing the mapped simulator values
  - `pred_cov_00` through `pred_cov_55`
- `predicted_covariances.npy` - raw `(N, 6, 6)` prediction array

The script also prints one selected covariance matrix, the machine-space inputs for that row, and the mapped simulator-space values. `--print-row` defaults to `0`.

## 9. (Optional) Create an Interpolation Holdout

To test interpolation performance, exclude a contiguous block of samples from the middle of the training set when sorted by a chosen input parameter:

```bash
python create_interpolation_holdout.py dataset-train.csv \
  --sort-column distgen:t_dist:sigma_t:value \
  --holdout-size 30 \
  --center-fraction 0.5 \
  --expand-equal-boundaries \
  --prefix dataset-train-sigmat-gap30
```

This produces:

- `dataset-train-sigmat-gap30-train.csv` - reduced training set with the gap removed
- `dataset-train-sigmat-gap30-secondary-test.csv` - held-out interpolation samples
- `dataset-train-sigmat-gap30-summary.csv` - gap range and row-count summary

Example training and evaluation on the gap split:

```bash
python train.py \
  --train-csv dataset-train-sigmat-gap30-train.csv \
  --val-csv dataset-val.csv \
  --test-csv dataset-train-sigmat-gap30-secondary-test.csv \
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
  --output-dir model-output-interp-sigmat-gap30-l1

python analyze_covariance.py \
  --model-dir model-output-interp-sigmat-gap30-l1 \
  --test-csv dataset-train-sigmat-gap30-secondary-test.csv \
  --agreement-csv dataset-train-sigmat-gap30-train.csv \
  --output-dir analysis-output-interp-sigmat-gap30-l1
```

## 10. (Optional) Plot Input Parameter Distributions

```bash
python plot_input_histograms.py \
  --train-csv dataset-train.csv \
  --output-dir analysis-input-histograms
```

Generates:

- `input_parameter_histograms.png` - histogram grid for all input features in the provided CSV
- `input_parameter_stats.csv` - summary statistics for each input feature

## Reproducible Command Sequence

Run the full pipeline in this order:

```bash
# Data preparation
python create_csv_from_yaml.py dump.yaml dump.csv
python cleanup.py dump.csv dump-particles_241-not-null.csv --column particles_241
python create_cov_targets_from_particles.py dump-particles_241-not-null.csv cov-targets.csv --progress-every 200 --drop-failed
python create_dataset.py cov-targets.csv dataset.csv
python split_dataset.py dataset.csv

# Training
python train.py \
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

# Analysis
python analyze_covariance.py \
  --model-dir model-output-cov-l1-ft \
  --output-dir analysis-output-cov-l1-ft \
  --agreement-csv dataset-train.csv

# Inference
python infer_covariance.py \
  --model-dir model-output-cov-l1-ft \
  --input-csv dataset-test.csv \
  --input-space sim \
  --output-dir inference-sim
```

## Outputs Produced by the Pipeline

- `dump.csv` - flat CSV converted from `dump.yaml`
- `dump-particles_241-not-null.csv` - cleaned CSV with valid particle paths
- `cov-targets.csv` - cleaned CSV plus 21 Cholesky target columns
- `dataset.csv` - final ML dataset with 7 inputs and 21 targets
- `dataset-train.csv`, `dataset-val.csv`, `dataset-test.csv` - reproducible splits
- `<model-dir>/model.pt` - trained model weights
- `<model-dir>/input_transformers.pt` - input feature scaler and feature column names
- `<model-dir>/output_transformers.pt` - Cholesky target scaler and target column names
- `<model-dir>/covariance_transformers.pt` - covariance-element normalizers
- `<model-dir>/training_history.csv` - per-epoch loss history
- `<inference-dir>/predicted_covariances.csv` - flat inference output with inputs and predictions
- `<inference-dir>/predicted_covariances.npy` - raw `(N, 6, 6)` predictions
- `lumetorchyaml/injector_machine.yaml` - exported `lume-torch` wrapper generated during inference
- `<analysis-dir>/test_metrics.csv` - per-element evaluation metrics
- `<analysis-dir>/*.png` - training and diagnostic plots

## Scripts Reference

| Script | Purpose |
|---|---|
| `create_csv_from_yaml.py` | Convert `dump.yaml` to a flat CSV |
| `cleanup.py` | Filter rows missing `particles_241` |
| `create_cov_targets_from_particles.py` | Compute Cholesky targets from OpenPMD particle files |
| `create_dataset.py` | Keep the approved 7 inputs plus the 21 covariance targets |
| `split_dataset.py` | Split the dataset into train, validation, and test CSVs |
| `train.py` | Train the covariance MLP and save scaling metadata |
| `analyze_covariance.py` | Evaluate a trained model and generate plots |
| `infer_covariance.py` | Run inference from simulator-space or machine-space CSV inputs |
| `pv_mapping.py` | Define the affine machine-PV to simulator-parameter mapping |
| `plot_input_histograms.py` | Plot input distributions from a split CSV |
| `create_interpolation_holdout.py` | Carve out a contiguous interpolation gap from the training set |
