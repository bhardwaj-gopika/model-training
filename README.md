# Modeling Data Preparation Pipeline

This repository contains the scripts used to turn `dump.yaml` into a reproducible machine learning dataset for predicting beam covariance targets.

## Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate modeling
```

## End-to-End Workflow

The pipeline has six stages:

1. Convert `dump.yaml` to CSV
2. Remove rows without `particles_241`
3. Create Cholesky covariance targets from particle files
4. Drop non-feature and non-target columns to build the final dataset
5. Split the dataset into train, validation, and test CSVs
6. Train the covariance MLP

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

Train the covariance prediction MLP on the prepared splits:

```bash
python train.py
```

Default behavior:

- Epochs: `200`
- Batch size: `256`
- Learning rate: `1e-3` (halved via `ReduceLROnPlateau` after 10 stagnant epochs)
- Early stopping patience: `20` epochs
- Seed: `42`

Generated files in `model-output/`:

- `model.pt` — trained model weights
- `input_transformers.pt` — input feature scaler (x_mean, x_std) and feature column names
- `output_transformers.pt` — output target scaler (y_mean, y_std) and target column names
- `training_history.csv` — per-epoch train and validation MSE

Model architecture (38 inputs → 21 outputs):

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
 Linear(100 → 21)
```

Example with custom settings:

```bash
python train.py \
  --epochs 500 \
  --batch-size 128 \
  --lr 5e-4 \
  --patience 30 \
  --output-dir model-output
```

## Reproducible Command Sequence

Run the full pipeline in this order:

```bash
python create_csv_from_yaml.py dump.yaml dump.csv
python cleanup.py dump.csv dump-particles_241-not-null.csv --column particles_241
python create_cov_targets_from_particles.py dump-particles_241-not-null.csv cov-targets.csv --progress-every 200 --drop-failed
python create_dataset.py cov-targets.csv dataset.csv
python split_dataset.py dataset.csv
python train.py
```

## Outputs Produced by the Pipeline

- `dump.csv`: flat CSV converted from `dump.yaml`
- `dump-particles_241-not-null.csv`: cleaned CSV with valid particle paths
- `cov-targets.csv`: cleaned CSV plus 21 Cholesky target columns
- `dataset.csv`: final ML dataset with feature and target columns only
- `dataset-train.csv`, `dataset-val.csv`, `dataset-test.csv`: reproducible splits
- `model-output/model.pt`: trained model weights
- `model-output/input_transformers.pt`: input feature scaler and column names
- `model-output/output_transformers.pt`: output target scaler and column names
- `model-output/training_history.csv`: per-epoch loss history