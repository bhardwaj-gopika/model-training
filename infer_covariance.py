"""Run covariance inference from raw machine-unit inputs.

The script expects a CSV containing the original feature columns in machine units.
It normalizes inputs using the saved training transformers, runs the trained model,
and writes predicted 6x6 covariance matrices back out in raw covariance units.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from lume_torch.variables import TorchScalarVariable, TorchNDVariable
from lume_torch.models import TorchModel, TorchModule
from botorch.models.transforms.input import AffineInputTransform



from train import build_model


def covariance_labels():
    return [f"cov_{row}{col}" for row in range(6) for col in range(6)]


def load_model_and_transformers(model_dir: Path, device: torch.device):
    input_tr = torch.load(model_dir / "input_transformers.pt")

    feature_cols = list(input_tr["feature_cols"])
    n_inputs = len(feature_cols)
    n_outputs = 21

    model = build_model(n_inputs, n_outputs)
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True, map_location=device))
    model.to(device)
    model.eval()
    return model, input_tr


def build_parser():
    parser = argparse.ArgumentParser(
        description="Infer 6x6 covariance matrices from raw machine-unit input CSV rows."
    )
    parser.add_argument(
        "--model-dir",
        default="model-output-clean-gap200",
        help="Directory containing model.pt and transformer files (default: model-output-clean-gap200)",
    )
    parser.add_argument(
        "--input-csv",
        default="dataset-test.csv",
        help="CSV with raw feature columns in machine units (default: dataset-test.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="inference-output",
        help="Directory for inference outputs (default: inference-output)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)",
    )
    parser.add_argument(
        "--print-row",
        type=int,
        default=0,
        help="Row index whose predicted covariance matrix should be printed (default: 0)",
    )
    return parser

def create_lume_torch(model, feature_cols, input_tr):
    feature_cols = list(input_tr["feature_cols"])
    x_mean = input_tr["x_mean"]
    x_std = input_tr["x_std"]

    # For wrapping the model in lume-torch
    input_variables = [TorchScalarVariable(name=col, default_value=x_mean[idx]) for idx, col in enumerate(feature_cols)]
    output_variables = [TorchNDVariable(name="covariance_matrix", shape=(6,6))]

    input_transform = AffineInputTransform(d=len(feature_cols), coefficient=x_std, offset=x_mean)

    model = TorchModel(
        model=model,
        input_variables=input_variables,
        output_variables=output_variables,
        input_transformers=[input_transform],
        precision = "single"
    )

    model.dump("lume-torch-yaml/injector_sim.yaml")

    return TorchModule(model = TorchModel("lume-torch-yaml/injector_sim.yaml"))



def main():
    args = build_parser().parse_args()

    model_dir = Path(args.model_dir)
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)
    print(f"[run] Loading model from {model_dir}", flush=True)
    model, input_tr = load_model_and_transformers(model_dir, device)

    feature_cols = list(input_tr["feature_cols"])
    x_mean = input_tr["x_mean"].cpu().numpy().astype(np.float32)
    x_std = input_tr["x_std"].cpu().numpy().astype(np.float32)

    
    print(f"[run] Reading input CSV: {input_csv}", flush=True)
    df = pd.read_csv(input_csv, low_memory=False)

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise SystemExit(
            "Input CSV is missing required feature columns: " + ", ".join(missing_features)
        )

    X_raw = df[feature_cols].values.astype(np.float32)
    X_norm = (X_raw - x_mean) / x_std
    loader1 = DataLoader(TensorDataset(torch.from_numpy(X_norm)), batch_size=args.batch_size)
    
    print(f"[run] Running inference for {len(df)} rows", flush=True)
    pred_batches = []
    pred_batches_lume = []
    with torch.no_grad():
        for (X_batch,) in loader1:
            pred_cov = model(X_batch.to(device)).cpu().numpy()
            pred_batches.append(pred_cov)

    lume_model = create_lume_torch(model, feature_cols, input_tr)

    loader2 = DataLoader(TensorDataset(torch.from_numpy(X_raw)), batch_size=args.batch_size)

    with torch.no_grad():
        for (X_batch,) in loader2:
            pred_cov_lume = lume_model(X_batch.to(device)).cpu().numpy()
            pred_batches_lume.append(pred_cov_lume)

    preds_cov = np.concatenate(pred_batches, axis=0)
    preds_cov_lume = np.concatenate(pred_batches_lume, axis=0)

    if not np.allclose(preds_cov, preds_cov_lume, rtol=1e-10, atol=1e-10):
        max_abs_diff = float(np.max(np.abs(preds_cov - preds_cov_lume)))
        raise AssertionError(
            f"Mismatch between direct model and lume-torch predictions; max abs diff={max_abs_diff:.6e}"
        )

    pred_flat = preds_cov.reshape(len(df), 36)
    cov_cols = covariance_labels()

    base_df = df[feature_cols].copy()
    base_df.insert(0, "sample_index", np.arange(len(base_df), dtype=np.int64))
    pred_df = pd.DataFrame(
        {f"pred_{col}": pred_flat[:, idx] for idx, col in enumerate(cov_cols)}
    )
    result_parts = [base_df, pred_df]

    result_df = pd.concat(result_parts, axis=1)
    result_df.to_csv(output_dir / "predicted_covariances.csv", index=False)
    np.save(output_dir / "predicted_covariances.npy", preds_cov)

    row_index = args.print_row
    if row_index < 0 or row_index >= len(df):
        raise SystemExit(f"--print-row must be between 0 and {len(df) - 1}")

    np.set_printoptions(precision=6, suppress=False)
    print(f"[run] Saved flat predictions to {output_dir / 'predicted_covariances.csv'}", flush=True)
    print(f"[run] Saved 3D covariance array to {output_dir / 'predicted_covariances.npy'}", flush=True)
    print(f"[run] Predicted covariance matrix for row {row_index}:", flush=True)
    print(preds_cov[row_index], flush=True)
    print("[run] Input feature values for that row:", flush=True)
    print(df.loc[row_index, feature_cols], flush=True)
    print(
        "[run] Flow: raw machine-unit inputs -> surrogate model -> predicted 6x6 covariance matrix.",
        flush=True,
    )


if __name__ == "__main__":
    main()