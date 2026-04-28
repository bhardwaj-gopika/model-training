"""Run covariance inference from sim-parameter or machine-PV inputs.

The script accepts either simulator-parameter columns or machine-facing PV-unit
columns. Machine inputs are first mapped into simulator parameter space and then
normalized using the saved training transformers before model evaluation.
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
from pv_mapping import (
    build_pv_to_sim_transform,
    machine_input_names,
    machine_to_sim_array,
    sim_to_machine_array,
)


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
        description="Infer 6x6 covariance matrices from sim-parameter or machine-PV input CSV rows."
    )
    parser.add_argument(
        "--model-dir",
        default="model-output-clean-gap200",
        help="Directory containing model.pt and transformer files (default: model-output-clean-gap200)",
    )
    parser.add_argument(
        "--input-csv",
        default="dataset-test.csv",
        help="CSV with either sim-parameter columns or machine PV columns (default: dataset-test.csv)",
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
    parser.add_argument(
        "--input-space",
        choices=["auto", "sim", "pv"],
        default="auto",
        help="Interpret input CSV columns as sim parameters, machine PVs, or auto-detect (default: auto)",
    )
    return parser

def resolve_input_space(df: pd.DataFrame, feature_cols, requested_space: str):
    sim_cols = list(feature_cols)
    pv_cols = machine_input_names(feature_cols)
    has_sim = all(col in df.columns for col in sim_cols)
    has_pv = all(col in df.columns for col in pv_cols)

    if requested_space == "sim":
        if not has_sim:
            missing = [col for col in sim_cols if col not in df.columns]
            raise SystemExit("Input CSV is missing required sim columns: " + ", ".join(missing))
        return "sim", sim_cols, pv_cols

    if requested_space == "pv":
        if not has_pv:
            missing = [col for col in pv_cols if col not in df.columns]
            raise SystemExit("Input CSV is missing required PV columns: " + ", ".join(missing))
        return "pv", sim_cols, pv_cols

    if has_pv:
        return "pv", sim_cols, pv_cols
    if has_sim:
        return "sim", sim_cols, pv_cols

    missing_sim = [col for col in sim_cols if col not in df.columns]
    missing_pv = [col for col in pv_cols if col not in df.columns]
    raise SystemExit(
        "Input CSV does not match either supported schema. "
        f"Missing sim columns: {missing_sim}. Missing PV columns: {missing_pv}."
    )


def create_lume_torch(model, input_tr):
    feature_cols = list(input_tr["feature_cols"])
    x_mean = input_tr["x_mean"].to(dtype=torch.float32)
    x_std = input_tr["x_std"].to(dtype=torch.float32)
    pv_cols = machine_input_names(feature_cols)
    pv_defaults = sim_to_machine_array(x_mean.cpu().numpy()[None, :], feature_cols)[0]

    input_variables = [
        TorchScalarVariable(name=col, default_value=float(pv_defaults[idx]))
        for idx, col in enumerate(pv_cols)
    ]
    output_variables = [TorchNDVariable(name="covariance_matrix", shape=(6, 6))]

    pv_to_sim_transform = build_pv_to_sim_transform(feature_cols)
    normalization_transform = AffineInputTransform(
        d=len(feature_cols), coefficient=x_std, offset=x_mean
    )

    torch_model = TorchModel(
        model=model,
        input_variables=input_variables,
        output_variables=output_variables,
        input_transformers=[pv_to_sim_transform, normalization_transform],
        precision="single",
    )

    torch_model.dump("lumetorchyaml-nojit/injector_machine.yaml")

    return TorchModule(model=torch_model)



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
    input_space, sim_cols, pv_cols = resolve_input_space(df, feature_cols, args.input_space)

    if input_space == "sim":
        X_sim = df[sim_cols].values.astype(np.float32)
        X_machine = sim_to_machine_array(X_sim, feature_cols)
        input_df = pd.DataFrame(X_machine, columns=pv_cols, index=df.index)
        print("[run] Detected simulator-parameter input columns", flush=True)
    else:
        X_machine = df[pv_cols].values.astype(np.float32)
        X_sim = machine_to_sim_array(X_machine, feature_cols)
        input_df = df[pv_cols].copy()
        print("[run] Detected machine-PV input columns; applying PV -> sim transform", flush=True)

    X_sim_for_lume_check = machine_to_sim_array(X_machine, feature_cols)
    X_norm = (X_sim - x_mean) / x_std
    X_norm_lume_check = (X_sim_for_lume_check - x_mean) / x_std
    loader1 = DataLoader(TensorDataset(torch.from_numpy(X_norm)), batch_size=args.batch_size)
    loader1_lume_check = DataLoader(
        TensorDataset(torch.from_numpy(X_norm_lume_check)), batch_size=args.batch_size
    )

    print(f"[run] Running inference for {len(df)} rows", flush=True)
    pred_batches = []
    pred_batches_lume = []
    with torch.no_grad():
        for (X_batch,) in loader1:
            pred_cov = model(X_batch.to(device)).cpu().numpy()
            pred_batches.append(pred_cov)

    pred_batches_lume_ref = []
    with torch.no_grad():
        for (X_batch,) in loader1_lume_check:
            pred_cov = model(X_batch.to(device)).cpu().numpy()
            pred_batches_lume_ref.append(pred_cov)

    lume_model = create_lume_torch(model, input_tr)

    loader2 = DataLoader(TensorDataset(torch.from_numpy(X_machine)), batch_size=args.batch_size)

    with torch.no_grad():
        for (X_batch,) in loader2:
            pred_cov_lume = lume_model(X_batch.to(device)).cpu().numpy()
            pred_batches_lume.append(pred_cov_lume)

    preds_cov = np.concatenate(pred_batches, axis=0)
    preds_cov_lume_ref = np.concatenate(pred_batches_lume_ref, axis=0)
    preds_cov_lume = np.concatenate(pred_batches_lume, axis=0)

    if not np.allclose(preds_cov_lume_ref, preds_cov_lume, rtol=1e-10, atol=1e-10):
        max_abs_diff = float(np.max(np.abs(preds_cov_lume_ref - preds_cov_lume)))
        raise AssertionError(
            f"Mismatch between direct model and lume-torch predictions; max abs diff={max_abs_diff:.6e}"
        )

    pred_flat = preds_cov.reshape(len(df), 36)
    cov_cols = covariance_labels()

    base_df = input_df.copy()
    base_df.insert(0, "sample_index", np.arange(len(base_df), dtype=np.int64))
    mapped_sim_df = pd.DataFrame(X_sim, columns=[f"sim_{col}" for col in feature_cols], index=df.index)
    pred_df = pd.DataFrame(
        {f"pred_{col}": pred_flat[:, idx] for idx, col in enumerate(cov_cols)}
    )
    result_parts = [base_df, mapped_sim_df, pred_df]

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
    print("[run] Machine-space input values for that row:", flush=True)
    print(input_df.loc[row_index, pv_cols], flush=True)
    print("[run] Mapped simulator-parameter values for that row:", flush=True)
    print(pd.Series(X_sim[row_index], index=feature_cols), flush=True)
    print(
        "[run] Flow: machine PV inputs -> PV-to-sim affine transform -> normalization -> surrogate model -> predicted 6x6 covariance matrix.",
        flush=True,
    )


if __name__ == "__main__":
    main()