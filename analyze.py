"""Analyze training results and model performance."""
# python analyze.py

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Load model and transformers ────────────────────────────────────────────────
def load_model_and_transformers(model_dir: Path, n_inputs=38, n_outputs=21):
    """Load trained model and transformation dictionaries."""
    from train import build_model
    
    model = build_model(n_inputs, n_outputs)
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    model.eval()
    
    input_tr = torch.load(model_dir / "input_transformers.pt")
    output_tr = torch.load(model_dir / "output_transformers.pt")
    
    return model, input_tr, output_tr


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_on_test(model, test_loader, output_tr, device):
    """Evaluate model on test set and return metrics."""
    y_mean = output_tr["y_mean"].to(device)
    y_std = output_tr["y_std"].to(device)
    target_cols = output_tr["target_cols"]
    
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    preds_norm = np.concatenate(all_preds)
    targets_norm = np.concatenate(all_targets)
    
    # Denormalize
    preds_raw = preds_norm * y_std.cpu().numpy() + y_mean.cpu().numpy()
    targets_raw = targets_norm * y_std.cpu().numpy() + y_mean.cpu().numpy()
    
    # Metrics
    mae_per_target = np.abs(preds_raw - targets_raw).mean(axis=0)
    mse_per_target = ((preds_raw - targets_raw) ** 2).mean(axis=0)
    rmse_per_target = np.sqrt(mse_per_target)

    # Mean absolute percentage error per target.
    # Rows where |true| is very small (< 1% of per-target std) are excluded
    # to avoid division by near-zero inflating MAPE.
    abs_true = np.abs(targets_raw)
    target_scale = abs_true.std(axis=0)
    mask = abs_true > 0.01 * target_scale[None, :]  # (N, 21) boolean
    pct_errors = np.where(mask, np.abs(preds_raw - targets_raw) / abs_true * 100, np.nan)
    mape_per_target = np.nanmean(pct_errors, axis=0)
    mape_overall = np.nanmean(pct_errors)

    mae_overall = mae_per_target.mean()
    rmse_overall = rmse_per_target.mean()
    
    return {
        "preds_raw": preds_raw,
        "targets_raw": targets_raw,
        "mae_per_target": mae_per_target,
        "rmse_per_target": rmse_per_target,
        "mape_per_target": mape_per_target,
        "mae_overall": mae_overall,
        "rmse_overall": rmse_overall,
        "mape_overall": mape_overall,
        "target_cols": target_cols,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze training results.")
    parser.add_argument(
        "--model-dir",
        default="model-output",
        help="Model output directory (default: model-output)",
    )
    parser.add_argument(
        "--test-csv",
        default="dataset-test.csv",
        help="Test CSV path (default: dataset-test.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis-output",
        help="Analysis output directory (default: analysis-output)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)
    
    # ── Load training history ────────────────────────────────────────────────────
    print(f"[run] Loading training history from {model_dir}/training_history.csv", flush=True)
    history_df = pd.read_csv(model_dir / "training_history.csv")
    n_epochs = len(history_df)
    print(f"[run] Loaded {n_epochs} epochs of training history", flush=True)
    
    # ── Load model and transformers ──────────────────────────────────────────────
    print(f"[run] Loading model and transformers from {model_dir}", flush=True)
    model, input_tr, output_tr = load_model_and_transformers(model_dir)
    model.to(device)
    
    # ── Load test set ────────────────────────────────────────────────────────────
    print(f"[run] Loading test data from {args.test_csv}", flush=True)
    test_df = pd.read_csv(args.test_csv, low_memory=False)
    feature_cols = input_tr["feature_cols"]
    target_cols = output_tr["target_cols"]
    x_mean = input_tr["x_mean"]
    x_std = input_tr["x_std"]
    y_mean = output_tr["y_mean"]
    y_std = output_tr["y_std"]
    
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_cols].values.astype(np.float32)
    
    X_test = (X_test - x_mean.numpy()) / x_std.numpy()
    y_test = (y_test - y_mean.numpy()) / y_std.numpy()
    
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    # ── Evaluate on test set ─────────────────────────────────────────────────────
    print("\n[run] Evaluating on test set ...", flush=True)
    metrics = evaluate_on_test(model, test_loader, output_tr, device)
    
    print(f"\n[results] Test MAE (overall): {metrics['mae_overall']:.6f}", flush=True)
    print(f"[results] Test RMSE (overall): {metrics['rmse_overall']:.6f}", flush=True)
    print(f"[results] Test MAPE (overall): {metrics['mape_overall']:.2f}%", flush=True)
    print("\n[results] Per-target MAE / MAPE (test set):", flush=True)
    for col, mae, rmse, mape in zip(
        metrics["target_cols"],
        metrics["mae_per_target"],
        metrics["rmse_per_target"],
        metrics["mape_per_target"],
    ):
        print(f"  {col}: MAE={mae:.6f}  RMSE={rmse:.6f}  MAPE={mape:.2f}%", flush=True)
    
    # ── Save metrics to CSV ──────────────────────────────────────────────────────
    metrics_df = pd.DataFrame({
        "target": metrics["target_cols"],
        "mae": metrics["mae_per_target"],
        "rmse": metrics["rmse_per_target"],
        "mape_pct": metrics["mape_per_target"],
    })
    metrics_df.to_csv(output_dir / "test_metrics.csv", index=False)
    print(f"\n[run] Test metrics saved to {output_dir}/test_metrics.csv", flush=True)
    
    # ── Plot training curve ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = np.arange(1, n_epochs + 1)
    ax.plot(epochs, history_df["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, history_df["val_loss"], label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (normalized)")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=150)
    print(f"[run] Training curve saved to {output_dir}/training_curve.png", flush=True)
    
    # ── Plot per-target MAE ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x_pos = np.arange(len(metrics["target_cols"]))
    ax.bar(x_pos, metrics["mae_per_target"], alpha=0.7, color="steelblue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics["target_cols"], rotation=45, ha="right")
    ax.set_ylabel("MAE (original units)")
    ax.set_title("Test MAE Per Target")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "mae_per_target.png", dpi=150)
    print(f"[run] Per-target MAE plot saved to {output_dir}/mae_per_target.png", flush=True)

    # ── Plot per-target MAPE ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    mape_vals = metrics["mape_per_target"]
    colors = ["tomato" if v > 50 else "steelblue" for v in mape_vals]
    ax.bar(x_pos, mape_vals, alpha=0.8, color=colors)
    ax.axhline(y=metrics["mape_overall"], color="black", linestyle="--",
               linewidth=1.2, label=f"Overall MAPE = {metrics['mape_overall']:.1f}%")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics["target_cols"], rotation=45, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Test Percentage Error Per Cholesky Factor (test set)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "mape_per_target.png", dpi=150)
    print(f"[run] Per-target MAPE plot saved to {output_dir}/mape_per_target.png", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────────
    print(f"\n[run] Analysis complete. Results in {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
