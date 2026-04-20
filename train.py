"""Train the covariance prediction MLP on the prepared dataset splits."""
# .venv/bin/python train.py

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Column definitions ────────────────────────────────────────────────────────
TARGET_PREFIX = "cov_chol_"


def get_feature_target_columns(df: pd.DataFrame):
    target_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    feature_cols = [c for c in df.columns if not c.startswith(TARGET_PREFIX)]
    return feature_cols, target_cols


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(n_inputs: int, n_outputs: int) -> nn.Sequential:
    """
    MLP matching the existing architecture, scaled to n_inputs / n_outputs.

    Original structure (in=16, out=5):
      Linear 16→100, ELU
      Linear 100→200, ELU, Dropout 0.05
      Linear 200→200, ELU, Dropout 0.05
      Linear 200→300, ELU, Dropout 0.05
      Linear 300→300, ELU, Dropout 0.05
      Linear 300→200, ELU, Dropout 0.05
      Linear 200→100, ELU, Dropout 0.05
      Linear 100→100, ELU
      Linear 100→100, ELU
      Linear 100→5
    """
    drop = 0.05
    return nn.Sequential(
        # (0)–(1)
        nn.Linear(n_inputs, 100),
        nn.ELU(),
        # (2)–(4)
        nn.Linear(100, 200),
        nn.ELU(),
        nn.Dropout(p=drop),
        # (5)–(7)
        nn.Linear(200, 200),
        nn.ELU(),
        nn.Dropout(p=drop),
        # (8)–(10)
        nn.Linear(200, 300),
        nn.ELU(),
        nn.Dropout(p=drop),
        # (11)–(13)
        nn.Linear(300, 300),
        nn.ELU(),
        nn.Dropout(p=drop),
        # (14)–(16)
        nn.Linear(300, 200),
        nn.ELU(),
        nn.Dropout(p=drop),
        # (17)–(19)
        nn.Linear(200, 100),
        nn.ELU(),
        nn.Dropout(p=drop),
        # (20)–(21)
        nn.Linear(100, 100),
        nn.ELU(),
        # (22)–(23)
        nn.Linear(100, 100),
        nn.ELU(),
        # (24) output
        nn.Linear(100, n_outputs),
    )


# ── Data loading ───────────────────────────────────────────────────────────────
def load_split(path: Path, feature_cols, target_cols, x_mean, x_std, y_mean, y_std):
    df = pd.read_csv(path, low_memory=False)
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    X = (X - x_mean) / x_std
    y = (y - y_mean) / y_std
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


# ── Training ───────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    n_samples = 0
    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(X_batch)
            n_samples += len(X_batch)
    return total_loss / n_samples


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train the covariance MLP on prepared dataset splits."
    )
    parser.add_argument("--train-csv", default="dataset-train.csv")
    parser.add_argument("--val-csv", default="dataset-val.csv")
    parser.add_argument("--test-csv", default="dataset-test.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="model-output",
        help="Directory to save model checkpoint and scalers (default: model-output)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early-stopping patience in epochs (default: 20; 0 disables)",
    )
    return parser


def main():
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[run] Device: {device}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Compute scalers from training set only ────────────────────────────────
    print(f"[run] Reading training CSV: {args.train_csv}", flush=True)
    train_df = pd.read_csv(args.train_csv, low_memory=False)
    feature_cols, target_cols = get_feature_target_columns(train_df)
    n_inputs = len(feature_cols)
    n_outputs = len(target_cols)
    print(f"[run] Features: {n_inputs}  Targets: {n_outputs}", flush=True)

    X_train_raw = train_df[feature_cols].values.astype(np.float32)
    y_train_raw = train_df[target_cols].values.astype(np.float32)

    x_mean = X_train_raw.mean(axis=0)
    x_std = X_train_raw.std(axis=0)
    x_std[x_std == 0] = 1.0  # avoid divide-by-zero for constant columns

    y_mean = y_train_raw.mean(axis=0)
    y_std = y_train_raw.std(axis=0)
    y_std[y_std == 0] = 1.0

    # Save input and output transformers separately
    input_transformers = {
        "x_mean": torch.from_numpy(x_mean),
        "x_std": torch.from_numpy(x_std),
        "feature_cols": feature_cols,
    }
    output_transformers = {
        "y_mean": torch.from_numpy(y_mean),
        "y_std": torch.from_numpy(y_std),
        "target_cols": target_cols,
    }
    torch.save(input_transformers, output_dir / "input_transformers.pt")
    torch.save(output_transformers, output_dir / "output_transformers.pt")
    print(f"[run] Input transformers saved to {output_dir}/input_transformers.pt", flush=True)
    print(f"[run] Output transformers saved to {output_dir}/output_transformers.pt", flush=True)

    # ── DataLoaders ────────────────────────────────────────────────────────────
    train_ds = load_split(
        Path(args.train_csv), feature_cols, target_cols,
        x_mean, x_std, y_mean, y_std,
    )
    val_ds = load_split(
        Path(args.val_csv), feature_cols, target_cols,
        x_mean, x_std, y_mean, y_std,
    )
    test_ds = load_split(
        Path(args.test_csv), feature_cols, target_cols,
        x_mean, x_std, y_mean, y_std,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # ── Model, loss, optimizer ─────────────────────────────────────────────────
    model = build_model(n_inputs, n_outputs).to(device)
    print(f"[run] Model architecture:\n{model}", flush=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"\n[run] Training for up to {args.epochs} epochs ...", flush=True)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:04d}/{args.epochs}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  t={elapsed:.1f}s",
            flush=True,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "model.pt")
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            print(
                f"[run] Early stopping triggered at epoch {epoch} "
                f"(no improvement for {args.patience} epochs)",
                flush=True,
            )
            break

    # ── Final evaluation on test set ───────────────────────────────────────────
    print("\n[run] Loading best checkpoint for test evaluation ...", flush=True)
    model.load_state_dict(torch.load(output_dir / "model.pt", weights_only=True))
    test_loss = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
    print(f"[run] Test MSE (normalized): {test_loss:.6f}", flush=True)

    # MAE in original (denormalized) units
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(device)).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y_batch.numpy())

    preds_raw = np.concatenate(all_preds) * y_std + y_mean
    targets_raw = np.concatenate(all_targets) * y_std + y_mean

    mae_per_target = np.abs(preds_raw - targets_raw).mean(axis=0)
    mae_overall = mae_per_target.mean()
    print(f"[run] Test MAE (original units, mean over targets): {mae_overall:.6f}", flush=True)
    print(f"[run] Test MAE per target:", flush=True)
    for col, mae in zip(target_cols, mae_per_target):
        print(f"       {col}: {mae:.6f}", flush=True)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    print(f"\n[run] Training history saved to {output_dir}/training_history.csv", flush=True)
    print(f"[run] Model saved to {output_dir}/model.pt", flush=True)


if __name__ == "__main__":
    main()
