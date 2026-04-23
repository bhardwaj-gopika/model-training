"""Plot histograms of input parameters from training data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_feature_target_columns(df):
    """Extract feature and target column names based on prefix."""
    TARGET_PREFIX = "cov_chol_"
    target_cols = [c for c in df.columns if c.startswith(TARGET_PREFIX)]
    feature_cols = [c for c in df.columns if not c.startswith(TARGET_PREFIX)]
    return feature_cols, target_cols


def plot_input_histograms(csv_path: Path, output_dir: Path, n_cols: int = 6, n_rows: int = 7):
    """Plot histograms of all input parameters in a grid."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] Loading data from {csv_path}", flush=True)
    df = pd.read_csv(csv_path, low_memory=False)
    feature_cols, target_cols = get_feature_target_columns(df)
    print(f"[run] Found {len(feature_cols)} input features and {len(target_cols)} target features", flush=True)

    X = df[feature_cols].values.astype(np.float32)
    print(f"[run] Input shape: {X.shape}", flush=True)

    # Create grid of histograms
    n_features = len(feature_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 18))
    fig.suptitle("Distribution of Input Parameters (Training Set)", fontsize=16, y=0.995)

    axes_flat = axes.flatten()
    for k, (col, ax) in enumerate(zip(feature_cols, axes_flat)):
        data = X[:, k]
        # Compute percentiles for robust binning (avoid extreme outliers)
        p1, p99 = np.nanpercentile(data, [1, 99])
        n_bins = 40
        ax.hist(data, bins=n_bins, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, axis="y")
        # Add statistics
        mean_val = np.nanmean(data)
        std_val = np.nanstd(data)
        ax.text(
            0.98,
            0.97,
            f"μ={mean_val:.2e}\nσ={std_val:.2e}",
            transform=ax.transAxes,
            fontsize=6,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    # Hide unused subplots
    for ax in axes_flat[n_features:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "input_parameter_histograms.png", dpi=150, bbox_inches="tight")
    print(f"[run] Histograms saved to {output_dir}/input_parameter_histograms.png", flush=True)

    # Save summary statistics
    stats_data = []
    for col, k in zip(feature_cols, range(len(feature_cols))):
        data = X[:, k]
        stats_data.append(
            {
                "parameter": col,
                "mean": np.nanmean(data),
                "std": np.nanstd(data),
                "min": np.nanmin(data),
                "max": np.nanmax(data),
                "median": np.nanmedian(data),
                "q25": np.nanpercentile(data, 25),
                "q75": np.nanpercentile(data, 75),
            }
        )
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(output_dir / "input_parameter_stats.csv", index=False)
    print(f"[run] Statistics saved to {output_dir}/input_parameter_stats.csv", flush=True)
    print("\n[results] Input parameter statistics (training set):")
    print(stats_df.to_string(index=False))


def build_parser():
    parser = argparse.ArgumentParser(description="Plot histograms of input parameters from training data.")
    parser.add_argument(
        "--train-csv",
        default="dataset-train.csv",
        help="Training CSV path (default: dataset-train.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis-input-histograms",
        help="Output directory for histograms (default: analysis-input-histograms)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    plot_input_histograms(Path(args.train_csv), Path(args.output_dir))
    print(f"\n[run] Analysis complete. Results in {args.output_dir}/", flush=True)


if __name__ == "__main__":
    main()
