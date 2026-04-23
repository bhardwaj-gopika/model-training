"""Reconstruct 6x6 covariance matrices from predicted Cholesky vectors."""

# Basic reconstruction
#./.venv/bin/python reconstruct_covariance.py model_predictions.csv

# With symmetry sanity check
#./.venv/bin/python reconstruct_covariance.py model_predictions.csv --verify-symmetry
#from reconstruct_covariance import reconstruct_covariance
#cov_6x6 = reconstruct_covariance(predicted_chol_vector)  # shape (21,) → (6, 6)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CHOL_COLS = [f"cov_chol_{i}" for i in range(21)]


def reconstruct_covariance(chol_vector):
    """Reconstruct a 6x6 covariance matrix from a 21-element Cholesky vector.

    The vector contains the lower-triangular elements of L where C = L @ L.T.
    This guarantees the result is always symmetric positive semi-definite.

    Parameters
    ----------
    chol_vector : array-like, shape (21,)
        Flattened lower-triangular Cholesky factors.

    Returns
    -------
    np.ndarray, shape (6, 6)
        Reconstructed covariance matrix.
    """
    L = np.zeros((6, 6))
    L[np.tril_indices(6)] = chol_vector
    return L @ L.T


def reconstruct_from_dataframe(df, chol_cols=None):
    """Apply reconstruct_covariance to every row of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the 21 Cholesky factor columns.
    chol_cols : list, optional
        Column names for the 21 Cholesky factors. Defaults to cov_chol_0..20.

    Returns
    -------
    list of np.ndarray
        One 6x6 covariance matrix per row. None for rows with missing values.
    """
    if chol_cols is None:
        chol_cols = CHOL_COLS

    matrices = []
    for _, row in df[chol_cols].iterrows():
        if row.isna().any():
            matrices.append(None)
        else:
            matrices.append(reconstruct_covariance(row.values))
    return matrices


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct 6x6 covariance matrices from predicted Cholesky "
            "vectors (cov_chol_0 .. cov_chol_20) in a CSV."
        )
    )
    parser.add_argument(
        "input_csv",
        help="CSV containing cov_chol_0 .. cov_chol_20 columns (model predictions)",
    )
    parser.add_argument(
        "--output-npy",
        default=None,
        help=(
            "Optional path to save all reconstructed matrices as a numpy .npy "
            "array of shape (N, 6, 6). Default: <input_stem>_cov_matrices.npy"
        ),
    )
    parser.add_argument(
        "--verify-symmetry",
        action="store_true",
        help="Print a check confirming reconstructed matrices are symmetric.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    input_csv = Path(args.input_csv)
    output_npy = Path(
        args.output_npy
        if args.output_npy
        else input_csv.with_name(f"{input_csv.stem}_cov_matrices.npy")
    )

    print(f"[run] Reading CSV: {input_csv}", flush=True)
    df = pd.read_csv(input_csv, low_memory=False)

    missing = [c for c in CHOL_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing Cholesky columns: {missing}")

    print(f"[run] Reconstructing covariance matrices for {len(df)} rows", flush=True)
    matrices = reconstruct_from_dataframe(df)

    valid = [(i, m) for i, m in enumerate(matrices) if m is not None]
    failed = len(matrices) - len(valid)
    print(f"[run] Reconstructed={len(valid)}, skipped_nulls={failed}", flush=True)

    if args.verify_symmetry and valid:
        _, sample = valid[0]
        is_sym = np.allclose(sample, sample.T)
        print(f"[verify] First matrix symmetric: {is_sym}")
        print(f"[verify] First matrix shape: {sample.shape}")
        print(sample)

    # Build (N, 6, 6) array; rows with missing values are filled with NaN
    all_matrices = np.full((len(df), 6, 6), np.nan)
    for i, mat in enumerate(matrices):
        if mat is not None:
            all_matrices[i] = mat

    np.save(output_npy, all_matrices)
    print(f"[run] Saved {all_matrices.shape} array to {output_npy}", flush=True)


if __name__ == "__main__":
    main()
