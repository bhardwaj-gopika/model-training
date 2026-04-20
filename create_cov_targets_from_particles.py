"""Create Cholesky-flattened covariance targets from particles_241 OpenPMD files."""
#python create_cov_targets.py dump-particles_241-not-null.csv cov-targets.csv --progress-every 100 --drop-failed
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from beamphysics import ParticleGroup


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read particles_241 .h5 paths from CSV, compute covariance via "
            "ParticleGroup, apply Cholesky, flatten lower-triangular non-zero "
            "entries, and save as target columns."
        )
    )
    parser.add_argument("input_csv", help="Input CSV containing particles_241 column")
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="dump-particles_241-cov-targets.csv",
        help="Output CSV path (default: dump-particles_241-cov-targets.csv)",
    )
    parser.add_argument(
        "--particles-column",
        default="particles_241",
        help="Column containing OpenPMD .h5 file paths (default: particles_241)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N rows (default: 200)",
    )
    parser.add_argument(
        "--nonzero-tol",
        type=float,
        default=0.0,
        help=(
            "Treat abs(value) <= tol as zero when flattening lower-triangular "
            "Cholesky entries (default: 0.0)"
        ),
    )
    parser.add_argument(
        "--drop-failed",
        action="store_true",
        help="Drop rows where covariance/cholesky extraction failed",
    )
    return parser


def cholesky_nonzero_vector(cov, tol=0.0):
    cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        raise ValueError(f"Expected square covariance matrix, got shape={cov_arr.shape}")

    chol = np.linalg.cholesky(cov_arr)
    lower_triangle = chol[np.tril_indices(chol.shape[0])]

    # Only filter if an explicit nonzero tolerance is set by the user.
    # By default (tol=0.0) keep all 21 elements for a 6x6 matrix so
    # every row has the same fixed-length output vector for ML training.
    if tol > 0:
        return lower_triangle[np.abs(lower_triangle) > tol]
    return lower_triangle


def main():
    args = build_parser().parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    print(f"[run] Reading CSV: {input_csv}", flush=True)
    df = pd.read_csv(input_csv, low_memory=False)

    if args.particles_column not in df.columns:
        raise SystemExit(f"Column not found: {args.particles_column}")

    total_rows = len(df)
    vectors = []
    statuses = []
    expected_len = None

    for i, file_path in enumerate(df[args.particles_column], start=1):
        status = "ok"
        vec = None

        try:
            if pd.isna(file_path) or not str(file_path).strip():
                status = "missing_path"
            else:
                group = ParticleGroup(str(file_path))
                cov = group.cov('x', 'px', 'y', 'py', 'z', 'pz')
                vec = cholesky_nonzero_vector(cov, tol=args.nonzero_tol)

                if expected_len is None:
                    expected_len = len(vec)
                elif len(vec) != expected_len:
                    status = f"shape_mismatch:{len(vec)}"
                    vec = None

        except Exception as exc:
            status = f"error:{type(exc).__name__}"
            vec = None

        vectors.append(vec)
        statuses.append(status)

        if args.progress_every and i % args.progress_every == 0:
            ok_count = sum(s == "ok" for s in statuses)
            fail_count = len(statuses) - ok_count
            print(
                f"[run] {i}/{total_rows} processed (ok={ok_count}, failed={fail_count})",
                flush=True,
            )

    if expected_len is None:
        raise SystemExit("No valid covariance vectors were generated. Check file access.")

    target_col_names = [f"cov_chol_{k}" for k in range(expected_len)]
    target_matrix = np.full((len(df), expected_len), np.nan, dtype=float)

    for row_idx, vec in enumerate(vectors):
        if vec is not None and len(vec) == expected_len:
            target_matrix[row_idx, :] = vec

    targets_df = pd.DataFrame(target_matrix, columns=target_col_names)
    out_df = pd.concat([df, targets_df], axis=1)
    out_df["cov_target_status"] = statuses

    if args.drop_failed:
        out_df = out_df[out_df["cov_target_status"] == "ok"].copy()

    ok_total = int((out_df["cov_target_status"] == "ok").sum())
    fail_total = len(out_df) - ok_total

    print(
        (
            f"[run] Writing {output_csv} "
            f"(rows={len(out_df)}, target_dim={expected_len}, ok={ok_total}, failed={fail_total})"
        ),
        flush=True,
    )
    out_df.to_csv(output_csv, index=False)
    print("[run] Done", flush=True)


if __name__ == "__main__":
    main()
