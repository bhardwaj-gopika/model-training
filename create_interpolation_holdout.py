"""Create an interpolation holdout by sorting on one feature and removing a contiguous block."""

import argparse
import re
from pathlib import Path

import pandas as pd


def sanitize_name(value: str) -> str:
    """Make a filesystem-friendly name fragment."""
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create a secondary interpolation test set by sorting a CSV on one feature "
            "and removing a contiguous block from the middle of that sorted ordering."
        )
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV to split, usually dataset-train.csv",
    )
    parser.add_argument(
        "--sort-column",
        required=True,
        help="Feature column used to sort rows before extracting the holdout block",
    )
    parser.add_argument(
        "--holdout-size",
        type=int,
        default=30,
        help="Number of contiguous rows to exclude as the interpolation holdout (default: 30)",
    )
    parser.add_argument(
        "--center-fraction",
        type=float,
        default=0.5,
        help=(
            "Center position of the excluded block in sorted order, between 0 and 1 "
            "(default: 0.5 for the middle)"
        ),
    )
    parser.add_argument(
        "--expand-equal-boundaries",
        action="store_true",
        help=(
            "Expand the excluded block to include any rows with the same value as the "
            "block boundaries, ensuring a true gap in the sorted feature values"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output CSVs (default: same directory as input)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix for output files (default: derived from input name and sort column)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir) if args.output_dir else input_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] Reading: {input_csv}", flush=True)
    df = pd.read_csv(input_csv, low_memory=False)
    if args.sort_column not in df.columns:
        raise SystemExit(f"Column not found: {args.sort_column}")

    n_rows = len(df)
    if args.holdout_size <= 0 or args.holdout_size >= n_rows:
        raise SystemExit(
            f"holdout-size must be in [1, {n_rows - 1}], got {args.holdout_size}"
        )
    if not 0.0 <= args.center_fraction <= 1.0:
        raise SystemExit("center-fraction must be between 0 and 1")

    sorted_df = df.sort_values(args.sort_column, kind="mergesort").reset_index()
    center_index = int(round((n_rows - 1) * args.center_fraction))
    start = max(0, center_index - args.holdout_size // 2)
    end = start + args.holdout_size
    if end > n_rows:
        end = n_rows
        start = end - args.holdout_size

    if args.expand_equal_boundaries:
        start_value = sorted_df.iloc[start][args.sort_column]
        end_value = sorted_df.iloc[end - 1][args.sort_column]
        while start > 0 and sorted_df.iloc[start - 1][args.sort_column] == start_value:
            start -= 1
        while end < n_rows and sorted_df.iloc[end][args.sort_column] == end_value:
            end += 1

    holdout_df = sorted_df.iloc[start:end].copy()
    train_df = pd.concat([sorted_df.iloc[:start], sorted_df.iloc[end:]], axis=0).copy()

    holdout_min = holdout_df[args.sort_column].min()
    holdout_max = holdout_df[args.sort_column].max()
    gap_left = sorted_df.iloc[start - 1][args.sort_column] if start > 0 else None
    gap_right = sorted_df.iloc[end][args.sort_column] if end < n_rows else None

    prefix = args.prefix
    if prefix is None:
        prefix = f"{input_csv.stem}-{sanitize_name(args.sort_column)}-gap{args.holdout_size}"

    train_output = output_dir / f"{prefix}-train.csv"
    holdout_output = output_dir / f"{prefix}-secondary-test.csv"
    summary_output = output_dir / f"{prefix}-summary.csv"

    # Drop the temporary original index before writing training/eval CSVs.
    train_df = train_df.drop(columns=["index"]).reset_index(drop=True)
    holdout_original_index = holdout_df["index"].copy()
    holdout_df = holdout_df.drop(columns=["index"]).reset_index(drop=True)

    train_df.to_csv(train_output, index=False)
    holdout_df.to_csv(holdout_output, index=False)

    summary_df = pd.DataFrame(
        [
            {
                "input_csv": str(input_csv),
                "sort_column": args.sort_column,
                "holdout_size": args.holdout_size,
                "center_fraction": args.center_fraction,
                "sorted_start_index": start,
                "sorted_end_index_exclusive": end,
                "holdout_min": holdout_min,
                "holdout_max": holdout_max,
                "left_neighbor": gap_left,
                "right_neighbor": gap_right,
                "train_rows": len(train_df),
                "secondary_test_rows": len(holdout_df),
                "first_original_row_index": int(holdout_original_index.iloc[0]),
                "last_original_row_index": int(holdout_original_index.iloc[-1]),
            }
        ]
    )
    summary_df.to_csv(summary_output, index=False)

    print(f"[run] Sorted by: {args.sort_column}", flush=True)
    print(f"[run] Excluded sorted rows [{start}, {end}) as interpolation holdout", flush=True)
    print(
        f"[run] Holdout {args.sort_column} range: {holdout_min:.6e} to {holdout_max:.6e}",
        flush=True,
    )
    if gap_left is not None and gap_right is not None:
        print(
            f"[run] Neighboring retained values: left={gap_left:.6e}, right={gap_right:.6e}",
            flush=True,
        )
    print(f"[run] Saved train split: {len(train_df)} rows -> {train_output}", flush=True)
    print(f"[run] Saved secondary test split: {len(holdout_df)} rows -> {holdout_output}", flush=True)
    print(f"[run] Saved summary: {summary_output}", flush=True)


if __name__ == "__main__":
    main()
