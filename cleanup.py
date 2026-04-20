"""Filter a CSV to keep only rows with a non-null value in a chosen column."""

import argparse
from pathlib import Path

import pandas as pd


def build_parser():
    parser = argparse.ArgumentParser(
        description="Keep only rows where a specified column is not null."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        default="dump.csv",
        help="Input CSV path (default: dump.csv)",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="dump-particles_241-not-null.csv",
        help="Output CSV path (default: dump-particles_241-not-null.csv)",
    )
    parser.add_argument(
        "--column",
        default="particles_241",
        help="Column that must be non-null to keep a row (default: particles_241)",
    )
    return parser


def main():
    args = build_parser().parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    key_col = args.column

    print(f"[run] Reading: {input_csv}", flush=True)
    df = pd.read_csv(input_csv, low_memory=False)

    if key_col not in df.columns:
        print(f"ERROR: column '{key_col}' not found", flush=True)
        similar = [c for c in df.columns if "particle" in c.lower() or "241" in c]
        print(f"possible_related_columns={similar}", flush=True)
        raise SystemExit(1)

    rows_before = len(df)
    clean_df = df[df[key_col].notna()].copy()
    rows_after = len(clean_df)
    rows_removed = rows_before - rows_after

    clean_df.to_csv(output_csv, index=False)

    print(f"input={input_csv}", flush=True)
    print(f"output={output_csv}", flush=True)
    print(f"filter_column={key_col}", flush=True)
    print(f"rows_before={rows_before}", flush=True)
    print(f"rows_after={rows_after}", flush=True)
    print(f"rows_removed={rows_removed}", flush=True)


if __name__ == "__main__":
    main()