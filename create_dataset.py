"""Create the ML training dataset by dropping non-feature/non-target columns."""
#./.venv/bin/python create_dataset.py dump-particles_241-cov-targets.csv dataset.csv --keep-columns emit_x_241 sigma_x_241
import argparse
from pathlib import Path

import pandas as pd


# Columns to drop from the dataset.
# These are metadata, diagnostic, path, and redundant output columns
# that should not be used as model inputs or targets.
COLUMNS_TO_DROP = [
    # Index
    "index",
    # Constants / calibration
    "Nmax",
    "Nmin",
    "PR10241:sample_frequency",
    "pxcal_241",
    "pxcal_571",
    "s",
    # Run metadata / bookkeeping
    "archiving_error",
    "command",
    "command_mpi",
    "impactT_ID",
    "mpi_run",
    "numprocs",          
    "stop_1:s",
    "xopt_error",
    "xopt_error_str",
    "xopt_runtime",
    "phase_offset",
    # bmad internals
    "bmad_bmad_init",
    "bmad_error",
    "bmad_final_particles",
    "bmad_idx",
    "bmad_tOffset",       
    # impact internals
    "impact_archive",
    "impact_bmad_init",
    "impact_error",
    "impact_final_particles",
    "impact_handoff_time_nominal",
    # distgen
    "distgen:n_particle", # 
    # header diagnostics
    "header:Flagdiag",
    "header:Nx",
    "header:Ny",
    "header:Nz",          
    # Image path columns (raw paths, not used as ML inputs)
    "image_241",
    "image_571",
    # Particle file paths
    "particles_241",
    # Particle fraction
    "particle_frac_241",  
    "particle_frac_571",
    # Emittance and sigma outputs (derivable from predicted covariance matrix)
    "emit_x_241",
    "emit_y_241",
    "sigma_x",
    "sigma_x45",
    "sigma_x_241",
    "sigma_x45_241",
    "sigma_y",
    "sigma_y45",
    "sigma_y_241",
    "sigma_y45_241",
    # 571 outputs not used in this model
    "emit_x_571",
    "emit_y_571",
    "n_particles_571",
    "scalarized_571",
    "scalarized_x_571",
    "scalarized_y_571",
    # 241 scalarized outputs (not primary targets)
    "scalarized_241",
    "scalarized_x_241",
    "scalarized_y_241",
    # Covariance processing status (metadata, not a model input/output)
    "cov_target_status",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create ML training dataset by removing non-feature/non-target columns."
    )
    parser.add_argument(
        "input_csv",
        help="Input CSV (e.g. dump-particles_241-cov-targets.csv from the cov targets script)",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="dataset.csv",
        help="Output dataset CSV path (default: dataset.csv)",
    )
    parser.add_argument(
        "--drop-null-rows",
        action="store_true",
        help="Drop any remaining rows with null values after column filtering",
    )
    parser.add_argument(
        "--keep-columns",
        nargs="+",
        default=None,
        help="Override: keep these columns even if they are in the drop list",
    )
    return parser


def main():
    args = build_parser().parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    print(f"[run] Reading: {input_csv}", flush=True)
    df = pd.read_csv(input_csv, low_memory=False)
    rows_in = len(df)
    cols_in = len(df.columns)
    print(f"[run] Loaded {rows_in} rows, {cols_in} columns", flush=True)

    drop_list = list(COLUMNS_TO_DROP)
    if args.keep_columns:
        drop_list = [c for c in drop_list if c not in args.keep_columns]
        print(f"[run] Keeping overridden columns: {args.keep_columns}", flush=True)

    # Only drop columns that actually exist; report the ones that don't
    present = [c for c in drop_list if c in df.columns]
    absent  = [c for c in drop_list if c not in df.columns]

    if absent:
        print(f"[run] Skipping {len(absent)} columns not found in CSV: {absent}", flush=True)

    df = df.drop(columns=present)
    print(f"[run] Dropped {len(present)} columns", flush=True)

    if args.drop_null_rows:
        rows_before_dropna = len(df)
        df = df.dropna(axis=0, how="any")
        dropped = rows_before_dropna - len(df)
        print(f"[run] Dropped {dropped} null rows ({len(df)} remaining)", flush=True)

    print(f"[run] Final dataset: {len(df)} rows, {len(df.columns)} columns", flush=True)
    print(f"[run] Remaining columns:\n  " + "\n  ".join(df.columns.tolist()), flush=True)

    df.to_csv(output_csv, index=False)
    print(f"[run] Saved to {output_csv}", flush=True)


if __name__ == "__main__":
    main()
