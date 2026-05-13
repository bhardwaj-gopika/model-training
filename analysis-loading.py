"""Data loading utilities for analysis pipelines."""

import argparse
import base64
import csv
import os
import tarfile
import time
from pathlib import Path

import pandas as pd
import yaml

import logging

logger = logging.getLogger(__name__)


def load_yaml(file):
    """Load and return data from a YAML file."""
    with open(file, "r") as f:
        return yaml.safe_load(f)


def dump_data_to_dataframe(dump):
    """Convert a dump.yaml payload into a normalized DataFrame."""
    opt_data = pd.DataFrame(dump["data"])

    try:
        opt_data.index = opt_data.index.astype(int)
        opt_data = opt_data.sort_index()
    except (TypeError, ValueError):
        opt_data = opt_data.sort_index()

    # Some dump formats store objective and constraint values as
    # single-key dicts: {'col_name': float}. Unwrap them into scalars.
    for col in opt_data.columns:
        if opt_data[col].apply(lambda value: isinstance(value, dict)).any():
            opt_data[col] = opt_data[col].apply(
                lambda value: next(iter(value.values()))
                if isinstance(value, dict)
                else value
            )

    return opt_data


def _encode_file_to_base64(file_path, with_status=False):
    """Return base64 text for a file path, or None if unavailable."""
    if not isinstance(file_path, str) or not file_path:
        return (None, "invalid_path") if with_status else None

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return (None, "missing") if with_status else None

    try:
        with path.open("rb") as file_handle:
            encoded = base64.b64encode(file_handle.read()).decode("ascii")
            return (encoded, "ok") if with_status else encoded
    except OSError:
        return (None, "error") if with_status else None


def add_embedded_file_columns(
    data_frame,
    source_columns,
    suffix="base64",
    progress_every=500,
):
    """Add base64-encoded file payload columns derived from path columns."""
    total_rows = len(data_frame)

    for source_col in source_columns:
        if source_col not in data_frame.columns:
            print(f"[embed] Skipping missing column: {source_col}", flush=True)
            continue

        target_col = f"{source_col}_{suffix}"
        print(
            f"[embed] Starting {source_col} -> {target_col} ({total_rows} rows)",
            flush=True,
        )

        start_time = time.monotonic()
        encoded_values = []
        encoded_count = 0
        missing_count = 0
        error_count = 0

        for i, file_path in enumerate(data_frame[source_col], start=1):
            encoded, status = _encode_file_to_base64(file_path, with_status=True)
            encoded_values.append(encoded)

            if status == "ok":
                encoded_count += 1
            elif status == "missing" or status == "invalid_path":
                missing_count += 1
            else:
                error_count += 1

            if progress_every and i % progress_every == 0:
                elapsed = time.monotonic() - start_time
                print(
                    (
                        f"[embed] {source_col}: {i}/{total_rows} rows "
                        f"({elapsed:.1f}s, ok={encoded_count}, "
                        f"missing={missing_count}, error={error_count})"
                    ),
                    flush=True,
                )

        data_frame[target_col] = encoded_values

        elapsed = time.monotonic() - start_time
        print(
            (
                f"[embed] Finished {source_col} in {elapsed:.1f}s "
                f"(ok={encoded_count}, missing={missing_count}, error={error_count})"
            ),
            flush=True,
        )

    return data_frame


def export_dump_yaml_to_csv(
    dump_yaml_path,
    csv_path=None,
    embed_file_columns=None,
    drop_source_path_columns=False,
    progress_every=500,
):
    """Load a dump.yaml file and write its data table to CSV."""
    dump_yaml_path = Path(dump_yaml_path)
    dump = load_yaml(dump_yaml_path)
    data_frame = dump_data_to_dataframe(dump)

    if embed_file_columns:
        data_frame = add_embedded_file_columns(
            data_frame,
            embed_file_columns,
            progress_every=progress_every,
        )
        if drop_source_path_columns:
            cols_to_drop = [col for col in embed_file_columns if col in data_frame.columns]
            data_frame = data_frame.drop(columns=cols_to_drop)

    if csv_path is None:
        csv_path = dump_yaml_path.with_suffix(".csv")
    csv_path = Path(csv_path)
    data_frame.to_csv(csv_path, index_label="index")

    return csv_path, data_frame


def embed_file_columns_in_csv(
    csv_input_path,
    csv_output_path=None,
    embed_file_columns=None,
    drop_source_path_columns=False,
    progress_every=500,
):
    """Load an existing CSV and embed file-path columns as base64 columns."""
    csv_input_path = Path(csv_input_path)
    if csv_output_path is None:
        csv_output_path = csv_input_path.with_name(
            f"{csv_input_path.stem}-embedded{csv_input_path.suffix}"
        )
    csv_output_path = Path(csv_output_path)

    print(f"[run] Reading CSV: {csv_input_path}", flush=True)
    data_frame = pd.read_csv(csv_input_path, low_memory=False)
    print(
        (
            f"[run] CSV loaded ({len(data_frame)} rows, "
            f"{len(data_frame.columns)} columns)"
        ),
        flush=True,
    )

    if embed_file_columns:
        data_frame = add_embedded_file_columns(
            data_frame,
            embed_file_columns,
            progress_every=progress_every,
        )
        if drop_source_path_columns:
            cols_to_drop = [col for col in embed_file_columns if col in data_frame.columns]
            data_frame = data_frame.drop(columns=cols_to_drop)

    data_frame.to_csv(csv_output_path, index=False)
    return csv_output_path, data_frame


def embed_file_columns_in_csv_streaming(
    csv_input_path,
    csv_output_path=None,
    embed_file_columns=None,
    drop_source_path_columns=False,
    progress_every=500,
):
    """Embed file-path columns as base64 in CSV using row-by-row streaming."""
    csv_input_path = Path(csv_input_path)
    if csv_output_path is None:
        csv_output_path = csv_input_path.with_name(
            f"{csv_input_path.stem}-embedded{csv_input_path.suffix}"
        )
    csv_output_path = Path(csv_output_path)

    if not embed_file_columns:
        raise ValueError("Streaming CSV mode requires --embed-file-columns.")

    print(f"[run] Streaming CSV read: {csv_input_path}", flush=True)
    start_time = time.monotonic()

    encoded_suffix = "base64"
    source_cols = list(embed_file_columns)
    target_cols = [f"{col}_{encoded_suffix}" for col in source_cols]

    with csv_input_path.open("r", newline="") as in_f:
        reader = csv.DictReader(in_f)
        input_fields = reader.fieldnames or []

        present_source_cols = [col for col in source_cols if col in input_fields]
        missing_source_cols = [col for col in source_cols if col not in input_fields]
        for col in missing_source_cols:
            print(f"[embed] Skipping missing column: {col}", flush=True)

        if drop_source_path_columns:
            output_fields = [f for f in input_fields if f not in present_source_cols]
        else:
            output_fields = list(input_fields)

        for col in present_source_cols:
            output_fields.append(f"{col}_{encoded_suffix}")

        with csv_output_path.open("w", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=output_fields)
            writer.writeheader()

            row_count = 0
            counts = {
                col: {"ok": 0, "missing": 0, "error": 0, "invalid_path": 0}
                for col in present_source_cols
            }

            for row in reader:
                row_count += 1

                for col in present_source_cols:
                    encoded, status = _encode_file_to_base64(row.get(col), with_status=True)
                    row[f"{col}_{encoded_suffix}"] = encoded
                    counts[col][status] += 1

                    if drop_source_path_columns and col in row:
                        del row[col]

                writer.writerow({field: row.get(field) for field in output_fields})

                if progress_every and row_count % progress_every == 0:
                    elapsed = time.monotonic() - start_time
                    summary = ", ".join(
                        (
                            f"{col}: ok={counts[col]['ok']} "
                            f"missing={counts[col]['missing'] + counts[col]['invalid_path']} "
                            f"error={counts[col]['error']}"
                        )
                        for col in present_source_cols
                    )
                    print(
                        f"[embed] rows={row_count} elapsed={elapsed:.1f}s | {summary}",
                        flush=True,
                    )

    elapsed = time.monotonic() - start_time
    print(f"[run] Streaming CSV write complete in {elapsed:.1f}s", flush=True)

    for col in present_source_cols:
        missing_total = counts[col]["missing"] + counts[col]["invalid_path"]
        print(
            (
                f"[embed] Finished {col} "
                f"(ok={counts[col]['ok']}, missing={missing_total}, error={counts[col]['error']})"
            ),
            flush=True,
        )

    return csv_output_path, None


def load_scan_data(scan_path):
    """Load input_file.yaml and dump.yaml from a scan results directory.

    Parameters
    ----------
    scan_path : str or Path
        Path to the scan results directory (contains input_file.yaml and
        dump.yaml).

    Returns
    -------
    dict
        Keys: ``'inputs'``, ``'dump'``, ``'data_sort'`` (DataFrame sorted by
        index), ``'vocs'``, ``'dir_path'`` (parent dir from dump_file).
    """
    scan_path = Path(scan_path)
    inputs = load_yaml(scan_path / "input_file.yaml")
    dump = load_yaml(scan_path / "dump.yaml")
    data_sort = dump_data_to_dataframe(dump)

    dir_path = Path(dump["dump_file"]).parent

    return {
        "inputs": inputs,
        "dump": dump,
        "data_sort": data_sort,
        "vocs": dump["vocs"],
        "dir_path": dir_path,
    }


def load_cal_data(inputs):
    """Load calibration data YAML files referenced in the input config."""
    locs = [ds["cal_data_loc"] for ds in inputs["datasets"]]
    return [load_yaml(loc) for loc in locs]


def get_per_run_tars(run_path):
    """Return sorted list of per_run_*.tar files in run_path."""
    tars = []
    if not os.path.isdir(run_path):
        return tars
    for fname in os.listdir(run_path):
        if fname.startswith("per_run_") and fname.endswith(".tar"):
            tars.append(os.path.join(run_path, fname))
    return sorted(tars)


def auto_detect_cleanup(run_path):
    """Return True if run_path contains per_run tar archives."""
    return len(get_per_run_tars(run_path)) > 0


def ensure_figures_extracted(run_path, shot_indices, _extracted_cache=None):
    """Extract Figures/ and Sim_Images/ from per-run tar archives.

    Returns True if any extraction was performed.
    """
    if _extracted_cache is None:
        _extracted_cache = set()

    if not auto_detect_cleanup(run_path):
        return False

    fig_dir = os.path.join(run_path, "Figures")
    needed = set()
    for idx in shot_indices:
        if idx is not None and idx >= 0:
            fig_path = os.path.join(fig_dir, f"{int(idx)}_fig.png")
            if not os.path.exists(fig_path):
                needed.add(int(idx))

    if not needed:
        return False

    extracted = False
    for tar_path in get_per_run_tars(run_path):
        if tar_path in _extracted_cache:
            continue
        try:
            with tarfile.open(tar_path, "r") as tar:
                members = [
                    m
                    for m in tar.getmembers()
                    if m.name.startswith("Figures/")
                    or m.name.startswith("Sim_Images/")
                ]
                if members:
                    tar.extractall(path=run_path, members=members)
                    extracted = True
            _extracted_cache.add(tar_path)
        except Exception as exc:
            logger.warning("Could not extract %s: %s", tar_path, exc)

    return extracted


def detect_cleanup_from_dump(dump):
    """Sample the first filepath in dump data to detect tar archives."""
    fps = pd.DataFrame(dump["data"]).get("filepath", pd.Series(dtype=object)).dropna()
    if len(fps) == 0:
        return False
    sample = fps.iloc[0]
    if isinstance(sample, list):
        sample = sample[0]
    return auto_detect_cleanup(sample)


def auto_add_module_path(modname: str) -> bool:
    """Try to import *modname*; if it fails, look for it as a sibling project.

    Searches for ``<repo_parent>/<modname>/src`` where ``<repo_parent>`` is the
    parent directory of the Model_Calibration project root.  Adds the first
    matching directory to ``sys.path``.

    Returns True if the path was added, False otherwise.
    """
    import sys
    import importlib
    from pathlib import Path

    # First make sure compat shims are registered (covers Helper_Functions_v3 etc.)
    try:
        import Model_Calibration.compat as _mc_compat
        _mc_compat.register()
    except Exception:
        pass

    # Already importable (including via shim)?
    try:
        importlib.import_module(modname)
        return False
    except ImportError:
        pass

    try:
        import Model_Calibration as _mc
        # src/Model_Calibration/__init__.py  →  repo_root/src/Model_Calibration
        # .parent twice → repo_root/src  →  .parent → repo_root
        mc_repo = Path(_mc.__file__).parent.parent.parent
        repo_parent = mc_repo.parent
        # Try <sibling>/src first (src-layout projects), then <sibling> directly
        for candidate in (repo_parent / modname / "src", repo_parent / modname):
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
                return True
    except Exception:
        pass
    return False


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Export dump.yaml to CSV, or embed file columns in an existing CSV."
    )
    parser.add_argument(
        "input_path",
        help="Path to input file (.yaml/.yml for dump export, .csv for embedding)",
    )
    parser.add_argument(
        "csv_output",
        nargs="?",
        help="Optional CSV output path. Defaults to dump.csv beside dump.yaml.",
    )
    parser.add_argument(
        "--embed-file-columns",
        nargs="+",
        default=None,
        help=(
            "Path columns to embed as base64 payload columns. "
            "Example: --embed-file-columns image_241 image_571"
        ),
    )
    parser.add_argument(
        "--drop-source-path-columns",
        action="store_true",
        help="Drop original path columns after embedding base64 payload columns.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N rows while embedding files (default: 500).",
    )
    parser.add_argument(
        "--stream-csv",
        action="store_true",
        help="Use low-memory streaming mode for CSV inputs.",
    )
    return parser


def main():
    args = _build_arg_parser().parse_args()
    input_path = Path(args.input_path)
    print(f"[run] Loading {input_path}", flush=True)

    embed_cols = args.embed_file_columns
    if embed_cols:
        print(
            (
                f"[run] Embedding columns: {', '.join(embed_cols)} "
                f"(progress every {args.progress_every} rows)"
            ),
            flush=True,
        )

    if input_path.suffix.lower() in {".yaml", ".yml"}:
        csv_path, data_frame = export_dump_yaml_to_csv(
            input_path,
            args.csv_output,
            embed_file_columns=embed_cols,
            drop_source_path_columns=args.drop_source_path_columns,
            progress_every=args.progress_every,
        )
    elif input_path.suffix.lower() == ".csv":
        if args.stream_csv:
            csv_path, data_frame = embed_file_columns_in_csv_streaming(
                input_path,
                args.csv_output,
                embed_file_columns=embed_cols,
                drop_source_path_columns=args.drop_source_path_columns,
                progress_every=args.progress_every,
            )
        else:
            csv_path, data_frame = embed_file_columns_in_csv(
                input_path,
                args.csv_output,
                embed_file_columns=embed_cols,
                drop_source_path_columns=args.drop_source_path_columns,
                progress_every=args.progress_every,
            )
    else:
        raise ValueError(
            "Unsupported input type. Use .yaml/.yml for dump export or .csv for CSV embedding."
        )

    if data_frame is None:
        print(f"Wrote output CSV to {csv_path}")
    else:
        print(f"Wrote {len(data_frame)} rows and {len(data_frame.columns)} columns to {csv_path}")


if __name__ == "__main__":
    main()