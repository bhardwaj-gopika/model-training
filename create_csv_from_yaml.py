"""Standalone YAML-to-CSV exporter using analysis-loading.py utilities."""

import argparse
import importlib.util
from pathlib import Path


def load_analysis_loading_module(module_path):
    """Dynamically load analysis-loading.py as a Python module."""
    spec = importlib.util.spec_from_file_location("analysis_loading", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Create a CSV from dump.yaml using analysis-loading.py"
    )
    parser.add_argument("dump_yaml", help="Path to dump.yaml")
    parser.add_argument(
        "csv_output",
        nargs="?",
        default=None,
        help="Optional output CSV path (default: same name with .csv)",
    )
    parser.add_argument(
        "--embed-file-columns",
        nargs="+",
        default=None,
        help="Path columns to embed as base64 payload columns",
    )
    parser.add_argument(
        "--drop-source-path-columns",
        action="store_true",
        help="Drop original path columns after embedding base64 payload columns",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N rows while embedding files",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    module_path = Path(__file__).with_name("analysis-loading.py")
    analysis_loading = load_analysis_loading_module(module_path)

    csv_path, data_frame = analysis_loading.export_dump_yaml_to_csv(
        args.dump_yaml,
        args.csv_output,
        embed_file_columns=args.embed_file_columns,
        drop_source_path_columns=args.drop_source_path_columns,
        progress_every=args.progress_every,
    )

    print(f"Wrote {len(data_frame)} rows and {len(data_frame.columns)} columns to {csv_path}")


if __name__ == "__main__":
    main()
