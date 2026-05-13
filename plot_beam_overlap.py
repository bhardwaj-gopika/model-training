"""Plot overlapping beam distributions: actual particles (OpenPMD) vs sampled from predicted covariance.

For each sample, loads the true particle distribution from the .h5 file and
samples particles from the predicted 6x6 covariance matrix (via the trained
model). Plots 2D phase-space projections (x-px, y-py) side by side.

Usage:
    python plot_beam_overlap.py
    python plot_beam_overlap.py --particles-csv particles-241.csv --num-samples 5
    python plot_beam_overlap.py --sample-indices 0 10 42
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from beamphysics import ParticleGroup

from facet2_inj_ml_model import load_model as load_lume_model

# Phase-space ordering: x, px, y, py, t, pz, z
# First 6 match the model's covariance output; z is appended as index 6
PHASE_SPACE_LABELS = ["x", "px", "y", "py", "t", "pz", "z"]
PHASE_SPACE_UNITS = ["m", "eV/c", "m", "eV/c", "s", "eV/c", "m"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot overlapping beam distributions: true particles vs predicted covariance."
    )
    parser.add_argument(
        "--particles-csv",
        default="particles-571.csv",
        help="CSV with bmad_final_particles column containing .h5 paths (default: particles-571.csv)",
    )
    parser.add_argument(
        "--particles-column",
        default="bmad_final_particles",
        help="Column containing OpenPMD .h5 file paths (default: bmad_final_particles)",
    )
    parser.add_argument(
        "--input-space",
        default="sim",
        choices=["sim", "machine"],
        help="Model input space: 'sim' or 'machine' (default: sim)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of random samples to plot (default: 5)",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific row indices to plot (overrides --num-samples)",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=10000,
        help="Number of particles to sample from predicted covariance (default: 10000)",
    )
    parser.add_argument(
        "--output-dir",
        default="overlap-plots",
        help="Directory for output plots (default: overlap-plots)",
    )
    parser.add_argument(
        "--projections",
        nargs="+",
        default=["x-px", "y-py"],
        help="Phase-space projections to plot, e.g. x-px y-py (default: x-px y-py)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser


def parse_projection(proj_str):
    """Parse 'x-px' into index pair (0, 1)."""
    parts = proj_str.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid projection format: {proj_str}. Use e.g. 'x-px'")
    idx_a = PHASE_SPACE_LABELS.index(parts[0])
    idx_b = PHASE_SPACE_LABELS.index(parts[1])
    return idx_a, idx_b


def load_true_particles(h5_path: str):
    """Load particle beam from h5 and return 7D phase-space array (N, 7): x, px, y, py, t, pz, z."""
    beam = ParticleGroup(h5=h5_path)
    # Extract: x, px, y, py, t, pz, z
    particles = np.column_stack([
        beam.x, beam.px, beam.y, beam.py, beam.t, beam.pz, beam.z
    ])
    return particles


def sample_from_covariance(cov_matrix: np.ndarray, n_particles: int, rng: np.random.Generator):
    """Sample particles from a multivariate normal with the given 6x6 covariance.

    Returns (N, 7) array: the 6 covariance dimensions plus z=0 (z is not modeled).
    """
    mean = np.zeros(6)
    # Ensure symmetry and positive semi-definiteness
    cov_sym = (cov_matrix + cov_matrix.T) / 2
    try:
        particles_6d = rng.multivariate_normal(mean, cov_sym, size=n_particles)
    except np.linalg.LinAlgError:
        # Fall back: add small regularization
        cov_reg = cov_sym + np.eye(6) * 1e-30
        particles_6d = rng.multivariate_normal(mean, cov_reg, size=n_particles)
    # Append z=0 column (z is not part of the covariance model)
    z_col = np.zeros((n_particles, 1))
    return np.hstack([particles_6d, z_col])


# Input variable names expected by the sim model (must match YAML config order)
SIM_INPUT_VARIABLES = [
    "CQ10121:b1_gradient",
    "GUNF:rf_field_scale",
    "GUNF:theta0_deg",
    "SOL10111:solenoid_field_scale",
    "SQ10122:b1_gradient",
    "distgen:t_dist:sigma_t:value",
    "distgen:total_charge:value",
]


def plot_overlap(
    true_particles: np.ndarray,
    pred_particles: np.ndarray,
    projections: list,
    sample_label: str,
    output_path: Path,
):
    """Plot overlapping scatter of true vs predicted particles for given projections."""
    n_proj = len(projections)
    fig, axes = plt.subplots(1, n_proj, figsize=(6 * n_proj, 5))
    if n_proj == 1:
        axes = [axes]

    for ax, (idx_a, idx_b) in zip(axes, projections):
        # True particles (blue)
        ax.scatter(
            true_particles[:, idx_a],
            true_particles[:, idx_b],
            s=0.3, alpha=0.4, color="tab:blue", label="True (OpenPMD)", rasterized=True,
        )
        # Predicted particles from covariance (orange)
        ax.scatter(
            pred_particles[:, idx_a],
            pred_particles[:, idx_b],
            s=0.5, alpha=0.6, color="tab:orange", label="Predicted (Covariance)", rasterized=True,
        )
        ax.set_xlabel(f"{PHASE_SPACE_LABELS[idx_a]} [{PHASE_SPACE_UNITS[idx_a]}]")
        ax.set_ylabel(f"{PHASE_SPACE_LABELS[idx_b]} [{PHASE_SPACE_UNITS[idx_b]}]")
        ax.legend(fontsize=8, markerscale=5)

    fig.suptitle(sample_label, fontsize=11)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = build_parser().parse_args()

    import pandas as pd

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    particles_col = args.particles_column

    # Parse projections
    projections = [parse_projection(p) for p in args.projections]

    # Load CSV with particle paths and input parameters
    print(f"[run] Loading particles CSV: {args.particles_csv}", flush=True)
    df = pd.read_csv(args.particles_csv, low_memory=False)

    if particles_col not in df.columns:
        raise SystemExit(f"CSV must contain '{particles_col}' column with .h5 file paths")

    # Load the lume-torch model
    print(f"[run] Loading lume-torch model (input_space={args.input_space!r})", flush=True)
    model = load_lume_model(args.input_space)
    feature_cols = SIM_INPUT_VARIABLES
    print(f"[run] Model input features: {feature_cols}", flush=True)

    # Check that input columns exist in CSV
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing model input columns: {missing}")

    # Select samples
    valid_mask = df[particles_col].notna() & (df[particles_col].str.strip() != "")
    valid_indices = df.index[valid_mask].tolist()

    if args.sample_indices is not None:
        sample_indices = args.sample_indices
    else:
        n = min(args.num_samples, len(valid_indices))
        sample_indices = sorted(rng.choice(valid_indices, size=n, replace=False))

    print(f"[run] Plotting {len(sample_indices)} samples", flush=True)

    for i, idx in enumerate(sample_indices):
        row = df.iloc[idx]
        h5_path = str(row[particles_col]).strip()

        if not h5_path or h5_path == "nan":
            print(f"[skip] Row {idx}: no particle file", flush=True)
            continue

        print(f"[run] Sample {i+1}/{len(sample_indices)}: row {idx}, file={Path(h5_path).name}", flush=True)

        # Load true particles (no drift for screen 571)
        try:
            true_particles = load_true_particles(h5_path)
        except Exception as e:
            print(f"[skip] Row {idx}: failed to load particles: {e}", flush=True)
            continue

        # Build input dictionary for the lume-torch model
        input_dict = {col: float(row[col]) for col in feature_cols}
        result = model.evaluate(input_dict)

        # The model returns a dict with 'covariance_matrix' as a tensor
        # Output is already in physical units (M-denormalization is in the lume-torch output transformer)
        pred_cov_phys = result["covariance_matrix"].detach().cpu().numpy()

        # Sample particles from predicted covariance
        pred_particles = sample_from_covariance(pred_cov_phys, args.n_particles, rng)

        # Plot
        sample_label = f"Sample {idx} — True particles vs Predicted covariance"
        output_path = output_dir / f"overlap_sample_{idx:05d}.png"
        plot_overlap(true_particles, pred_particles, projections, sample_label, output_path)
        print(f"[run] Saved: {output_path}", flush=True)

    print(f"[run] Done. Plots saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
