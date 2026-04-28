"""Analyze covariance-output models using saved transformer dimensions."""

from pathlib import Path

import torch

import analyze_covariance as base_analysis
from train import build_model


def load_model_and_transformers(model_dir: Path):
    """Load a trained model using input/output dimensions from saved transformers."""
    input_tr = torch.load(model_dir / "input_transformers.pt", weights_only=False)
    output_tr = torch.load(model_dir / "output_transformers.pt", weights_only=False)
    cov_tr = torch.load(model_dir / "covariance_transformers.pt", weights_only=False)

    n_inputs = len(input_tr["feature_cols"])
    n_outputs = len(output_tr["target_cols"])
    y_mean = output_tr["y_mean"]
    y_std = output_tr["y_std"]

    model = build_model(n_inputs, n_outputs, y_mean=y_mean, y_std=y_std)
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=True))
    model.eval()
    return model, input_tr, output_tr, cov_tr


if __name__ == "__main__":
    base_analysis.load_model_and_transformers = load_model_and_transformers
    base_analysis.main()