import numpy as np
import torch

from pathlib import Path

from plots import (
    _balanced_accuracy_by_square_and_ply,
    _default_activation_path,
    _primary_intervention_row,
    compute_probe_balanced_accuracy_by_ply,
    plot_causal_empty_intervention,
    plot_probe_balanced_accuracy_by_ply,
)
from train_probes import BatchedLinearProbes


def test_balanced_accuracy_by_square_and_ply_averages_class_recalls():
    labels = np.array([[0], [0], [1], [1]], dtype=np.int8)
    predictions = np.array([[0], [1], [1], [1]], dtype=np.int8)
    plies = np.array([5, 5, 5, 5], dtype=np.int16)

    balanced, recalls, support = _balanced_accuracy_by_square_and_ply(
        labels, predictions, plies
    )

    assert np.isclose(recalls[0, 0, 0], 0.5)
    assert np.isclose(recalls[0, 0, 1], 1.0)
    assert np.isclose(balanced[0, 0], 0.75)
    assert support[0, 0, 0] == 2


def test_default_activation_path_prefers_content_over_drive(monkeypatch):
    content_path = Path("/content/activations.npz")
    original_exists = Path.exists

    def fake_exists(self):
        if self == content_path:
            return True
        return original_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)
    assert _default_activation_path() == content_path


def test_compute_and_plot_probe_balanced_accuracy_by_ply(tmp_path):
    rng = np.random.default_rng(42)
    n_positions = 120
    n_layers = 2
    n_squares = 4
    n_features = 3
    activations = rng.normal(
        size=(n_positions, n_layers, n_features)
    ).astype(np.float16)
    labels = rng.integers(0, 3, size=(n_positions, n_squares), dtype=np.int8)
    plies = np.tile([5, 15, 25, 35, 45, 55], n_positions // 6).astype(np.int16)
    game_ids = np.repeat(np.arange(20), 6).astype(np.int32)
    activation_path = tmp_path / "activations.npz"
    np.savez_compressed(
        activation_path,
        activations=activations,
        labels=labels,
        plies=plies,
        game_ids=game_ids,
    )

    layers = []
    for _ in range(n_layers):
        model = BatchedLinearProbes(n_features, n_squares)
        layers.append(
            {
                "state_dict": model.state_dict(),
                "feature_mean": torch.zeros(n_features),
                "feature_std": torch.ones(n_features),
            }
        )
    checkpoint_path = tmp_path / "probe_weights.pt"
    torch.save(
        {
            "layers": layers,
            "n_layers": n_layers,
            "n_squares": n_squares,
            "n_features": n_features,
            "n_classes": 13,
        },
        checkpoint_path,
    )
    split_path = tmp_path / "probe_split.npz"
    np.savez(
        split_path,
        train_indices=np.arange(60),
        test_indices=np.arange(60, n_positions),
    )
    metrics_path = tmp_path / "probe_balanced_accuracy_by_ply.npz"

    result = compute_probe_balanced_accuracy_by_ply(
        activation_path=activation_path,
        checkpoint_path=checkpoint_path,
        split_path=split_path,
        output_path=metrics_path,
        device="cpu",
        batch_size=16,
    )
    with np.load(result) as metrics:
        assert metrics["balanced_accuracy_per_square"].shape == (
            n_layers,
            n_squares,
            6,
        )
        assert metrics["balanced_accuracy_mean"].shape == (n_layers, 6)
        assert metrics["recall_per_class"].shape == (
            n_layers,
            n_squares,
            6,
            13,
        )

    heatmap_path, line_path = plot_probe_balanced_accuracy_by_ply(
        result, output_dir=tmp_path / "figures"
    )
    assert heatmap_path.exists()
    assert line_path.exists()
    assert line_path.name == "probe_balanced_accuracy_layer6_by_ply.pdf"


def test_causal_empty_intervention_bar_plot_uses_scale_one_all_scope(tmp_path):
    summary_path = tmp_path / "causal_empty_intervention_summary_v4.csv"
    summary_path.write_text(
        "scope,piece_class,scale,"
        "treatment_plan_retention,control_plan_retention,"
        "treatment_source_square_usage,control_source_square_usage,"
        "treatment_forced_alternative,control_forced_alternative,"
        "treatment_legality,control_legality,"
        "specificity_plan_retention,specificity_source_square_usage,"
        "specificity_forced_alternative,treatment_minus_control_legality\n"
        "all,-1,0.5,72.1,78.3,76.7,82.2,23.3,17.8,95.7,97.3,6.2,5.5,5.5,-1.6\n"
        "all,-1,1.0,61.825,69.075,67.546,74.469,32.454,25.531,93.825,95.375,7.25,6.92,6.92,-1.55\n"
        "white pawn,1,1.0,70.8,76.8,73.4,79.0,26.6,21.0,96.8,97.8,6.0,5.6,5.6,-1.0\n",
        encoding="utf-8",
    )
    row = _primary_intervention_row(summary_path)
    assert float(row["treatment_plan_retention"]) == 61.825

    output_path = plot_causal_empty_intervention(
        summary_path, output_dir=tmp_path / "figures"
    )
    assert output_path.exists()
    assert output_path.name == "causal_empty_intervention_scale1_bars.pdf"
    assert output_path.with_suffix(".png").exists()
