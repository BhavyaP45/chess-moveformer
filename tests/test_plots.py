import numpy as np
import torch

from pathlib import Path

from plots import (
    _balanced_accuracy_by_square_and_ply,
    _default_activation_path,
    compute_probe_balanced_accuracy_by_ply,
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
