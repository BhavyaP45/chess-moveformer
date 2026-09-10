import numpy as np
import pytest
import torch

from pathlib import Path

from plots import (
    _balanced_accuracy_by_square_and_ply,
    _default_activation_path,
    _load_intervention_pairs,
    _primary_intervention_rows,
    compute_intervention_confidence_intervals,
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


EXAMPLE_HEADER = (
    "example_id,game_id,piece_class,piece,condition,scale,"
    "plan_retention,source_square_usage,legal\n"
)


def _write_example_csv(path, rows):
    path.write_text(EXAMPLE_HEADER + "".join(rows), encoding="utf-8")
    return path


def test_intervention_confidence_intervals_are_paired_deterministic_and_signed(
    tmp_path,
):
    example_path = _write_example_csv(
        tmp_path / "examples.csv",
        [
            "0,10,1,white pawn,treatment,1.0,False,False,False\n",
            "0,10,1,white pawn,control,1.0,True,True,True\n",
            "1,11,1,white pawn,treatment,1.0,True,True,True\n",
            "1,11,1,white pawn,control,1.0,True,True,True\n",
            "2,12,1,white pawn,treatment,1.0,False,False,True\n",
            "2,12,1,white pawn,control,1.0,True,True,True\n",
            "3,13,1,white pawn,treatment,1.0,True,False,True\n",
            "3,13,1,white pawn,control,1.0,True,True,True\n",
        ],
    )
    first_csv, first_npz = compute_intervention_confidence_intervals(
        example_path, tmp_path / "first", n_bootstrap=500, seed=42
    )
    second_csv, _ = compute_intervention_confidence_intervals(
        example_path, tmp_path / "second", n_bootstrap=500, seed=42
    )

    assert first_npz.exists()
    assert first_csv.read_text(encoding="utf-8") == second_csv.read_text(
        encoding="utf-8"
    )
    primary = _primary_intervention_rows(first_csv)
    source = primary["source_square_usage"]
    assert float(source["treatment_percentage"]) == 25.0
    assert float(source["control_percentage"]) == 100.0
    assert float(source["treatment_minus_control_pp"]) == -75.0
    assert float(source["ci_lower_pp"]) <= -75.0 <= float(
        source["ci_upper_pp"]
    )

    output_path = plot_causal_empty_intervention(
        first_csv, output_dir=tmp_path / "figures"
    )
    assert output_path.exists()
    assert output_path.name == "causal_empty_intervention_scale1_bars.pdf"
    assert output_path.with_suffix(".png").exists()


@pytest.mark.parametrize(
    "rows,error_text",
    [
        (
            ["0,10,1,white pawn,treatment,1.0,True,True,True\n"],
            "Missing control",
        ),
        (
            [
                "0,10,1,white pawn,treatment,1.0,True,True,True\n",
                "0,10,1,white pawn,treatment,1.0,True,True,True\n",
            ],
            "Duplicate treatment",
        ),
        (
            [
                "0,10,1,white pawn,treatment,1.0,True,True,True\n",
                "0,11,1,white pawn,control,1.0,True,True,True\n",
            ],
            "Mismatched game_id",
        ),
    ],
)
def test_intervention_pair_validation(tmp_path, rows, error_text):
    example_path = _write_example_csv(tmp_path / "invalid.csv", rows)
    with pytest.raises(ValueError, match=error_text):
        _load_intervention_pairs(example_path)


def test_scale_one_source_usage_recomputes_with_all_4000_positions():
    example_path = (
        Path(__file__).parents[1]
        / "results"
        / "causal_empty_intervention_examples_v4.csv"
    )
    records = [
        record
        for record in _load_intervention_pairs(example_path)
        if record["scale"] == 1.0
    ]

    assert len(records) == 4000
    treatment_count = sum(
        record["treatment"]["source_square_usage"] for record in records
    )
    control_count = sum(
        record["control"]["source_square_usage"] for record in records
    )
    assert treatment_count == 2535
    assert control_count == 2841
    assert 100.0 * treatment_count / len(records) == 63.375
    assert 100.0 * control_count / len(records) == 71.025
