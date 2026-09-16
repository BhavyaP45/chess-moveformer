import numpy as np
import pytest
import torch

from train_probes import (
    BatchedNonlinearProbes,
    _relative_labels,
    load_probe_checkpoint,
    predict_probe,
    train_probes,
)


def test_relative_labels_follow_player_to_move():
    labels = np.array(
        [
            [0, 1, 5, 7, 11],
            [0, 1, 5, 7, 11],
        ],
        dtype=np.int8,
    )
    plies = np.array([2, 3], dtype=np.int16)

    relative = _relative_labels(labels, plies)

    assert relative.tolist() == [
        [0, 1, 5, 7, 11],
        [0, 7, 11, 1, 5],
    ]


@pytest.fixture(scope="module")
def trained_probes(tmp_path_factory):
    rng = np.random.default_rng(42)
    n_games = 36
    positions_per_game = 6
    n_positions = n_games * positions_per_game
    n_layers = 2
    n_squares = 4
    n_features = 6

    latent = rng.normal(size=(n_positions, n_features)).astype(np.float32)
    activations = np.stack(
        [
            latent + rng.normal(scale=0.20, size=latent.shape),
            latent + rng.normal(scale=0.10, size=latent.shape),
        ],
        axis=1,
    ).astype(np.float16)
    labels = np.stack(
        [
            np.digitize(latent[:, square], bins=(-0.5, 0.5))
            for square in range(n_squares)
        ],
        axis=1,
    ).astype(np.int8)
    plies = np.tile([5, 15, 25, 35, 45, 55], n_games).astype(np.int16)
    game_ids = np.repeat(np.arange(n_games), positions_per_game).astype(np.int32)

    output_dir = tmp_path_factory.mktemp("probe_results")
    activation_path = output_dir / "activations.npz"
    np.savez_compressed(
        activation_path,
        activations=activations,
        labels=labels,
        plies=plies,
        game_ids=game_ids,
    )
    results = train_probes(
        activation_path=activation_path,
        output_dir=output_dir,
        device="cpu",
        epochs=10,
        batch_size=64,
        learning_rate=0.01,
    )
    return output_dir, results, game_ids, n_layers, n_squares, n_features


def test_primary_result_shapes(trained_probes):
    output_dir, _, _, n_layers, n_squares, _ = trained_probes
    assert np.load(output_dir / "probe_accuracies.npy").shape == (
        n_layers,
        2,
        n_squares,
    )
    assert np.load(output_dir / "probe_accuracies_per_class.npy").shape == (
        n_layers,
        2,
        n_squares,
        13,
    )
    assert np.load(output_dir / "probe_accuracies_per_ply.npy").shape == (
        n_layers,
        2,
        n_squares,
        6,
    )
    assert np.load(output_dir / "probe_baseline_accuracies.npy").shape == (
        n_layers,
        2,
        n_squares,
    )


def test_reported_accuracies_are_valid_probabilities(trained_probes):
    _, results, _, _, _, _ = trained_probes
    for values in (
        results["accuracies"],
        results["balanced_accuracies"],
        results["per_class"],
        results["per_ply"],
        results["baselines"],
    ):
        observed = values[~np.isnan(values)]
        assert np.all((observed >= 0) & (observed <= 1))


def test_split_keeps_games_independent(trained_probes):
    _, results, game_ids, _, _, _ = trained_probes
    train_games = set(game_ids[results["train_indices"]])
    test_games = set(game_ids[results["test_indices"]])
    assert train_games.isdisjoint(test_games)


def test_saved_probes_have_expected_count_and_can_predict(trained_probes):
    output_dir, _, _, n_layers, n_squares, n_features = trained_probes
    checkpoint_path = output_dir / "probe_weights.pt"
    checkpoint = load_probe_checkpoint(checkpoint_path)
    assert len(checkpoint["layers"]) * checkpoint["n_squares"] == n_layers * n_squares
    assert checkpoint["probe_architecture"] == "one_hidden_layer_mlp"
    assert checkpoint["layers"][0]["state_dict"]["banks.0.0.input.weight"].shape == (
        checkpoint["hidden_dim"],
        n_features,
    )
    assert checkpoint["layers"][0]["state_dict"]["banks.0.0.output.weight"].shape == (
        n_squares * 13,
        checkpoint["hidden_dim"],
    )
    assert checkpoint["target_encoding"] == "player_relative"
    assert checkpoint["turn_names"] == ("white", "black")

    prediction = predict_probe(
        checkpoint_path,
        np.zeros((1, n_features), dtype=np.float32),
        layer=0,
        square=0,
        player_to_move="white",
    )
    assert prediction.shape == (1,)


def test_nonlinear_probe_output_shape():
    model = BatchedNonlinearProbes(
        n_features=6,
        n_squares=4,
        hidden_dim=8,
    )
    output = model(torch.zeros(3, 6))
    assert output.shape == (3, 2, 2, 4, 13)


def test_support_counts_are_saved(trained_probes):
    output_dir, _, _, n_layers, n_squares, _ = trained_probes
    assert np.load(output_dir / "probe_support_per_class.npy").shape == (
        n_layers,
        2,
        n_squares,
        13,
    )
    assert np.load(output_dir / "probe_support_per_ply.npy").shape == (
        n_layers,
        2,
        n_squares,
        6,
    )
