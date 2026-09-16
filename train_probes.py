import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from project_utils import root_dir


RANDOM_STATE = 42
N_PIECE_CLASSES = 13
N_TURNS = 2
WHITE_TO_MOVE = 0
BLACK_TO_MOVE = 1
TURN_NAMES = ("white", "black")
RELATIVE_CLASS_NAMES = (
    "empty",
    "my pawn",
    "my knight",
    "my bishop",
    "my rook",
    "my queen",
    "my king",
    "opponent pawn",
    "opponent knight",
    "opponent bishop",
    "opponent rook",
    "opponent queen",
    "opponent king",
)
DEFAULT_HIDDEN_DIM = 256
DEFAULT_OUTPUT_SUBDIRECTORY = "nonlinear_probe_results"
VALIDATION_GAME_FRACTION = 0.1
TEST_GAME_FRACTION = 0.2
PLY_BUCKETS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, None))


class PlayerRelativeMLPProbe(nn.Module):
    """One nonlinear board-state probe for a fixed player-to-move condition."""

    def __init__(
        self,
        n_features,
        n_squares,
        hidden_dim,
        n_classes=N_PIECE_CLASSES,
    ):
        super().__init__()
        self.n_squares = n_squares
        self.n_classes = n_classes
        self.input = nn.Linear(n_features, hidden_dim)
        self.activation = nn.GELU()
        self.output = nn.Linear(hidden_dim, n_squares * n_classes)

    def forward(self, activations):
        logits = self.output(self.activation(self.input(activations)))
        return logits.reshape(-1, self.n_squares, self.n_classes)


class BatchedNonlinearProbes(nn.Module):
    """Two turn-specific MLP probes and their shuffled-label controls."""

    def __init__(
        self,
        n_features,
        n_squares,
        hidden_dim=DEFAULT_HIDDEN_DIM,
        n_classes=N_PIECE_CLASSES,
    ):
        super().__init__()
        self.n_squares = n_squares
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.banks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PlayerRelativeMLPProbe(
                            n_features,
                            n_squares,
                            hidden_dim,
                            n_classes,
                        )
                        for _branch in range(2)
                    ]
                )
                for _turn in range(N_TURNS)
            ]
        )

    def forward(self, activations):
        return torch.stack(
            [
                torch.stack(
                    [probe(activations) for probe in turn_bank],
                    dim=1,
                )
                for turn_bank in self.banks
            ],
            dim=1,
        )


def _turn_indices(plies):
    """Return 0 for White to move and 1 for Black to move after each ply."""
    return np.asarray(plies, dtype=np.int64) % 2


def _relative_labels(labels, plies):
    """Map absolute piece colours to mine/opponent for the player to move."""
    labels = np.asarray(labels)
    turns = _turn_indices(plies)
    relative = labels.copy()
    occupied = labels != 0
    white_pieces = (labels >= 1) & (labels <= 6)
    black_pieces = labels >= 7
    black_to_move = turns[:, None] == BLACK_TO_MOVE

    mine = occupied & ((white_pieces & ~black_to_move) | (black_pieces & black_to_move))
    opponent = occupied & ~mine
    piece_types = np.where(labels <= 6, labels, labels - 6)
    relative[mine] = piece_types[mine]
    relative[opponent] = piece_types[opponent] + 6
    return relative.astype(np.int8, copy=False)


def _shuffle_within_turns(labels, turns, rng):
    shuffled = np.empty_like(labels)
    for turn in range(N_TURNS):
        indices = np.flatnonzero(turns == turn)
        shuffled[indices] = labels[indices[rng.permutation(len(indices))]]
    return shuffled


def _select_turn_bank(logits, turns):
    rows = torch.arange(len(logits), device=logits.device)
    return logits[rows, turns]


def _validate_arrays(activations, labels, plies, game_ids):
    if activations.ndim != 3:
        raise ValueError(f"activations must have shape (N, layers, features), got {activations.shape}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape (N, squares), got {labels.shape}")

    n_positions, _, _ = activations.shape
    if labels.shape[0] != n_positions or plies.shape != (n_positions,) or game_ids.shape != (n_positions,):
        raise ValueError("activations, labels, plies, and game_ids must contain the same positions")
    if labels.size and (labels.min() < 0 or labels.max() >= N_PIECE_CLASSES):
        raise ValueError(f"labels must be integers from 0 to {N_PIECE_CLASSES - 1}")
    if np.unique(game_ids).size < 3:
        raise ValueError(
            "At least three games are required for game-level train, "
            "validation, and test splits"
        )


def _group_split(
    game_ids,
    validation_size=VALIDATION_GAME_FRACTION,
    test_size=TEST_GAME_FRACTION,
):
    unique_games = np.unique(game_ids)
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled_games = rng.permutation(unique_games)
    n_test_games = max(1, round(len(unique_games) * test_size))
    n_validation_games = max(1, round(len(unique_games) * validation_size))
    if n_test_games + n_validation_games >= len(unique_games):
        raise ValueError(
            "The validation and test fractions must leave at least one training game"
        )
    test_games = shuffled_games[:n_test_games]
    validation_games = shuffled_games[
        n_test_games : n_test_games + n_validation_games
    ]
    test_mask = np.isin(game_ids, test_games)
    validation_mask = np.isin(game_ids, validation_games)
    train_mask = ~(test_mask | validation_mask)
    return (
        np.flatnonzero(train_mask),
        np.flatnonzero(validation_mask),
        np.flatnonzero(test_mask),
    )


def _ply_mask(plies, lower, upper):
    return plies >= lower if upper is None else (plies >= lower) & (plies <= upper)


def _accuracy_by_class(y_true, y_pred):
    n_squares = y_true.shape[1]
    accuracies = np.full((n_squares, N_PIECE_CLASSES), np.nan, dtype=np.float32)
    supports = np.zeros((n_squares, N_PIECE_CLASSES), dtype=np.int32)

    for square in range(n_squares):
        for piece_class in range(N_PIECE_CLASSES):
            mask = y_true[:, square] == piece_class
            supports[square, piece_class] = mask.sum()
            if supports[square, piece_class]:
                accuracies[square, piece_class] = np.mean(
                    y_pred[mask, square] == y_true[mask, square]
                )

    return accuracies, supports


def _accuracy_by_ply(y_true, y_pred, test_plies):
    n_squares = y_true.shape[1]
    accuracies = np.full((n_squares, len(PLY_BUCKETS)), np.nan, dtype=np.float32)
    supports = np.zeros((n_squares, len(PLY_BUCKETS)), dtype=np.int32)

    for bucket_index, (lower, upper) in enumerate(PLY_BUCKETS):
        mask = _ply_mask(test_plies, lower, upper)
        supports[:, bucket_index] = mask.sum()
        if mask.any():
            accuracies[:, bucket_index] = np.mean(y_pred[mask] == y_true[mask], axis=0)

    return accuracies, supports


def _class_weights(labels, turns):
    n_positions, n_squares = labels.shape
    counts = torch.zeros(
        (N_TURNS, n_squares, N_PIECE_CLASSES),
        dtype=torch.float32,
        device=labels.device,
    )
    for turn in range(N_TURNS):
        turn_labels = labels[turns == turn]
        for piece_class in range(N_PIECE_CLASSES):
            counts[turn, :, piece_class] = (
                turn_labels == piece_class
            ).sum(dim=0)

    turn_sizes = torch.stack([(turns == turn).sum() for turn in range(N_TURNS)])
    present_classes = (counts > 0).sum(dim=2, keepdim=True).clamp_min(1)
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = (
        turn_sizes[:, None, None].expand_as(counts)[present]
        / (present_classes.expand_as(counts)[present] * counts[present])
    )
    return weights


def _weighted_loss_components(logits, targets, class_weights, turns):
    losses = F.cross_entropy(
        logits.reshape(-1, N_PIECE_CLASSES),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    expanded_weights = class_weights[turns, None].expand(
        -1,
        targets.shape[1],
        -1,
        -1,
    )
    sample_weights = expanded_weights.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    weighted_sums = (losses * sample_weights).sum(dim=(0, 2))
    weight_sums = sample_weights.sum(dim=(0, 2))
    return weighted_sums, weight_sums


def _evaluate_losses(
    model,
    activations,
    targets,
    turns,
    class_weights,
    batch_size,
    device,
    use_bf16,
):
    total_weighted_sums = torch.zeros(2, dtype=torch.float64, device=device)
    total_weight_sums = torch.zeros(2, dtype=torch.float64, device=device)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(activations), batch_size):
            batch = activations[start:start + batch_size]
            batch_targets = targets[start:start + batch_size]
            batch_turns = turns[start:start + batch_size]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = _select_turn_bank(model(batch), batch_turns)
                weighted_sums, weight_sums = _weighted_loss_components(
                    logits,
                    batch_targets,
                    class_weights,
                    batch_turns,
                )
            total_weighted_sums += weighted_sums.double()
            total_weight_sums += weight_sums.double()
    model.train()
    return (total_weighted_sums / total_weight_sums).cpu().numpy()


def _predict(model, activations, turns, batch_size, device, use_bf16):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(activations), batch_size):
            batch = activations[start:start + batch_size]
            batch_turns = turns[start:start + batch_size]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = _select_turn_bank(model(batch), batch_turns)
                predictions.append(logits.argmax(dim=-1).cpu())
    return torch.cat(predictions).numpy()


def _train_layer(
    x_train,
    train_labels,
    baseline_labels,
    train_turns,
    x_val,
    val_labels,
    val_turns,
    n_squares,
    n_features,
    device,
    epochs,
    batch_size,
    learning_rate,
    layer,
    hidden_dim,
):
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    storage_dtype = torch.bfloat16 if use_bf16 else torch.float32

    x_train = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    feature_mean = x_train.mean(dim=0)
    feature_std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_train = ((x_train - feature_mean) / feature_std).to(storage_dtype)
    train_labels = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    baseline_labels = torch.as_tensor(baseline_labels, dtype=torch.long, device=device)
    train_turns = torch.as_tensor(train_turns, dtype=torch.long, device=device)
    targets = torch.stack((train_labels, baseline_labels), dim=1)
    class_weights = _class_weights(train_labels, train_turns)

    x_val = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    x_val = ((x_val - feature_mean) / feature_std).to(storage_dtype)
    val_labels = torch.as_tensor(val_labels, dtype=torch.long, device=device)
    val_turns = torch.as_tensor(val_turns, dtype=torch.long, device=device)
    val_rng = np.random.default_rng(RANDOM_STATE + layer + 1)
    val_baseline_labels = torch.as_tensor(
        _shuffle_within_turns(
            val_labels.cpu().numpy(),
            val_turns.cpu().numpy(),
            val_rng,
        ),
        dtype=torch.long,
        device=device,
    )
    val_targets = torch.stack((val_labels, val_baseline_labels), dim=1)

    model = BatchedNonlinearProbes(
        n_features,
        n_squares,
        hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(RANDOM_STATE + layer)
    best_epoch = None
    best_real_val_loss = float("inf")
    best_val_losses = None
    best_state_dict = None

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(len(x_train), generator=generator, device=device)
        train_weighted_sums = torch.zeros(
            2,
            dtype=torch.float64,
            device=device,
        )
        train_weight_sums = torch.zeros(
            2,
            dtype=torch.float64,
            device=device,
        )

        for start in range(0, len(x_train), batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                batch_turns = train_turns[indices]
                logits = _select_turn_bank(model(x_train[indices]), batch_turns)
                weighted_sums, weight_sums = _weighted_loss_components(
                    logits,
                    targets[indices],
                    class_weights,
                    batch_turns,
                )
                loss = (weighted_sums / weight_sums).mean()
            loss.backward()
            optimizer.step()
            train_weighted_sums += weighted_sums.detach().double()
            train_weight_sums += weight_sums.detach().double()

        train_losses = (train_weighted_sums / train_weight_sums).cpu().numpy()
        val_losses = _evaluate_losses(
            model,
            x_val,
            val_targets,
            val_turns,
            class_weights,
            batch_size,
            device,
            use_bf16,
        )
        if val_losses[0] < best_real_val_loss:
            best_epoch = epoch + 1
            best_real_val_loss = float(val_losses[0])
            best_val_losses = val_losses.copy()
            best_state_dict = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        print(
            f"Layer {layer + 1}, epoch {epoch + 1}/{epochs} | "
            f"train real {train_losses[0]:.4f}, "
            f"shuffled {train_losses[1]:.4f} | "
            f"val real {val_losses[0]:.4f}, "
            f"shuffled {val_losses[1]:.4f} | "
            f"best real epoch {best_epoch}"
        )

    model.load_state_dict(best_state_dict)
    return (
        model,
        feature_mean,
        feature_std,
        best_state_dict,
        use_bf16,
        best_epoch,
        best_val_losses,
    )


def _prepare_test_activations(activations, feature_mean, feature_std, device, use_bf16):
    storage_dtype = torch.bfloat16 if use_bf16 else torch.float32
    activations = torch.as_tensor(activations, dtype=torch.float32, device=device)
    return ((activations - feature_mean) / feature_std).to(storage_dtype)


def _save_results(output_dir, results, checkpoint):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "probe_accuracies.npy", results["accuracies"])
    np.save(output_dir / "probe_balanced_accuracies.npy", results["balanced_accuracies"])
    np.save(output_dir / "probe_accuracies_per_class.npy", results["per_class"])
    np.save(output_dir / "probe_support_per_class.npy", results["class_support"])
    np.save(output_dir / "probe_accuracies_per_ply.npy", results["per_ply"])
    np.save(output_dir / "probe_support_per_ply.npy", results["ply_support"])
    np.save(output_dir / "probe_baseline_accuracies.npy", results["baselines"])
    np.savez(
        output_dir / "probe_split.npz",
        train_indices=results["train_indices"],
        validation_indices=results["validation_indices"],
        test_indices=results["test_indices"],
    )
    torch.save(checkpoint, output_dir / "probe_weights.pt")


def load_probe_checkpoint(checkpoint_path, map_location="cpu"):
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def predict_probe(
    checkpoint,
    activations,
    layer,
    square,
    player_to_move,
    device="cpu",
    baseline=False,
):
    if isinstance(checkpoint, (str, Path)):
        checkpoint = load_probe_checkpoint(checkpoint, map_location=device)

    device = torch.device(device)
    model = BatchedNonlinearProbes(
        checkpoint["n_features"],
        checkpoint["n_squares"],
        checkpoint["hidden_dim"],
        checkpoint["n_classes"],
    ).to(device)
    layer_state = checkpoint["layers"][layer]
    model.load_state_dict(layer_state["state_dict"])
    model.eval()

    inputs = torch.as_tensor(activations, dtype=torch.float32, device=device)
    feature_mean = layer_state["feature_mean"].to(device)
    feature_std = layer_state["feature_std"].to(device)
    inputs = (inputs - feature_mean) / feature_std
    if player_to_move not in TURN_NAMES:
        raise ValueError(
            f"player_to_move must be one of {TURN_NAMES}, got {player_to_move!r}"
        )
    turn = TURN_NAMES.index(player_to_move)
    branch = 1 if baseline else 0

    with torch.inference_mode():
        predictions = model(inputs)[:, turn, branch, square].argmax(dim=-1)
    return predictions.cpu().numpy()


def train_probes(
    activation_path=None,
    output_dir=None,
    device=None,
    epochs=50,
    batch_size=8192,
    learning_rate=1e-3,
    hidden_dim=DEFAULT_HIDDEN_DIM,
):
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs}")
    start_time = time.perf_counter()
    activation_path = Path(activation_path) if activation_path else root_dir() / "activations.npz"
    output_dir = (
        Path(output_dir)
        if output_dir
        else root_dir() / DEFAULT_OUTPUT_SUBDIRECTORY
    )
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_STATE)
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    with np.load(activation_path) as data:
        activations = data["activations"]
        labels = data["labels"]
        plies = data["plies"]
        game_ids = data["game_ids"]

    _validate_arrays(activations, labels, plies, game_ids)
    n_positions, n_layers, n_features = activations.shape
    n_squares = labels.shape[1]
    train_indices, validation_indices, test_indices = _group_split(game_ids)
    split_game_sets = (
        set(game_ids[train_indices]),
        set(game_ids[validation_indices]),
        set(game_ids[test_indices]),
    )
    if any(
        split_game_sets[left] & split_game_sets[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError(
            "Game-level split leaked a game across training, validation, or test"
        )

    rng = np.random.default_rng(RANDOM_STATE)
    relative_labels = _relative_labels(labels, plies)
    turns = _turn_indices(plies)
    train_labels = relative_labels[train_indices]
    train_turns = turns[train_indices]
    baseline_labels = _shuffle_within_turns(train_labels, train_turns, rng)
    validation_labels = relative_labels[validation_indices]
    validation_turns = turns[validation_indices]
    test_labels = relative_labels[test_indices]
    test_turns = turns[test_indices]
    test_plies = plies[test_indices]

    accuracies = np.full(
        (n_layers, N_TURNS, n_squares),
        np.nan,
        dtype=np.float32,
    )
    balanced_accuracies = np.full_like(accuracies, np.nan)
    per_class = np.full(
        (n_layers, N_TURNS, n_squares, N_PIECE_CLASSES),
        np.nan,
        dtype=np.float32,
    )
    class_support = np.zeros_like(per_class, dtype=np.int32)
    per_ply = np.full(
        (n_layers, N_TURNS, n_squares, len(PLY_BUCKETS)),
        np.nan,
        dtype=np.float32,
    )
    ply_support = np.zeros_like(per_ply, dtype=np.int32)
    baselines = np.full_like(accuracies, np.nan)
    layer_checkpoints = []

    print(
        f"Training probes on {device} with {len(train_indices):,} train, "
        f"{len(validation_indices):,} validation, and "
        f"{len(test_indices):,} test positions"
    )
    for layer in range(n_layers):
        x_train = np.asarray(activations[train_indices, layer], dtype=np.float32)
        x_val = np.asarray(
            activations[validation_indices, layer],
            dtype=np.float32,
        )
        (
            model,
            feature_mean,
            feature_std,
            state_dict,
            use_bf16,
            best_epoch,
            best_val_losses,
        ) = _train_layer(
            x_train,
            train_labels,
            baseline_labels,
            train_turns,
            x_val,
            validation_labels,
            validation_turns,
            n_squares,
            n_features,
            device,
            epochs,
            batch_size,
            learning_rate,
            layer,
            hidden_dim,
        )
        del x_train, x_val

        x_test = _prepare_test_activations(
            np.asarray(activations[test_indices, layer], dtype=np.float32),
            feature_mean,
            feature_std,
            device,
            use_bf16,
        )
        test_turn_tensor = torch.as_tensor(
            test_turns,
            dtype=torch.long,
            device=device,
        )
        predictions = _predict(
            model,
            x_test,
            test_turn_tensor,
            batch_size,
            device,
            use_bf16,
        )
        real_predictions = predictions[:, 0]
        baseline_predictions = predictions[:, 1]

        for turn in range(N_TURNS):
            turn_mask = test_turns == turn
            turn_labels = test_labels[turn_mask]
            turn_real = real_predictions[turn_mask]
            turn_baseline = baseline_predictions[turn_mask]
            turn_plies = test_plies[turn_mask]
            accuracies[layer, turn] = np.mean(
                turn_real == turn_labels,
                axis=0,
            )
            baselines[layer, turn] = np.mean(
                turn_baseline == turn_labels,
                axis=0,
            )
            (
                per_class[layer, turn],
                class_support[layer, turn],
            ) = _accuracy_by_class(turn_labels, turn_real)
            balanced_accuracies[layer, turn] = np.nanmean(
                per_class[layer, turn],
                axis=1,
            )
            (
                per_ply[layer, turn],
                ply_support[layer, turn],
            ) = _accuracy_by_ply(turn_labels, turn_real, turn_plies)
        layer_checkpoints.append(
            {
                "state_dict": state_dict,
                "feature_mean": feature_mean.cpu(),
                "feature_std": feature_std.cpu(),
                "best_epoch": best_epoch,
                "best_real_validation_loss": float(best_val_losses[0]),
                "shuffled_validation_loss_at_best_epoch": float(
                    best_val_losses[1]
                ),
            }
        )

        del model, x_test, predictions
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elapsed = time.perf_counter() - start_time
        print(f"Layer {layer + 1}, Square {n_squares} done — elapsed {elapsed:.1f}s")

    results = {
        "accuracies": accuracies,
        "balanced_accuracies": balanced_accuracies,
        "per_class": per_class,
        "class_support": class_support,
        "per_ply": per_ply,
        "ply_support": ply_support,
        "baselines": baselines,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
    }
    checkpoint = {
        "layers": layer_checkpoints,
        "n_layers": n_layers,
        "n_squares": n_squares,
        "n_features": n_features,
        "n_classes": N_PIECE_CLASSES,
        "probe_architecture": "one_hidden_layer_mlp",
        "hidden_dim": hidden_dim,
        "activation": "GELU",
        "n_turns": N_TURNS,
        "turn_names": TURN_NAMES,
        "target_encoding": "player_relative",
        "class_names": RELATIVE_CLASS_NAMES,
        "random_state": RANDOM_STATE,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "validation_game_fraction": VALIDATION_GAME_FRACTION,
        "test_game_fraction": TEST_GAME_FRACTION,
        "hyperparameters": {
            "optimizer": "Adam",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
        },
    }
    _save_results(output_dir, results, checkpoint)

    for layer, (mean_accuracy, mean_balanced) in enumerate(
        zip(
            np.nanmean(accuracies, axis=(1, 2)),
            np.nanmean(balanced_accuracies, axis=(1, 2)),
        ),
        start=1,
    ):
        print(
            f"Layer {layer} mean accuracy: {mean_accuracy:.4f} | "
            f"balanced accuracy: {mean_balanced:.4f}"
        )
    print(f"Mean baseline accuracy: {np.nanmean(baselines):.4f}")
    print(f"Mean balanced accuracy: {np.nanmean(balanced_accuracies):.4f}")

    ply_means = np.nanmean(per_ply, axis=(0, 1, 2))
    for (lower, upper), mean_accuracy in zip(PLY_BUCKETS, ply_means):
        label = f"{lower}+" if upper is None else f"{lower}-{upper}"
        print(f"Ply {label} mean accuracy: {mean_accuracy:.4f}")

    print(f"Total wall time: {time.perf_counter() - start_time:.1f}s")
    print(f"Saved probe results to: {output_dir}")
    return {**results, "checkpoint": checkpoint}


def main():
    parser = argparse.ArgumentParser(
        description="Train player-relative nonlinear probes on chess activations"
    )
    parser.add_argument("--activations", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default= 'cuda')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    args = parser.parse_args()
    train_probes(
        args.activations,
        args.output_dir,
        args.device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.hidden_dim,
    )


if __name__ == "__main__":
    main()
