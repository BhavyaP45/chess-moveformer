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
PLY_BUCKETS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, None))


class BatchedLinearProbes(nn.Module):
    """Independent real and shuffled-label linear probes for every square."""

    def __init__(self, n_features, n_squares, n_classes=N_PIECE_CLASSES):
        super().__init__()
        self.n_squares = n_squares
        self.n_classes = n_classes
        self.linear = nn.Linear(n_features, 2 * n_squares * n_classes)

    def forward(self, activations):
        logits = self.linear(activations)
        return logits.reshape(-1, 2, self.n_squares, self.n_classes)


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
    if np.unique(game_ids).size < 2:
        raise ValueError("At least two games are required for a game-level train/test split")


def _group_split(game_ids, test_size=0.2):
    unique_games = np.unique(game_ids)
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled_games = rng.permutation(unique_games)
    n_test_games = min(max(1, round(len(unique_games) * test_size)), len(unique_games) - 1)
    test_games = shuffled_games[:n_test_games]
    test_mask = np.isin(game_ids, test_games)
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


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


def _class_weights(labels):
    n_positions, n_squares = labels.shape
    counts = torch.zeros(
        (n_squares, N_PIECE_CLASSES),
        dtype=torch.float32,
        device=labels.device,
    )
    for piece_class in range(N_PIECE_CLASSES):
        counts[:, piece_class] = (labels == piece_class).sum(dim=0)

    present_classes = (counts > 0).sum(dim=1, keepdim=True).clamp_min(1)
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = (
        n_positions
        / (present_classes.expand_as(counts)[present] * counts[present])
    )
    return weights


def _weighted_loss(logits, targets, class_weights):
    losses = F.cross_entropy(
        logits.reshape(-1, N_PIECE_CLASSES),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    expanded_weights = class_weights[None, None].expand(
        targets.shape[0],
        targets.shape[1],
        -1,
        -1,
    )
    sample_weights = expanded_weights.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (losses * sample_weights).sum() / sample_weights.sum()


def _predict(model, activations, batch_size, device, use_bf16):
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(activations), batch_size):
            batch = activations[start:start + batch_size]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                predictions.append(model(batch).argmax(dim=-1).cpu())
    return torch.cat(predictions).numpy()


def _train_layer(x_train, train_labels, baseline_labels, n_squares, n_features, device, epochs, batch_size, learning_rate, layer):
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    storage_dtype = torch.bfloat16 if use_bf16 else torch.float32

    x_train = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    feature_mean = x_train.mean(dim=0)
    feature_std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_train = ((x_train - feature_mean) / feature_std).to(storage_dtype)
    train_labels = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    baseline_labels = torch.as_tensor(baseline_labels, dtype=torch.long, device=device)
    targets = torch.stack((train_labels, baseline_labels), dim=1)
    class_weights = _class_weights(train_labels)

    model = BatchedLinearProbes(n_features, n_squares).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(RANDOM_STATE + layer)

    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(len(x_train), generator=generator, device=device)
        total_loss = 0.0

        for start in range(0, len(x_train), batch_size):
            indices = permutation[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(x_train[indices])
                loss = _weighted_loss(logits, targets[indices], class_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            mean_loss = total_loss / len(x_train)
            print(f"Layer {layer + 1}, epoch {epoch + 1}/{epochs}, loss {mean_loss:.4f}")

    state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    return model, feature_mean, feature_std, state_dict, use_bf16


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
        test_indices=results["test_indices"],
    )
    torch.save(checkpoint, output_dir / "probe_weights.pt")


def load_probe_checkpoint(checkpoint_path, map_location="cpu"):
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def predict_probe(checkpoint, activations, layer, square, device="cpu", baseline=False):
    if isinstance(checkpoint, (str, Path)):
        checkpoint = load_probe_checkpoint(checkpoint, map_location=device)

    device = torch.device(device)
    model = BatchedLinearProbes(
        checkpoint["n_features"],
        checkpoint["n_squares"],
        checkpoint["n_classes"],
    ).to(device)
    layer_state = checkpoint["layers"][layer]
    model.load_state_dict(layer_state["state_dict"])
    model.eval()

    inputs = torch.as_tensor(activations, dtype=torch.float32, device=device)
    feature_mean = layer_state["feature_mean"].to(device)
    feature_std = layer_state["feature_std"].to(device)
    inputs = (inputs - feature_mean) / feature_std
    branch = 1 if baseline else 0

    with torch.inference_mode():
        predictions = model(inputs)[:, branch, square].argmax(dim=-1)
    return predictions.cpu().numpy()


def train_probes(activation_path=None, output_dir=None, device=None, epochs=50, batch_size=8192, learning_rate=1e-3):
    start_time = time.perf_counter()
    activation_path = Path(activation_path) if activation_path else root_dir() / "activations.npz"
    output_dir = Path(output_dir) if output_dir else root_dir()
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
    train_indices, test_indices = _group_split(game_ids)
    if np.intersect1d(game_ids[train_indices], game_ids[test_indices]).size:
        raise RuntimeError("Game-level split leaked a game between training and testing")

    rng = np.random.default_rng(RANDOM_STATE)
    shuffled_order = rng.permutation(len(train_indices))
    train_labels = labels[train_indices]
    baseline_labels = train_labels[shuffled_order]
    test_labels = labels[test_indices]
    test_plies = plies[test_indices]

    accuracies = np.full((n_layers, n_squares), np.nan, dtype=np.float32)
    balanced_accuracies = np.full_like(accuracies, np.nan)
    per_class = np.full((n_layers, n_squares, N_PIECE_CLASSES), np.nan, dtype=np.float32)
    class_support = np.zeros((n_layers, n_squares, N_PIECE_CLASSES), dtype=np.int32)
    per_ply = np.full((n_layers, n_squares, len(PLY_BUCKETS)), np.nan, dtype=np.float32)
    ply_support = np.zeros((n_layers, n_squares, len(PLY_BUCKETS)), dtype=np.int32)
    baselines = np.full_like(accuracies, np.nan)
    layer_checkpoints = []

    print(f"Training probes on {device} with {len(train_indices):,} train positions")
    for layer in range(n_layers):
        x_train = np.asarray(activations[train_indices, layer], dtype=np.float32)
        model, feature_mean, feature_std, state_dict, use_bf16 = _train_layer(
            x_train,
            train_labels,
            baseline_labels,
            n_squares,
            n_features,
            device,
            epochs,
            batch_size,
            learning_rate,
            layer,
        )
        del x_train

        x_test = np.asarray(activations[test_indices, layer], dtype=np.float32)
        x_test = _prepare_test_activations(
            x_test,
            feature_mean,
            feature_std,
            device,
            use_bf16,
        )
        predictions = _predict(model, x_test, batch_size, device, use_bf16)
        real_predictions = predictions[:, 0]
        baseline_predictions = predictions[:, 1]

        accuracies[layer] = np.mean(real_predictions == test_labels, axis=0)
        baselines[layer] = np.mean(baseline_predictions == test_labels, axis=0)
        per_class[layer], class_support[layer] = _accuracy_by_class(
            test_labels,
            real_predictions,
        )
        balanced_accuracies[layer] = np.nanmean(per_class[layer], axis=1)
        per_ply[layer], ply_support[layer] = _accuracy_by_ply(
            test_labels,
            real_predictions,
            test_plies,
        )
        layer_checkpoints.append(
            {
                "state_dict": state_dict,
                "feature_mean": feature_mean.cpu(),
                "feature_std": feature_std.cpu(),
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
        "test_indices": test_indices,
    }
    checkpoint = {
        "layers": layer_checkpoints,
        "n_layers": n_layers,
        "n_squares": n_squares,
        "n_features": n_features,
        "n_classes": N_PIECE_CLASSES,
        "random_state": RANDOM_STATE,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "hyperparameters": {
            "optimizer": "Adam",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
    }
    _save_results(output_dir, results, checkpoint)

    for layer, mean_accuracy in enumerate(np.nanmean(accuracies, axis=1), start=1):
        print(f"Layer {layer} mean accuracy: {mean_accuracy:.4f}")
    print(f"Mean baseline accuracy: {np.nanmean(baselines):.4f}")

    ply_means = np.nanmean(per_ply, axis=(0, 1))
    for (lower, upper), mean_accuracy in zip(PLY_BUCKETS, ply_means):
        label = f"{lower}+" if upper is None else f"{lower}-{upper}"
        print(f"Ply {label} mean accuracy: {mean_accuracy:.4f}")

    print(f"Total wall time: {time.perf_counter() - start_time:.1f}s")
    print(f"Saved probe results to: {output_dir}")
    return {**results, "checkpoint": checkpoint}


def main():
    parser = argparse.ArgumentParser(description="Train CUDA linear probes on chess activations")
    parser.add_argument("--activations", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    train_probes(
        args.activations,
        args.output_dir,
        args.device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
    )


if __name__ == "__main__":
    main()
