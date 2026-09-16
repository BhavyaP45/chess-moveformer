import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from project_utils import checkpoint_dir, checkpoint_metadata, read_checkpoint, root_dir
from train_probes import (
    BatchedNonlinearProbes,
    N_PIECE_CLASSES,
    PLY_BUCKETS,
    _relative_labels,
    _select_turn_bank,
    _turn_indices,
    load_probe_checkpoint,
)

INTERVENTION_SUMMARY_FILENAME = "causal_empty_intervention_summary_v4.csv"
INTERVENTION_EXAMPLE_FILENAME = "causal_empty_intervention_examples_v4.csv"
INTERVENTION_CI_FILENAME = "causal_empty_intervention_confidence_intervals_v4.csv"
INTERVENTION_CI_NPZ_FILENAME = "causal_empty_intervention_confidence_intervals_v4.npz"
INTERVENTION_BAR_METRICS = (
    ("plan_retention", "Plan Retention", "plan_retention"),
    ("source_square_usage", "Source-Square Usage", "source_square_usage"),
    ("legality", "Legality", "legal"),
)
PRIMARY_INTERVENTION_SCALE = 1.0
DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 42

LEGALITY_RATES = {
    6: {1: 99.9140625, 5: 99.6328125, 10: 99.640625},
    10: {1: 99.296875, 5: 98.484375, 10: 98.078125},
    14: {1: 98.203125, 5: 95.765625, 10: 95.4453125},
    18: {1: 95.953125, 5: 92.84375, 10: 91.8984375},
    22: {1: 93.390625, 5: 88.859375, 10: 87.7734375},
    26: {1: 90.2421875, 5: 85.109375, 10: 84.015625},
    30: {1: 87.96875, 5: 81.5625, 10: 80.8359375},
    34: {1: 86.015625, 5: 79.65625, 10: 78.453125},
    38: {1: 83.375, 5: 77.234375, 10: 76.265625},
    42: {1: 81.9296875, 5: 75.71875, 10: 74.0859375},
    46: {1: 81.3984375, 5: 74.828125, 10: 73.4140625},
    50: {1: 80.765625, 5: 73.8359375, 10: 72.578125},
}
PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _figures_dir() -> Path:
    figures_dir = root_dir() / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _checkpoint_records():
    records_by_step = {}

    for checkpoint_path in checkpoint_dir().glob("*.pt"):
        metadata = checkpoint_metadata(checkpoint_path)
        if metadata is None:
            warnings.warn(f"Ignoring malformed checkpoint filename: {checkpoint_path.name}")
            continue

        record = {
            "path": checkpoint_path,
            **metadata,
        }
        step = metadata["step"]
        existing = records_by_step.get(step)
        if existing is None or existing["is_best"] and not record["is_best"]:
            records_by_step[step] = record

    if not records_by_step:
        raise FileNotFoundError(f"No valid checkpoints found in {checkpoint_dir()}")

    records = sorted(records_by_step.values(), key=lambda record: record["step"])
    for record in records:
        checkpoint = read_checkpoint(record["path"])
        if "train_loss" not in checkpoint:
            raise KeyError(f"Checkpoint has no train_loss: {record['path']}")
        train_loss = checkpoint["train_loss"]
        record["train_loss"] = float(train_loss.item() if torch.is_tensor(train_loss) else train_loss)

    return records


def _minimal_axes(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def plot_loss_curve() -> Path:
    records = _checkpoint_records()
    steps = [record["step"] for record in records]
    train_losses = [record["train_loss"] for record in records]
    val_losses = [record["val_loss"] for record in records]

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
        ax.plot(steps, train_losses, color="#0072B2", marker="o", markersize=3, label="Train")
        ax.plot(steps, val_losses, color="#D55E00", marker="s", markersize=3, label="Validation")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False)
        _minimal_axes(ax)

        output_path = _figures_dir() / "loss_curve.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    return output_path


def plot_legality_curve() -> Path:
    ply_depths = sorted(LEGALITY_RATES)
    colors = {1: "#0072B2", 5: "#E69F00", 10: "#009E73"}
    markers = {1: "o", 5: "s", 10: "^"}

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
        for k in (1, 5, 10):
            rates = [LEGALITY_RATES[ply][k] for ply in ply_depths]
            ax.plot(
                ply_depths,
                rates,
                color=colors[k],
                marker=markers[k],
                markersize=3,
                label=f"k={k}",
            )

        ax.axhline(0, color="#777777", linestyle="--", linewidth=1.1, label="Random baseline")
        ax.set_xlabel("Ply Depth")
        ax.set_ylabel("Legality Rate (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(ply_depths[::2])
        ax.set_yticks(range(0, 101, 20))
        ax.legend(frameon=False, ncol=2)
        _minimal_axes(ax)

        output_path = _figures_dir() / "legality_curve.pdf"
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    return output_path


def _ply_bucket_labels():
    return [
        f"{lower}+" if upper is None else f"{lower}–{upper}"
        for lower, upper in PLY_BUCKETS
    ]


def _balanced_accuracy_by_square_and_ply(labels, predictions, plies):
    n_squares = labels.shape[1]
    recalls = np.full(
        (n_squares, len(PLY_BUCKETS), N_PIECE_CLASSES),
        np.nan,
        dtype=np.float32,
    )
    supports = np.zeros_like(recalls, dtype=np.int32)

    for bucket_index, (lower, upper) in enumerate(PLY_BUCKETS):
        ply_mask = (
            plies >= lower
            if upper is None
            else (plies >= lower) & (plies <= upper)
        )
        bucket_labels = labels[ply_mask]
        bucket_predictions = predictions[ply_mask]
        for piece_class in range(N_PIECE_CLASSES):
            class_mask = bucket_labels == piece_class
            class_support = class_mask.sum(axis=0)
            supports[:, bucket_index, piece_class] = class_support
            correct = ((bucket_predictions == piece_class) & class_mask).sum(axis=0)
            present = class_support > 0
            recalls[present, bucket_index, piece_class] = (
                correct[present] / class_support[present]
            )

    valid = np.isfinite(recalls)
    recall_sum = np.nansum(recalls, axis=-1)
    recall_count = valid.sum(axis=-1)
    balanced = np.divide(
        recall_sum,
        recall_count,
        out=np.full_like(recall_sum, np.nan),
        where=recall_count > 0,
    )
    return balanced.astype(np.float32), recalls, supports


def _predict_saved_probe_layer(
    checkpoint,
    activations,
    test_indices,
    plies,
    layer,
    device,
    batch_size,
):
    n_squares = checkpoint["n_squares"]
    model = BatchedNonlinearProbes(
        checkpoint["n_features"],
        n_squares,
        checkpoint["hidden_dim"],
        checkpoint["n_classes"],
    ).to(device)
    layer_state = checkpoint["layers"][layer]
    model.load_state_dict(layer_state["state_dict"])
    model.eval()
    feature_mean = layer_state["feature_mean"].to(device)
    feature_std = layer_state["feature_std"].to(device)
    predictions = np.empty(
        (len(test_indices), 2, n_squares),
        dtype=np.int8,
    )
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    turns = _turn_indices(plies[test_indices])

    with torch.inference_mode():
        for start in range(0, len(test_indices), batch_size):
            indices = test_indices[start : start + batch_size]
            inputs = torch.as_tensor(
                np.asarray(activations[indices, layer], dtype=np.float32),
                device=device,
            )
            inputs = (inputs - feature_mean) / feature_std
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                batch_turns = torch.as_tensor(
                    turns[start : start + len(indices)],
                    dtype=torch.long,
                    device=device,
                )
                batch_predictions = _select_turn_bank(
                    model(inputs),
                    batch_turns,
                ).argmax(dim=-1)
            predictions[start : start + len(indices)] = (
                batch_predictions.cpu().numpy()
            )
    return predictions


def _default_activation_path():
    local_path = Path("/content/activations.npz")
    if local_path.exists():
        return local_path
    drive_path = root_dir() / "activations.npz"
    if drive_path.exists():
        return drive_path
    raise FileNotFoundError(
        "activations.npz not found at /content/activations.npz or "
        f"{drive_path}. Copy the file into /content before plotting."
    )


def compute_probe_balanced_accuracy_by_ply(
    activation_path=None,
    checkpoint_path=None,
    split_path=None,
    output_path=None,
    device=None,
    batch_size=8192,
):
    drive_dir = root_dir()
    activation_path = (
        Path(activation_path) if activation_path else _default_activation_path()
    )
    checkpoint_path = (
        Path(checkpoint_path) if checkpoint_path else drive_dir / "probe_weights.pt"
    )
    split_path = Path(split_path) if split_path else drive_dir / "probe_split.npz"
    output_path = (
        Path(output_path)
        if output_path
        else drive_dir / "probe_balanced_accuracy_by_ply.npz"
    )
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print(f"Activations: {activation_path}")
    print(f"Probe weights: {checkpoint_path}")
    print(f"Probe split: {split_path}")
    print(f"Metrics output: {output_path}")

    checkpoint = load_probe_checkpoint(checkpoint_path, map_location="cpu")
    with np.load(activation_path) as activation_data:
        activations = np.asarray(activation_data["activations"])
        labels = np.asarray(activation_data["labels"])
        plies = np.asarray(activation_data["plies"])
    with np.load(split_path) as split_data:
        test_indices = np.asarray(split_data["test_indices"], dtype=np.int64)

    n_layers = checkpoint["n_layers"]
    n_squares = checkpoint["n_squares"]
    if activations.shape[:2] != (len(labels), n_layers):
        raise ValueError("Activation file does not match the saved probe checkpoint")
    if labels.shape[1] != n_squares:
        raise ValueError("Label squares do not match the saved probe checkpoint")
    n_positions = len(labels)
    if len(test_indices) == 0 or test_indices.max() >= n_positions:
        raise ValueError(
            "Probe split does not match the activation file: "
            f"n_positions={n_positions}, "
            f"len(test_indices)={len(test_indices)}, "
            f"test_indices.max()={int(test_indices.max()) if len(test_indices) else 'n/a'}. "
            "Use the activations.npz that was used when training these probes, "
            "or retrain probes on the current activations.npz."
        )

    relative_labels = _relative_labels(labels, plies)
    test_labels = relative_labels[test_indices]
    test_plies = plies[test_indices]
    shape = (n_layers, n_squares, len(PLY_BUCKETS))
    real_balanced = np.full(shape, np.nan, dtype=np.float32)
    shuffled_balanced = np.full(shape, np.nan, dtype=np.float32)
    real_recalls = np.full(
        (*shape, N_PIECE_CLASSES), np.nan, dtype=np.float32
    )
    shuffled_recalls = np.full_like(real_recalls, np.nan)
    class_support = np.zeros_like(real_recalls, dtype=np.int32)

    for layer in range(n_layers):
        predictions = _predict_saved_probe_layer(
            checkpoint,
            activations,
            test_indices,
            plies,
            layer,
            device,
            batch_size,
        )
        (
            real_balanced[layer],
            real_recalls[layer],
            class_support[layer],
        ) = _balanced_accuracy_by_square_and_ply(
            test_labels,
            predictions[:, 0],
            test_plies,
        )
        (
            shuffled_balanced[layer],
            shuffled_recalls[layer],
            _,
        ) = _balanced_accuracy_by_square_and_ply(
            test_labels,
            predictions[:, 1],
            test_plies,
        )
        print(f"Balanced ply metrics: layer {layer + 1}/{n_layers}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        balanced_accuracy_per_square=real_balanced,
        balanced_accuracy_mean=np.nanmean(real_balanced, axis=1),
        balanced_accuracy_std=np.nanstd(real_balanced, axis=1),
        shuffled_balanced_accuracy_per_square=shuffled_balanced,
        shuffled_balanced_accuracy_mean=np.nanmean(
            shuffled_balanced, axis=1
        ),
        shuffled_balanced_accuracy_std=np.nanstd(
            shuffled_balanced, axis=1
        ),
        recall_per_class=real_recalls,
        shuffled_recall_per_class=shuffled_recalls,
        class_support=class_support,
        ply_bucket_labels=np.asarray(_ply_bucket_labels()),
    )
    print(f"Saved {output_path}")
    return output_path


def plot_probe_balanced_accuracy_by_ply(metrics_path=None, output_dir=None):
    metrics_path = (
        Path(metrics_path)
        if metrics_path
        else root_dir() / "probe_balanced_accuracy_by_ply.npz"
    )
    output_dir = Path(output_dir) if output_dir else _figures_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(metrics_path) as data:
        means = np.asarray(data["balanced_accuracy_mean"])
        per_square = np.asarray(data["balanced_accuracy_per_square"])
        shuffled = np.asarray(data["shuffled_balanced_accuracy_per_square"])
        bucket_labels = data["ply_bucket_labels"].astype(str).tolist()

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
        image = ax.imshow(means, aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(bucket_labels)), bucket_labels)
        ax.set_yticks(np.arange(means.shape[0]), np.arange(1, means.shape[0] + 1))
        ax.set_xlabel("Ply Bucket")
        ax.set_ylabel("Transformer Layer")
        ax.set_title("Probe Balanced Accuracy by Layer and Ply Bucket")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Balanced Accuracy")
        heatmap_path = output_dir / "probe_balanced_accuracy_layer_ply_heatmap.pdf"
        heatmap_png = heatmap_path.with_suffix(".png")
        fig.savefig(heatmap_path, format="pdf", bbox_inches="tight")
        fig.savefig(heatmap_png, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved heatmap: {heatmap_path}", flush=True)
        print(f"Saved heatmap: {heatmap_png}", flush=True)

        layer = per_square.shape[0] - 1
        layer_values = per_square[layer]
        layer_means = np.nanmean(layer_values, axis=0)
        layer_stds = np.nanstd(layer_values, axis=0)
        shuffled_chance = float(np.nanmean(shuffled[layer]))
        x_values = np.arange(len(bucket_labels))

        fig, ax = plt.subplots(figsize=(4.3, 3.0), constrained_layout=True)
        ax.errorbar(
            x_values,
            layer_means,
            yerr=layer_stds,
            color="#0072B2",
            marker="o",
            markersize=4,
            capsize=3,
        )
        ax.axhline(
            shuffled_chance,
            color="#777777",
            linestyle="--",
            linewidth=1.1,
            label="Chance (shuffled labels)",
        )
        ax.set_xticks(x_values, bucket_labels)
        ax.set_xlabel("Ply Bucket")
        ax.set_ylabel("Mean Balanced Accuracy")
        ax.set_title(f"Layer {layer + 1} Probe Accuracy by Ply Bucket")
        ax.legend(frameon=False)
        _minimal_axes(ax)
        line_path = output_dir / "probe_balanced_accuracy_layer6_by_ply.pdf"
        line_png = line_path.with_suffix(".png")
        fig.savefig(line_path, format="pdf", bbox_inches="tight")
        fig.savefig(line_png, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved line plot: {line_path}", flush=True)
        print(f"Saved line plot: {line_png}", flush=True)

    return heatmap_path, line_path


def _default_intervention_path(filename):
    candidates = (root_dir() / filename, root_dir() / "results" / filename)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Intervention artifact {filename} not found at "
        + " or ".join(str(path) for path in candidates)
    )


def _parse_csv_bool(value, field_name):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean {value!r} in field {field_name}")


def _load_intervention_pairs(example_path):
    paired = {}
    with Path(example_path).open(encoding="utf-8", newline="") as file:
        for row_number, row in enumerate(csv.DictReader(file), start=2):
            try:
                scale = float(row["scale"])
                example_id = int(row["example_id"])
                condition = row["condition"]
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid intervention example row {row_number}"
                ) from error
            if condition not in {"treatment", "control"}:
                raise ValueError(
                    f"Unknown condition {condition!r} at row {row_number}"
                )
            key = (scale, example_id)
            conditions = paired.setdefault(key, {})
            if condition in conditions:
                raise ValueError(
                    f"Duplicate {condition} row for scale={scale:g}, "
                    f"example_id={example_id}"
                )
            conditions[condition] = row

    records = []
    for (scale, example_id), conditions in sorted(paired.items()):
        missing = {"treatment", "control"} - set(conditions)
        if missing:
            raise ValueError(
                f"Missing {', '.join(sorted(missing))} row for "
                f"scale={scale:g}, example_id={example_id}"
            )
        treatment = conditions["treatment"]
        control = conditions["control"]
        for field in ("game_id", "piece_class", "piece"):
            if treatment[field] != control[field]:
                raise ValueError(
                    f"Mismatched {field} for scale={scale:g}, "
                    f"example_id={example_id}"
                )
        records.append(
            {
                "scale": scale,
                "example_id": example_id,
                "game_id": int(treatment["game_id"]),
                "piece_class": int(treatment["piece_class"]),
                "piece": treatment["piece"],
                "treatment": {
                    name: _parse_csv_bool(treatment[csv_field], csv_field)
                    for name, _label, csv_field in INTERVENTION_BAR_METRICS
                },
                "control": {
                    name: _parse_csv_bool(control[csv_field], csv_field)
                    for name, _label, csv_field in INTERVENTION_BAR_METRICS
                },
            }
        )
    if not records:
        raise ValueError(f"No paired intervention examples found in {example_path}")
    return records


def _bootstrap_confidence_rows(
    records, scope, piece_class, scale, n_bootstrap, seed
):
    game_ids = np.asarray([record["game_id"] for record in records], dtype=np.int64)
    unique_games, game_inverse = np.unique(game_ids, return_inverse=True)
    cluster_counts = np.bincount(game_inverse).astype(np.int64)
    metric_names = [name for name, _label, _csv_field in INTERVENTION_BAR_METRICS]
    treatment = np.asarray(
        [
            [record["treatment"][name] for name in metric_names]
            for record in records
        ],
        dtype=np.float64,
    )
    control = np.asarray(
        [
            [record["control"][name] for name in metric_names]
            for record in records
        ],
        dtype=np.float64,
    )
    differences = treatment - control
    cluster_difference_sums = np.stack(
        [
            np.bincount(game_inverse, weights=differences[:, column])
            for column in range(len(metric_names))
        ],
        axis=1,
    )

    rng = np.random.default_rng(seed)
    bootstrap_differences = np.empty(
        (n_bootstrap, len(metric_names)), dtype=np.float64
    )
    chunk_size = 256
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        sampled_clusters = rng.integers(
            0,
            len(unique_games),
            size=(stop - start, len(unique_games)),
        )
        sampled_counts = cluster_counts[sampled_clusters].sum(axis=1)
        sampled_sums = cluster_difference_sums[sampled_clusters].sum(axis=1)
        bootstrap_differences[start:stop] = (
            100.0 * sampled_sums / sampled_counts[:, None]
        )

    lower = np.percentile(bootstrap_differences, 2.5, axis=0)
    upper = np.percentile(bootstrap_differences, 97.5, axis=0)
    rows = []
    for column, name in enumerate(metric_names):
        rows.append(
            {
                "scope": scope,
                "piece_class": piece_class,
                "scale": scale,
                "metric": name,
                "n_pairs": len(records),
                "treatment_percentage": 100.0 * treatment[:, column].mean(),
                "control_percentage": 100.0 * control[:, column].mean(),
                "treatment_minus_control_pp": 100.0
                * differences[:, column].mean(),
                "ci_lower_pp": float(lower[column]),
                "ci_upper_pp": float(upper[column]),
                "bootstrap_seed": seed,
                "n_bootstrap": n_bootstrap,
            }
        )
    return rows


def _save_confidence_rows(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / INTERVENTION_CI_FILENAME
    npz_path = output_dir / INTERVENTION_CI_NPZ_FILENAME
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        npz_path,
        **{field: np.asarray([row[field] for row in rows]) for field in fieldnames},
    )
    return csv_path, npz_path


def compute_intervention_confidence_intervals(
    example_path=None,
    output_dir=None,
    n_bootstrap=DEFAULT_BOOTSTRAP_RESAMPLES,
    seed=DEFAULT_BOOTSTRAP_SEED,
):
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    example_path = (
        Path(example_path)
        if example_path
        else _default_intervention_path(INTERVENTION_EXAMPLE_FILENAME)
    )
    output_dir = Path(output_dir) if output_dir else example_path.parent
    records = _load_intervention_pairs(example_path)
    rows = []
    scales = sorted({record["scale"] for record in records})
    piece_classes = sorted(
        {(record["piece_class"], record["piece"]) for record in records}
    )
    for scale in scales:
        scale_records = [record for record in records if record["scale"] == scale]
        rows.extend(
            _bootstrap_confidence_rows(
                scale_records, "all", -1, scale, n_bootstrap, seed
            )
        )
        for piece_class, piece in piece_classes:
            scoped_records = [
                record
                for record in scale_records
                if record["piece_class"] == piece_class
            ]
            if scoped_records:
                rows.extend(
                    _bootstrap_confidence_rows(
                        scoped_records,
                        piece,
                        piece_class,
                        scale,
                        n_bootstrap,
                        seed,
                    )
                )
    paths = _save_confidence_rows(rows, output_dir)
    for path in paths:
        print(f"Saved intervention confidence intervals: {path}", flush=True)
    return paths


def _primary_intervention_rows(confidence_path):
    selected = {}
    with Path(confidence_path).open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if (
                row["scope"] == "all"
                and float(row["scale"]) == PRIMARY_INTERVENTION_SCALE
            ):
                selected[row["metric"]] = row
    expected = {name for name, _label, _csv_field in INTERVENTION_BAR_METRICS}
    missing = expected - set(selected)
    if missing:
        raise ValueError(
            f"Missing primary confidence rows for {sorted(missing)} in "
            f"{confidence_path}"
        )
    return selected


def plot_causal_empty_intervention(confidence_path=None, output_dir=None):
    confidence_path = (
        Path(confidence_path)
        if confidence_path
        else _default_intervention_path(INTERVENTION_CI_FILENAME)
    )
    output_dir = Path(output_dir) if output_dir else _figures_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _primary_intervention_rows(confidence_path)

    labels = [label for _name, label, _csv_field in INTERVENTION_BAR_METRICS]
    treatment = [
        float(rows[name]["treatment_percentage"])
        for name, _label, _csv_field in INTERVENTION_BAR_METRICS
    ]
    control = [
        float(rows[name]["control_percentage"])
        for name, _label, _csv_field in INTERVENTION_BAR_METRICS
    ]
    deltas = [
        float(rows[name]["treatment_minus_control_pp"])
        for name, _label, _csv_field in INTERVENTION_BAR_METRICS
    ]
    intervals = [
        (float(rows[name]["ci_lower_pp"]), float(rows[name]["ci_upper_pp"]))
        for name, _label, _csv_field in INTERVENTION_BAR_METRICS
    ]

    x_values = np.arange(len(labels))
    width = 0.36
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
        ax.bar(
            x_values - width / 2,
            treatment,
            width,
            color="#0072B2",
            label="Treatment",
        )
        ax.bar(
            x_values + width / 2,
            control,
            width,
            color="#D55E00",
            label="Control",
        )
        for x_value, left, right, delta, interval in zip(
            x_values, treatment, control, deltas, intervals
        ):
            ax.annotate(
                f"Delta {delta:+.2f} pp\n"
                f"95% CI [{interval[0]:+.2f}, {interval[1]:+.2f}]",
                (x_value, max(left, right) + 2.2),
                ha="center",
                va="bottom",
                fontsize=6.5,
            )
        ax.set_xticks(x_values, labels)
        ax.set_ylabel("Percentage")
        ax.set_ylim(0, 112)
        ax.set_title(
            "Plan Retention, Source-Square Usage, and Legality "
            "under Empty Intervention"
        )
        ax.legend(frameon=False)
        _minimal_axes(ax)
        output_path = output_dir / "causal_empty_intervention_scale1_bars.pdf"
        png_path = output_path.with_suffix(".png")
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved intervention bars: {output_path}", flush=True)
    print(f"Saved intervention bars: {png_path}", flush=True)
    return output_path


def compute_and_plot_probe_balanced_accuracy_by_ply(**kwargs):
    metrics_path = compute_probe_balanced_accuracy_by_ply(**kwargs)
    return metrics_path, *plot_probe_balanced_accuracy_by_ply(metrics_path)


def main():
    loss_path = plot_loss_curve()
    legality_path = plot_legality_curve()
    print(f"Saved {loss_path}")
    print(f"Saved {legality_path}")
    metrics_path = root_dir() / "probe_balanced_accuracy_by_ply.npz"
    if metrics_path.exists():
        heatmap_path, line_path = plot_probe_balanced_accuracy_by_ply(metrics_path)
        print(f"Saved {heatmap_path}")
        print(f"Saved {line_path}")
    try:
        try:
            confidence_path = _default_intervention_path(INTERVENTION_CI_FILENAME)
        except FileNotFoundError:
            confidence_path, _ = compute_intervention_confidence_intervals()
        intervention_path = plot_causal_empty_intervention(confidence_path)
        print(f"Saved {intervention_path}")
    except FileNotFoundError as error:
        print(error)


if __name__ == "__main__":
    main()
