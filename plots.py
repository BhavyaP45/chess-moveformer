import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from project_utils import checkpoint_dir, checkpoint_metadata, read_checkpoint, root_dir

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


def main():
    loss_path = plot_loss_curve()
    legality_path = plot_legality_curve()
    print(f"Saved {loss_path}")
    print(f"Saved {legality_path}")


if __name__ == "__main__":
    main()
