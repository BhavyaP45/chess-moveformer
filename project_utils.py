import os
import re
from pathlib import Path

import torch
from dotenv import load_dotenv


load_dotenv()

CHECKPOINT_PATTERN = re.compile(
    r"^(?P<best>best_)?ckpt_step(?P<step>\d+)_valloss"
    r"(?P<val_loss>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\.pt$"
)


def root_dir() -> Path:
    return Path(os.environ["ROOT_DIR"])


def checkpoint_dir() -> Path:
    configured_path = Path(os.environ["CHECKPOINT_DIR"])
    path = configured_path if configured_path.is_absolute() else root_dir() / configured_path
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_metadata(checkpoint_path: Path):
    match = CHECKPOINT_PATTERN.fullmatch(checkpoint_path.name)
    if match is None:
        return None

    return {
        "step": int(match.group("step")),
        "val_loss": float(match.group("val_loss")),
        "is_best": match.group("best") is not None,
    }


def valid_checkpoints():
    return tuple(
        (checkpoint_path, metadata)
        for checkpoint_path in checkpoint_dir().glob("*.pt")
        if (metadata := checkpoint_metadata(checkpoint_path)) is not None
    )


def _required_checkpoints():
    checkpoints = valid_checkpoints()
    if not checkpoints:
        raise FileNotFoundError(f"No valid checkpoints found in {checkpoint_dir()}")
    return checkpoints


def find_latest_checkpoint() -> Path:
    return max(_required_checkpoints(), key=lambda item: item[1]["step"])[0]


def find_step_0_checkpoint() -> Path:
    return min(_required_checkpoints(), key=lambda item: item[1]["step"])[0]


def find_best_checkpoint() -> Path:
    return min(_required_checkpoints(), key=lambda item: item[1]["val_loss"])[0]


def save_checkpoint(model, optimizer, step, train_loss, val_loss, config):
    filename = f"ckpt_step{step}_valloss{val_loss:.4f}.pt"
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }
    torch.save(checkpoint, checkpoint_dir() / filename)


def read_checkpoint(checkpoint_path, map_location="cpu"):
    return torch.load(checkpoint_path, map_location=map_location, weights_only=False)


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = read_checkpoint(checkpoint_path, map_location=next(model.parameters()).device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"]


def load_latest_checkpoint(model, optimizer):
    checkpoints = valid_checkpoints()
    if not checkpoints:
        return None
    return load_checkpoint(find_latest_checkpoint(), model, optimizer)


def load_step_0_checkpoint(model, optimizer):
    checkpoints = valid_checkpoints()
    if not checkpoints:
        return None
    return load_checkpoint(find_step_0_checkpoint(), model, optimizer)


def load_best_checkpoint(model, optimizer):
    checkpoints = valid_checkpoints()
    if not checkpoints:
        return None
    return load_checkpoint(find_best_checkpoint(), model, optimizer)


def iter_games(data_path: str | Path, min_plies: int = 0):
    with Path(data_path).open("r", encoding="utf-8") as file:
        for line in file:
            game = line.rstrip("\r\n")
            if game and (min_plies == 0 or len(game.split()) >= min_plies):
                yield game


def load_games(data_path: str | Path, min_plies: int = 0):
    return tuple(iter_games(data_path, min_plies))
