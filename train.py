from dataclasses import asdict, dataclass
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from model import MoveFormerConfig, MoveFormerModel


load_dotenv()


@dataclass
class TrainingConfig:
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    eval_iters: int = 200


training_config = TrainingConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT_DIR = Path(os.environ["ROOT_DIR"])
DATA_PATH = ROOT_DIR / "data" / "train.txt"
PAD_TOKEN = "\0"
BOS_TOKEN = "\1"


def _checkpoint_dir() -> Path:
    checkpoint_dir = ROOT_DIR / os.environ["CHECKPOINT_DIR"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


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
    torch.save(checkpoint, _checkpoint_dir() / filename)


def load_latest_checkpoint(model, optimizer):
    checkpoints = list(_checkpoint_dir().glob("*ckpt_step*_valloss*.pt"))
    if not checkpoints:
        return None

    selected_checkpoint = min(
        checkpoints,
        key=lambda path: float(path.stem.rsplit("_valloss", 1)[1]),
    )
    checkpoint = torch.load(
        selected_checkpoint,
        map_location=next(model.parameters()).device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"]


def _load_training_games() -> tuple[str, ...]:
    """Load the corpus once; get_batch samples from this in-memory tuple."""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        games = tuple(
            game
            for line in file
            if (game := line.rstrip("\r\n"))
        )
    return games


games = _load_training_games()
chars = [PAD_TOKEN, BOS_TOKEN] + sorted({char for game in games for char in game})
vocab_size = len(chars)
stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for char, index in stoi.items()}

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

encoded_games = tuple(bytes(encode(BOS_TOKEN + game)) for game in games)
split_index = int(0.9 * len(encoded_games))
train_data = encoded_games[:split_index]
val_data = encoded_games[split_index:]
del games, encoded_games


model_config = MoveFormerConfig(vocab_size=vocab_size)
model = MoveFormerModel(
    model_config.block_size,
    model_config.n_layer,
    model_config.n_head,
    model_config.n_embd,
    model_config.dropout,
    model_config.vocab_size,
).to(DEVICE)



def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample games and return next-character input and target tensors."""
    data = train_data if split == "train" else val_data
    sequence_length = model_config.block_size + 1
    batch_sequences: list[list[int]] = []
    game_indices = torch.randint(len(data), (training_config.batch_size,))

    for game_index in game_indices:
        game = data[game_index.item()]

        if len(game) > sequence_length:
            start = torch.randint(len(game) - sequence_length + 1, (1,)).item()
            sequence = game[start : start + sequence_length]
        else:
            sequence = game.ljust(sequence_length, b"\0")

        batch_sequences.append(list(sequence))

    tokens = torch.tensor(batch_sequences, dtype=torch.long, device=DEVICE)
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return x, y


@torch.no_grad()
def estimate_loss():
    losses_by_split = {}
    model.eval()

    for split in ("train", "val"):
        losses = torch.zeros(training_config.eval_iters)
        for evaluation in range(training_config.eval_iters):
            inputs, targets = get_batch(split)
            _, loss = model(inputs, targets)
            losses[evaluation] = loss.item()
        losses_by_split[split] = losses.mean()

    model.train()
    return losses_by_split


def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    resume_step = load_latest_checkpoint(model, optimizer)
    start_step = 0 if resume_step is None else resume_step + 1
    checkpoint_config = {
        "model": asdict(model_config),
        "training": asdict(training_config),
        "stoi": stoi,
    }

    for step in range(start_step, training_config.max_iters):
        if step % training_config.eval_interval == 0 or step == training_config.max_iters - 1:
            losses = estimate_loss()
            train_loss = losses["train"].item()
            val_loss = losses["val"].item()
            print(f"step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

            save_checkpoint(model, optimizer, step, train_loss, val_loss, checkpoint_config)

        inputs, targets = get_batch("train")
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.tensor([[stoi[BOS_TOKEN]]], dtype=torch.long, device=DEVICE)
    generated = model.generate(context, max_new_tokens=500)[0].tolist()
    print(decode(generated[1:]))


if __name__ == "__main__":
    train()
