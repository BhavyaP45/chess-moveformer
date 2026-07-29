from dataclasses import dataclass
from pathlib import Path

import torch

from model import MoveFormerConfig, MoveFormerModel


@dataclass
class TrainingConfig:
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    eval_iters: int = 200


training_config = TrainingConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = Path(__file__).parent / "data" / "train.txt"
PAD_TOKEN = "\0"


def _load_training_games() -> tuple[str, ...]:
    """Load the corpus once; get_batch samples from this in-memory tuple."""
    with DATA_PATH.open("r", encoding="utf-8") as file:
        games = tuple(
            game
            for line in file
            if (game := line.rstrip("\r\n"))
        )
    return games


TRAIN_GAMES = _load_training_games()

chars = [PAD_TOKEN] + sorted({char for game in TRAIN_GAMES for char in game})
vocab_size = len(chars)
stoi = {char: index for index, char in enumerate(chars)}
itos = {index: char for char, index in stoi.items()}

encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string



def get_batch(split: str, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample games and return next-character input and target tensors."""
    sequence_length = block_size + 1
    encoded_games: list[list[int]] = []
    game_indices = torch.randint(len(TRAIN_GAMES), (batch_size,))

    for game_index in game_indices:
        game = TRAIN_GAMES[game_index.item()]

        if len(game) > sequence_length:
            start = torch.randint(len(game) - sequence_length + 1, (1,)).item()
            sequence = game[start : start + sequence_length]
        else:
            sequence = game.ljust(sequence_length, PAD_TOKEN)

        encoded_games.append([stoi[char] for char in sequence])

    tokens = torch.tensor(encoded_games, dtype=torch.long, device=DEVICE)
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    return x, y


model_config = MoveFormerConfig(vocab_size=vocab_size)
model = MoveFormerModel(
    model_config.block_size,
    model_config.n_layer,
    model_config.n_head,
    model_config.n_embd,
    model_config.dropout,
    model_config.vocab_size,
).to(DEVICE)
