import argparse
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch

from model import MoveFormerConfig, MoveFormerModel
from project_utils import find_best_checkpoint, load_games, read_checkpoint, root_dir

N_GAMES = 10000
BATCH_SIZE = 256
CONTEXT_LENGTH = 768
MIN_GAME_PLIES = 51
OUTPUT_FILENAME = "activations.npz"


@dataclass
class PreparedGame:
    tokens: list[int]
    boundary_positions: np.ndarray
    labels: np.ndarray
    plies: np.ndarray
    game_id: int


def _load_model(device):
    checkpoint_path = find_best_checkpoint()
    checkpoint = read_checkpoint(checkpoint_path)
    saved_config = checkpoint["config"]
    model_values = saved_config["model"]
    config = MoveFormerConfig(**model_values)
    stoi = saved_config["stoi"]
    itos = {index: char for char, index in stoi.items()}

    model = MoveFormerModel(
        config.block_size,
        config.n_layer,
        config.n_head,
        config.n_embd,
        config.dropout,
        config.vocab_size,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path.name}")
    return model, config, stoi, itos


def _board_labels(board: chess.Board) -> np.ndarray:
    labels = np.zeros(64, dtype=np.int8)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            labels[square] = piece.piece_type if piece.color == chess.WHITE else piece.piece_type + 6

    return labels


def _prepare_game(game_text: str, game_id: int, stoi, context_length: int):
    moves = game_text.split()
    board = chess.Board()
    boundary_positions = []
    board_states = []
    plies = []
    character_position = 1

    try:
        for ply, san in enumerate(moves, start=1):
            board.push_san(san)
            character_position += len(san)

            if ply < len(moves):
                if character_position < context_length:
                    boundary_positions.append(character_position)
                    board_states.append(_board_labels(board))
                    plies.append(ply)
                character_position += 1
    except ValueError:
        return None

    if not boundary_positions:
        return None

    encoded_text = ("\1" + game_text)[:context_length]
    tokens = [stoi[char] for char in encoded_text]
    return PreparedGame(
        tokens=tokens,
        boundary_positions=np.asarray(boundary_positions, dtype=np.int64),
        labels=np.stack(board_states),
        plies=np.asarray(plies, dtype=np.int16),
        game_id=game_id,
    )


def _sample_valid_games(data_path: Path, stoi, context_length: int):
    games = load_games(data_path, min_plies=MIN_GAME_PLIES)
    sampled_games = []
    for game_id in np.random.default_rng().permutation(len(games)):
        prepared = _prepare_game(games[game_id], int(game_id), stoi, context_length)
        if prepared is not None:
            sampled_games.append(prepared)
        if len(sampled_games) == N_GAMES:
            break

    if len(sampled_games) < N_GAMES:
        raise RuntimeError(f"Only found {len(sampled_games)} valid games; requested {N_GAMES}")

    sampled_games.sort(key=lambda game: len(game.tokens))
    return sampled_games


def _make_hook(layer_index, captured_outputs):
    def hook(_module, _inputs, output):
        captured_outputs[layer_index] = output

    return hook


def extract_activations(output_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, stoi, _ = _load_model(device)
    context_length = min(CONTEXT_LENGTH, config.block_size)
    data_path = root_dir() / "data" / "val.txt"
    games = _sample_valid_games(data_path, stoi, context_length)
    use_cuda = device.type == "cuda"
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    activation_batches = []
    label_batches = []
    ply_batches = []
    game_id_batches = []
    captured_outputs = [None] * config.n_layer
    hooks = [
        block.register_forward_hook(_make_hook(layer_index, captured_outputs))
        for layer_index, block in enumerate(model.blocks)
    ]
    total_boundaries = 0
    total_batches = (len(games) + BATCH_SIZE - 1) // BATCH_SIZE

    try:
        with torch.no_grad():
            for batch_number, start in enumerate(range(0, len(games), BATCH_SIZE), start=1):
                batch = games[start:start + BATCH_SIZE]
                max_length = max(len(game.tokens) for game in batch)
                cpu_tokens = torch.zeros(
                    (len(batch), max_length),
                    dtype=torch.long,
                    pin_memory=use_cuda,
                )

                batch_rows = []
                boundary_positions = []
                for row, game in enumerate(batch):
                    cpu_tokens[row, :len(game.tokens)] = torch.tensor(game.tokens, dtype=torch.long)
                    batch_rows.extend([row] * len(game.boundary_positions))
                    boundary_positions.extend(game.boundary_positions.tolist())

                input_tokens = cpu_tokens.to(device, non_blocking=use_cuda)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    model(input_tokens)

                row_indices = torch.tensor(batch_rows, dtype=torch.long, device=device)
                position_indices = torch.tensor(boundary_positions, dtype=torch.long, device=device)
                layer_activations = torch.stack(
                    [
                        output[row_indices, position_indices]
                        for output in captured_outputs
                    ],
                    dim=1,
                )
                activation_batches.append(layer_activations.to(dtype=torch.float16).cpu().numpy())
                label_batches.append(np.concatenate([game.labels for game in batch]))
                ply_batches.append(np.concatenate([game.plies for game in batch]))
                game_id_batches.append(
                    np.concatenate(
                        [
                            np.full(len(game.plies), game.game_id, dtype=np.int32)
                            for game in batch
                        ]
                    )
                )

                total_boundaries += len(boundary_positions)
                captured_outputs[:] = [None] * config.n_layer
                del input_tokens, layer_activations

                if batch_number % 10 == 0 or batch_number == total_batches:
                    gpu_memory = (
                        torch.cuda.memory_allocated(device) / 1024**3
                        if use_cuda
                        else 0.0
                    )
                    print(
                        f"Batch {batch_number}/{total_batches} | "
                        f"boundaries {total_boundaries:,} | "
                        f"GPU memory {gpu_memory:.2f} GiB"
                    )
    finally:
        for hook in hooks:
            hook.remove()

    activations = np.concatenate(activation_batches).astype(np.float16, copy=False)
    labels = np.concatenate(label_batches).astype(np.int8, copy=False)
    plies = np.concatenate(ply_batches).astype(np.int16, copy=False)
    game_ids = np.concatenate(game_id_batches).astype(np.int32, copy=False)

    output_dir = Path(output_dir) if output_dir else root_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    np.savez_compressed(
        output_path,
        activations=activations,
        labels=labels,
        plies=plies,
        game_ids=game_ids,
    )

    print(f"Total move boundaries collected: {len(activations):,}")
    print(f"Total games processed: {len(games):,}")
    print(f"activations shape: {activations.shape}")
    print(f"labels shape: {labels.shape}")
    print(f"plies shape: {plies.shape}")
    print(f"game_ids shape: {game_ids.shape}")
    print(f"Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Extract MoveFormer activations for probing")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    extract_activations(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
