import argparse
import csv
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn

from model import MoveFormerConfig, MoveFormerModel
from project_utils import find_best_checkpoint, iter_games, read_checkpoint, root_dir


RANDOM_STATE = 42
BOS_TOKEN = "\1"
SCALES = (0.5, 1.0, 2.0)
PRIMARY_SCALE = 1.0
MAX_MOVE_CHARS = 16
PLY_RANGE = (15, 45)
NON_KING_CLASSES = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)
EXAMPLE_CSV = "causal_empty_intervention_examples_v4.csv"
EXAMPLE_NPZ = "causal_empty_intervention_examples_v4.npz"
SUMMARY_CSV = "causal_empty_intervention_summary_v4.csv"
SUMMARY_NPZ = "causal_empty_intervention_summary_v4.npz"
EXPECTED_CONFIG = {
    "n_embd": 384,
    "n_layer": 6,
    "block_size": 768,
    "vocab_size": 30,
}


@dataclass
class CandidatePosition:
    context: list[int]
    board: chess.Board
    game_id: int
    ply: int
    ground_truth_move: chess.Move


@dataclass
class InterventionPosition(CandidatePosition):
    piece_class: int
    source_square: int
    control_piece_class: int
    control_square: int
    greedy_move: chess.Move


@dataclass(frozen=True)
class CentroidSpec:
    layer: int
    square: int
    piece_class: int
    direction: torch.Tensor
    piece_target: torch.Tensor
    empty_target: torch.Tensor
    gap: torch.Tensor
    piece_support: int
    empty_support: int


@dataclass
class GeometryDiagnostics:
    pre_progress: np.ndarray
    post_progress: np.ndarray
    edit_fraction: np.ndarray
    step_norm: np.ndarray
    reference_pre_progress: np.ndarray
    reference_post_progress: np.ndarray


def piece_name(piece_class):
    if piece_class == 0:
        return "empty"
    color = "white" if piece_class <= 6 else "black"
    piece_type = piece_class if piece_class <= 6 else piece_class - 6
    return f"{color} {chess.piece_name(piece_type)}"


def board_piece_class(piece):
    if piece is None:
        return 0
    return piece.piece_type if piece.color == chess.WHITE else piece.piece_type + 6


def move_involves_square(board, move, square):
    if square in (move.from_square, move.to_square):
        return True
    if board.is_en_passant(move):
        captured_square = move.to_square + (-8 if board.turn else 8)
        if square == captured_square:
            return True
    if board.is_castling(move):
        rank = 0 if board.turn == chess.WHITE else 7
        if board.is_kingside_castling(move):
            return square in (chess.square(7, rank), chess.square(5, rank))
        return square in (chess.square(0, rank), chess.square(3, rank))
    return False


def _context_ids(moves, ply, stoi, block_size):
    text = BOS_TOKEN + " ".join(moves[:ply]) + " "
    return [stoi[character] for character in text][-block_size:]


def _load_transformer(device):
    checkpoint_path = find_best_checkpoint()
    checkpoint = read_checkpoint(checkpoint_path)
    saved_config = checkpoint["config"]
    config = MoveFormerConfig(**saved_config["model"])
    for name, expected in EXPECTED_CONFIG.items():
        actual = getattr(config, name)
        if actual != expected:
            raise ValueError(
                f"Expected {name}={expected}, found {actual} in {checkpoint_path.name}"
            )

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
    stoi = saved_config["stoi"]
    itos = {index: character for character, index in stoi.items()}
    print(f"Loaded transformer checkpoint: {checkpoint_path.name}")
    return model, config, stoi, itos


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for greedy-plan discovery and causal interventions"
        )
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    return device


def sample_candidate_positions(games, stoi, block_size, pool_size=40000):
    rng = np.random.default_rng(RANDOM_STATE)
    reservoir = []
    seen = 0
    for game_id, game_text in enumerate(games):
        moves = game_text.split()
        maximum_ply = min(PLY_RANGE[1], len(moves) - 1)
        if maximum_ply < PLY_RANGE[0]:
            continue
        ply = int(rng.integers(PLY_RANGE[0], maximum_ply + 1))
        board = chess.Board()
        try:
            for san in moves[:ply]:
                board.push_san(san)
            ground_truth_move = board.parse_san(moves[ply])
            context = _context_ids(moves, ply, stoi, block_size)
        except (ValueError, KeyError):
            continue

        candidate = CandidatePosition(
            context=context,
            board=board.copy(stack=False),
            game_id=game_id,
            ply=ply,
            ground_truth_move=ground_truth_move,
        )
        seen += 1
        if len(reservoir) < pool_size:
            reservoir.append(candidate)
        else:
            replacement = int(rng.integers(seen))
            if replacement < pool_size:
                reservoir[replacement] = candidate
    rng.shuffle(reservoir)
    return reservoir


def generate_greedy_moves(
    model,
    samples,
    stoi,
    itos,
    block_size,
    device,
    label,
    batch_size=50,
):
    moves = [None] * len(samples)
    ordered_indices = sorted(range(len(samples)), key=lambda index: len(samples[index].context))
    with torch.inference_mode():
        for start in range(0, len(ordered_indices), batch_size):
            batch_indices = ordered_indices[start : start + batch_size]
            batch = [samples[index] for index in batch_indices]
            generated, terminated = _generate_greedy_batch(
                model,
                [sample.context for sample in batch],
                stoi,
                itos,
                block_size,
                device,
            )
            for move_text, did_terminate, sample_index, sample in zip(
                generated, terminated, batch_indices, batch
            ):
                if not did_terminate:
                    continue
                try:
                    moves[sample_index] = sample.board.parse_san(move_text)
                except ValueError:
                    pass
            print(
                f"{label}: "
                f"{min(start + batch_size, len(ordered_indices))}/{len(ordered_indices)}"
            )
    return moves


def _generate_greedy_batch(model, contexts, stoi, itos, block_size, device):
    row_count = len(contexts)
    if row_count == 0:
        return [], []

    pad_index = stoi["\0"]
    bos_index = stoi[BOS_TOKEN]
    space_index = stoi[" "]
    tokens = torch.full(
        (row_count, block_size),
        pad_index,
        dtype=torch.long,
        device=device,
    )
    lengths = torch.tensor(
        [len(context) for context in contexts],
        dtype=torch.long,
        device=device,
    )
    for row, context in enumerate(contexts):
        tokens[row, : len(context)] = torch.tensor(
            context, dtype=torch.long, device=device
        )

    active = torch.ones(row_count, dtype=torch.bool, device=device)
    terminated = torch.zeros(row_count, dtype=torch.bool, device=device)
    generated_tokens = torch.full(
        (row_count, MAX_MOVE_CHARS),
        pad_index,
        dtype=torch.long,
        device=device,
    )
    generated_lengths = torch.zeros(row_count, dtype=torch.long, device=device)
    device_type = torch.device(device).type
    use_bf16 = device_type == "cuda" and torch.cuda.is_bf16_supported()

    for step in range(MAX_MOVE_CHARS):
        active_rows = torch.nonzero(active, as_tuple=False).squeeze(1)
        if active_rows.numel() == 0:
            break
        active_lengths = lengths[active_rows]
        current_width = int(active_lengths.max().item())
        model_input = tokens[active_rows, :current_width]
        with torch.autocast(
            device_type=device_type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits, _ = model(model_input)
        batch_rows = torch.arange(active_rows.numel(), device=device)
        next_logits = logits[batch_rows, active_lengths - 1].float()
        next_logits[:, pad_index] = float("-inf")
        next_logits[:, bos_index] = float("-inf")
        next_tokens = torch.argmax(next_logits, dim=-1)

        ending = next_tokens == space_index
        ending_rows = active_rows[ending]
        terminated[ending_rows] = True
        active[ending_rows] = False

        continuing_rows = active_rows[~ending]
        continuing_tokens = next_tokens[~ending]
        if continuing_rows.numel() == 0:
            continue
        generated_tokens[continuing_rows, step] = continuing_tokens
        generated_lengths[continuing_rows] += 1

        full = lengths[continuing_rows] == block_size
        full_rows = continuing_rows[full]
        if full_rows.numel():
            tokens[full_rows, :-1] = tokens[full_rows, 1:].clone()
            tokens[full_rows, -1] = continuing_tokens[full]
        growing_rows = continuing_rows[~full]
        if growing_rows.numel():
            tokens[growing_rows, lengths[growing_rows]] = continuing_tokens[~full]
            lengths[growing_rows] += 1

    generated_cpu = generated_tokens.cpu()
    generated_length_cpu = generated_lengths.cpu().tolist()
    generated = [
        "".join(
            itos[int(token)]
            for token in generated_cpu[row, :generated_length_cpu[row]]
        )
        for row in range(row_count)
    ]
    return generated, terminated.cpu().tolist()


def select_control_square(board, greedy_move, game_id=0, ply=0):
    source_piece = board.piece_at(greedy_move.from_square)
    if source_piece is None:
        return None
    candidates = []
    for square, piece in board.piece_map().items():
        if piece.color != source_piece.color or piece.piece_type == chess.KING:
            continue
        if move_involves_square(board, greedy_move, square):
            continue
        candidates.append((square, piece))
    if not candidates:
        return None

    same_type = [
        (square, piece)
        for square, piece in candidates
        if piece.piece_type == source_piece.piece_type
    ]
    pool = sorted(same_type or candidates, key=lambda item: item[0])
    deterministic_index = (
        game_id * 1315423911 + ply * 2654435761 + greedy_move.from_square
    ) % len(pool)
    square, piece = pool[deterministic_index]
    return square, board_piece_class(piece)


def identify_intervention_positions(
    candidates,
    model,
    stoi,
    itos,
    block_size,
    device,
    batch_size=50,
):
    greedy_moves = generate_greedy_moves(
        model,
        candidates,
        stoi,
        itos,
        block_size,
        device,
        "Discovering greedy plans",
        batch_size,
    )
    positions = []
    for candidate, greedy_move in zip(candidates, greedy_moves):
        if greedy_move is None:
            continue
        piece = candidate.board.piece_at(greedy_move.from_square)
        piece_class = board_piece_class(piece)
        if piece_class not in NON_KING_CLASSES:
            continue
        control = select_control_square(
            candidate.board, greedy_move, candidate.game_id, candidate.ply
        )
        if control is None:
            continue
        control_square, control_piece_class = control
        positions.append(
            InterventionPosition(
                context=candidate.context,
                board=candidate.board,
                game_id=candidate.game_id,
                ply=candidate.ply,
                ground_truth_move=candidate.ground_truth_move,
                piece_class=piece_class,
                source_square=greedy_move.from_square,
                control_piece_class=control_piece_class,
                control_square=control_square,
                greedy_move=greedy_move,
            )
        )
    return positions


def _mean_activations(activations, indices, batch_size=65536):
    if len(indices) == 0:
        raise ValueError("Cannot compute centroid from zero samples")
    total = np.zeros(activations.shape[1:], dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        total += np.asarray(activations[batch_indices], dtype=np.float32).sum(
            axis=0, dtype=np.float64
        )
    return (total / len(indices)).astype(np.float32)


def estimate_centroid_specs(
    activations,
    labels,
    target_keys,
    device,
    min_support=100,
    indices=None,
):
    if indices is None:
        indices = np.arange(len(labels))
    indices = np.asarray(indices, dtype=np.int64)
    selected_labels = labels[indices]
    empty_cache = {}
    specs = {}

    for piece_class, square in sorted(target_keys):
        if square not in empty_cache:
            empty_local = np.flatnonzero(selected_labels[:, square] == 0)
            if len(empty_local) >= min_support:
                empty_indices = indices[empty_local]
                empty_cache[square] = (
                    _mean_activations(activations, empty_indices),
                    len(empty_indices),
                )
            else:
                empty_cache[square] = None
        empty_entry = empty_cache[square]
        piece_local = np.flatnonzero(selected_labels[:, square] == piece_class)
        if empty_entry is None or len(piece_local) < min_support:
            continue

        empty_centroid, empty_support = empty_entry
        piece_indices = indices[piece_local]
        piece_centroid = _mean_activations(activations, piece_indices)
        layer_specs = []
        supported = True
        for layer in range(activations.shape[1]):
            difference = empty_centroid[layer] - piece_centroid[layer]
            gap = float(np.linalg.norm(difference))
            if not np.isfinite(gap) or gap <= 1e-8:
                supported = False
                break
            direction = torch.tensor(
                difference / gap, dtype=torch.float32, device=device
            )
            piece_vector = torch.tensor(
                piece_centroid[layer], dtype=torch.float32, device=device
            )
            empty_vector = torch.tensor(
                empty_centroid[layer], dtype=torch.float32, device=device
            )
            layer_specs.append(
                CentroidSpec(
                    layer=layer,
                    square=square,
                    piece_class=piece_class,
                    direction=direction,
                    piece_target=torch.dot(piece_vector, direction),
                    empty_target=torch.dot(empty_vector, direction),
                    gap=torch.tensor(gap, dtype=torch.float32, device=device),
                    piece_support=len(piece_indices),
                    empty_support=empty_support,
                )
            )
        if supported:
            specs[(piece_class, square)] = tuple(layer_specs)
    return specs


def centroid_progress(activations, spec):
    projection = (activations.float() * spec.direction).sum(dim=-1)
    return (projection - spec.piece_target) / spec.gap


def apply_centroid_edit(activations, spec, scale):
    values = activations.float()
    projection = (values * spec.direction).sum(dim=-1, keepdim=True)
    return values + scale * (spec.empty_target - projection) * spec.direction


class MultiLayerCentroidIntervention(nn.Module, AbstractContextManager):
    def __init__(
        self,
        model,
        specs,
        scale,
        space_token,
        reference_specs=None,
        record_geometry=False,
    ):
        super().__init__()
        self.model = model
        self.specs = tuple(specs)
        self.scale = float(scale)
        self.space_token = int(space_token)
        self.reference_specs = tuple(reference_specs or specs)
        self.record_geometry = record_geometry
        self.boundary_positions = None
        self.handles = []
        self.geometry = {}
        self.row_reference_specs = None
        if len(self.specs) != len(model.blocks):
            raise ValueError("One centroid specification is required for every layer")
        if [spec.layer for spec in self.specs] != list(range(len(model.blocks))):
            raise ValueError("Centroid specifications must be ordered by layer")
        if len(self.reference_specs) != len(self.specs):
            raise ValueError("Reference specifications must match intervention layers")

    def _make_hook(self, spec, reference_spec):
        def hook(_module, _inputs, output):
            rows = torch.arange(output.shape[0], device=output.device)
            selected = output[rows, self.boundary_positions].float()
            modified = apply_centroid_edit(selected, spec, self.scale)
            if self.record_geometry:
                before_projection = (
                    selected * spec.direction
                ).sum(dim=-1)
                after_projection = (
                    modified * spec.direction
                ).sum(dim=-1)
                denominator = spec.empty_target - before_projection
                edit_fraction = torch.where(
                    denominator.abs() > 1e-8,
                    (after_projection - before_projection) / denominator,
                    torch.full_like(denominator, torch.nan),
                )
                if self.row_reference_specs is None:
                    reference_pre = centroid_progress(selected, reference_spec)
                    reference_post = centroid_progress(modified, reference_spec)
                else:
                    row_specs = [
                        specs[spec.layer] for specs in self.row_reference_specs
                    ]
                    reference_directions = torch.stack(
                        [row_spec.direction for row_spec in row_specs]
                    )
                    reference_piece_targets = torch.stack(
                        [row_spec.piece_target for row_spec in row_specs]
                    )
                    reference_gaps = torch.stack(
                        [row_spec.gap for row_spec in row_specs]
                    )
                    reference_pre = (
                        (selected * reference_directions).sum(dim=-1)
                        - reference_piece_targets
                    ) / reference_gaps
                    reference_post = (
                        (modified * reference_directions).sum(dim=-1)
                        - reference_piece_targets
                    ) / reference_gaps
                self.geometry[spec.layer] = {
                    "pre_progress": centroid_progress(selected, spec).detach().cpu(),
                    "post_progress": centroid_progress(modified, spec).detach().cpu(),
                    "edit_fraction": edit_fraction.detach().cpu(),
                    "step_norm": (modified - selected).norm(dim=-1).detach().cpu(),
                    "reference_pre_progress": reference_pre.detach().cpu(),
                    "reference_post_progress": reference_post.detach().cpu(),
                }
            intervened = output.clone()
            intervened[rows, self.boundary_positions] = modified.to(output.dtype)
            return intervened

        return hook

    def __enter__(self):
        if self.handles:
            raise RuntimeError("Intervention hooks are already registered")
        self.handles = [
            self.model.blocks[spec.layer].register_forward_hook(
                self._make_hook(spec, reference_spec)
            )
            for spec, reference_spec in zip(self.specs, self.reference_specs)
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.boundary_positions = None
        self.row_reference_specs = None
        return False

    def forward(self, idx, targets=None):
        space_mask = idx == self.space_token
        if not bool(space_mask.any(dim=1).all()):
            raise ValueError("Every context must contain a move-boundary space token")
        positions = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0)
        self.boundary_positions = (space_mask * positions).max(dim=1).values
        return self.model(idx, targets)


def _condition_keys(sample, condition):
    source_key = sample.piece_class, sample.source_square
    if condition == "treatment":
        return source_key, source_key
    if condition == "control":
        return (sample.control_piece_class, sample.control_square), source_key
    raise ValueError(f"Unknown condition: {condition}")


def group_indices(samples, condition, include_reference=True):
    groups = {}
    for index, sample in enumerate(samples):
        intervention_key, reference_key = _condition_keys(sample, condition)
        key = (
            (intervention_key, reference_key)
            if include_reference
            else intervention_key
        )
        groups.setdefault(key, []).append(index)
    return groups


def select_supported_positions(positions, specs, per_piece):
    selected = {piece_class: [] for piece_class in NON_KING_CLASSES}
    for sample in positions:
        treatment_key = (sample.piece_class, sample.source_square)
        control_key = (sample.control_piece_class, sample.control_square)
        if treatment_key not in specs or control_key not in specs:
            continue
        bucket = selected[sample.piece_class]
        if len(bucket) < per_piece:
            bucket.append(sample)
    shortages = {
        piece_name(piece_class): len(samples)
        for piece_class, samples in selected.items()
        if len(samples) < per_piece
    }
    if shortages:
        raise RuntimeError(
            f"Only found supported positions for {shortages}; increase "
            "--candidate-pool or reduce --per-piece/--min-centroid-support."
        )
    return [
        sample
        for piece_class in NON_KING_CLASSES
        for sample in selected[piece_class]
    ]


def generate_condition_moves(
    model,
    samples,
    condition,
    specs,
    scale,
    stoi,
    itos,
    block_size,
    device,
    batch_size=50,
):
    parsed_moves = [None] * len(samples)
    groups = group_indices(samples, condition, include_reference=False)
    for group_number, (intervention_key, indices) in enumerate(
        groups.items(), start=1
    ):
        group_samples = [samples[index] for index in indices]
        wrapper = MultiLayerCentroidIntervention(
            model,
            specs[intervention_key],
            scale,
            stoi[" "],
        )
        with wrapper:
            group_moves = generate_greedy_moves(
                wrapper,
                group_samples,
                stoi,
                itos,
                block_size,
                device,
                f"{condition} scale {scale:g} group {group_number}/{len(groups)}",
                batch_size,
            )
        for local_index, global_index in enumerate(indices):
            parsed_moves[global_index] = group_moves[local_index]
    return parsed_moves


def capture_condition_geometry(
    model,
    samples,
    condition,
    specs,
    scale,
    stoi,
    device,
    batch_size=50,
):
    n_layers = len(model.blocks)
    shape = (len(samples), n_layers)
    arrays = {
        name: np.full(shape, np.nan, dtype=np.float32)
        for name in (
            "pre_progress",
            "post_progress",
            "edit_fraction",
            "step_norm",
            "reference_pre_progress",
            "reference_post_progress",
        )
    }
    groups = group_indices(samples, condition, include_reference=False)
    use_bf16 = torch.device(device).type == "cuda" and torch.cuda.is_bf16_supported()
    for intervention_key, indices in groups.items():
        group_samples = [samples[index] for index in indices]
        wrapper = MultiLayerCentroidIntervention(
            model,
            specs[intervention_key],
            scale,
            stoi[" "],
            record_geometry=True,
        )
        with wrapper, torch.inference_mode():
            for start in range(0, len(group_samples), batch_size):
                batch = group_samples[start : start + batch_size]
                width = max(len(sample.context) for sample in batch)
                tokens = torch.full(
                    (len(batch), width),
                    stoi["\0"],
                    dtype=torch.long,
                    device=device,
                )
                for row, sample in enumerate(batch):
                    tokens[row, : len(sample.context)] = torch.tensor(
                        sample.context, dtype=torch.long, device=device
                    )
                wrapper.row_reference_specs = [
                    specs[(sample.piece_class, sample.source_square)]
                    for sample in batch
                ]
                with torch.autocast(
                    device_type=torch.device(device).type,
                    dtype=torch.bfloat16,
                    enabled=use_bf16,
                ):
                    wrapper(tokens)
                global_indices = indices[start : start + len(batch)]
                for layer in range(n_layers):
                    for name in arrays:
                        arrays[name][global_indices, layer] = (
                            wrapper.geometry[layer][name].numpy()
                        )
    return GeometryDiagnostics(**arrays)


def _percentage(mask):
    return 100.0 * float(np.mean(mask)) if len(mask) else np.nan


def _legal_metrics(samples, moves, mask):
    chosen_samples = [sample for sample, keep in zip(samples, mask) if keep]
    chosen_moves = [move for move, keep in zip(moves, mask) if keep]
    legal = np.array([move is not None for move in chosen_moves], dtype=bool)
    retained = np.array(
        [
            move is not None and move == sample.greedy_move
            for sample, move in zip(chosen_samples, chosen_moves)
        ],
        dtype=bool,
    )
    source_usage = np.array(
        [
            move.from_square == sample.source_square
            for sample, move in zip(chosen_samples, chosen_moves)
            if move is not None
        ],
        dtype=bool,
    )
    forced_alternative = ~source_usage
    return {
        "legality": _percentage(legal),
        "plan_retention": _percentage(retained),
        "source_square_usage": _percentage(source_usage),
        "forced_alternative": _percentage(forced_alternative),
        "n_legal": int(legal.sum()),
    }


def summarize_paired(
    samples,
    treatment_moves,
    control_moves,
    treatment_geometry,
    control_geometry,
    scale,
):
    rows = []
    scopes = [("all", None)] + [
        (piece_name(piece_class), piece_class)
        for piece_class in NON_KING_CLASSES
    ]
    for scope_name, piece_class in scopes:
        mask = np.array(
            [
                piece_class is None or sample.piece_class == piece_class
                for sample in samples
            ],
            dtype=bool,
        )
        if not mask.any():
            continue
        treatment = _legal_metrics(samples, treatment_moves, mask)
        control = _legal_metrics(samples, control_moves, mask)
        row = {
            "scope": scope_name,
            "piece_class": -1 if piece_class is None else piece_class,
            "scale": scale,
            "is_primary_scale": scale == PRIMARY_SCALE,
            "n_examples": int(mask.sum()),
        }
        for name in (
            "plan_retention",
            "source_square_usage",
            "forced_alternative",
            "legality",
            "n_legal",
        ):
            row[f"treatment_{name}"] = treatment[name]
            row[f"control_{name}"] = control[name]
        row["specificity_plan_retention"] = (
            control["plan_retention"] - treatment["plan_retention"]
        )
        row["specificity_source_square_usage"] = (
            control["source_square_usage"] - treatment["source_square_usage"]
        )
        row["specificity_forced_alternative"] = (
            treatment["forced_alternative"] - control["forced_alternative"]
        )
        row["treatment_minus_control_legality"] = (
            treatment["legality"] - control["legality"]
        )
        for layer in range(treatment_geometry.pre_progress.shape[1]):
            layer_name = f"l{layer + 1}"
            row[f"treatment_pre_progress_{layer_name}"] = float(
                np.nanmean(treatment_geometry.pre_progress[mask, layer])
            )
            row[f"treatment_post_progress_{layer_name}"] = float(
                np.nanmean(treatment_geometry.post_progress[mask, layer])
            )
            row[f"treatment_progress_gt_half_{layer_name}"] = _percentage(
                treatment_geometry.post_progress[mask, layer] > 0.5
            )
            row[f"treatment_step_norm_{layer_name}"] = float(
                np.nanmean(treatment_geometry.step_norm[mask, layer])
            )
            row[f"control_pre_progress_{layer_name}"] = float(
                np.nanmean(control_geometry.pre_progress[mask, layer])
            )
            row[f"control_post_progress_{layer_name}"] = float(
                np.nanmean(control_geometry.post_progress[mask, layer])
            )
            row[f"control_source_pre_progress_{layer_name}"] = float(
                np.nanmean(control_geometry.reference_pre_progress[mask, layer])
            )
            row[f"control_source_post_progress_{layer_name}"] = float(
                np.nanmean(control_geometry.reference_post_progress[mask, layer])
            )
        rows.append(row)
    return rows


def _san_or_empty(board, move):
    return board.san(move) if move is not None else ""


def example_records(samples, condition, moves, geometry, scale):
    rows = []
    for index, (sample, move) in enumerate(zip(samples, moves)):
        legal = move is not None
        row = {
            "example_id": index,
            "game_id": sample.game_id,
            "ply": sample.ply,
            "piece_class": sample.piece_class,
            "piece": piece_name(sample.piece_class),
            "source_square": chess.square_name(sample.source_square),
            "control_piece_class": sample.control_piece_class,
            "control_square": chess.square_name(sample.control_square),
            "condition": condition,
            "scale": scale,
            "is_primary_scale": scale == PRIMARY_SCALE,
            "clean_greedy_move": _san_or_empty(sample.board, sample.greedy_move),
            "intervened_move": _san_or_empty(sample.board, move),
            "legal": legal,
            "plan_retention": legal and move == sample.greedy_move,
            "source_square_usage": legal
            and move.from_square == sample.source_square,
            "forced_alternative": legal
            and move.from_square != sample.source_square,
        }
        for layer in range(geometry.pre_progress.shape[1]):
            layer_name = f"l{layer + 1}"
            row[f"pre_progress_{layer_name}"] = geometry.pre_progress[index, layer]
            row[f"post_progress_{layer_name}"] = geometry.post_progress[index, layer]
            row[f"edit_fraction_{layer_name}"] = geometry.edit_fraction[index, layer]
            row[f"step_norm_{layer_name}"] = geometry.step_norm[index, layer]
            row[f"source_pre_progress_{layer_name}"] = (
                geometry.reference_pre_progress[index, layer]
            )
            row[f"source_post_progress_{layer_name}"] = (
                geometry.reference_post_progress[index, layer]
            )
        rows.append(row)
    return rows


def _save_rows(rows, csv_path, npz_path):
    if not rows:
        raise ValueError("Cannot save an empty result table")
    fields = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        npz_path,
        **{field: np.asarray([row[field] for row in rows]) for field in fields},
    )


def save_results(example_rows, summary_rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "example_csv": output_dir / EXAMPLE_CSV,
        "example_npz": output_dir / EXAMPLE_NPZ,
        "summary_csv": output_dir / SUMMARY_CSV,
        "summary_npz": output_dir / SUMMARY_NPZ,
    }
    _save_rows(example_rows, paths["example_csv"], paths["example_npz"])
    _save_rows(summary_rows, paths["summary_csv"], paths["summary_npz"])
    return paths


def print_summary(rows):
    print("\nCausal empty intervention summary")
    for row in rows:
        if row["scope"] != "all":
            continue
        print(
            f"scale={row['scale']:.1f} n={row['n_examples']} | "
            f"plan specificity={row['specificity_plan_retention']:.2f}pp | "
            f"source specificity={row['specificity_source_square_usage']:.2f}pp | "
            f"forced-alt specificity={row['specificity_forced_alternative']:.2f}pp | "
            f"legality T-C={row['treatment_minus_control_legality']:.2f}pp"
        )


def _centroid_indices(output_dir, n_rows):
    split_path = output_dir / "probe_split.npz"
    if not split_path.exists():
        split_path = root_dir() / "probe_split.npz"
    if not split_path.exists():
        print("probe_split.npz not found; using all activation rows for centroids")
        return np.arange(n_rows)
    with np.load(split_path) as split:
        indices = np.asarray(split["test_indices"], dtype=np.int64)
    if len(indices) == 0 or indices.min() < 0 or indices.max() >= n_rows:
        raise ValueError(
            f"{split_path.name} does not match activations.npz; retrain probes "
            "after extracting activations"
        )
    print(f"Using {len(indices):,} held-out probe rows for centroid estimates")
    return indices


def run_interventions(
    per_piece=400,
    candidate_pool=40000,
    batch_size=50,
    discovery_batch_size=2048,
    min_centroid_support=100,
    output_dir=None,
):
    device = require_cuda()
    output_dir = Path(output_dir) if output_dir else root_dir()
    model, config, stoi, itos = _load_transformer(device)

    activation_path = output_dir / "activations.npz"
    if not activation_path.exists():
        activation_path = root_dir() / "activations.npz"
    with np.load(activation_path) as data:
        activations = np.asarray(data["activations"])
        labels = np.asarray(data["labels"])
    if activations.shape[1] != config.n_layer:
        raise ValueError(
            f"Activation layers {activations.shape[1]} != model layers {config.n_layer}"
        )
    centroid_indices = _centroid_indices(output_dir, len(labels))

    candidates = sample_candidate_positions(
        iter_games(root_dir() / "data" / "val.txt"),
        stoi,
        config.block_size,
        candidate_pool,
    )
    discovered = identify_intervention_positions(
        candidates,
        model,
        stoi,
        itos,
        config.block_size,
        device,
        discovery_batch_size,
    )
    target_keys = {
        key
        for sample in discovered
        for key in (
            (sample.piece_class, sample.source_square),
            (sample.control_piece_class, sample.control_square),
        )
    }
    specs = estimate_centroid_specs(
        activations,
        labels,
        target_keys,
        device,
        min_centroid_support,
        centroid_indices,
    )
    samples = select_supported_positions(discovered, specs, per_piece)
    print(
        f"Selected {len(samples):,} examples; "
        f"{len(specs):,}/{len(target_keys):,} target keys have all-layer centroids"
    )
    del activations, labels

    example_rows = []
    summary_rows = []
    for scale in SCALES:
        treatment_geometry = capture_condition_geometry(
            model,
            samples,
            "treatment",
            specs,
            scale,
            stoi,
            device,
            batch_size,
        )
        control_geometry = capture_condition_geometry(
            model,
            samples,
            "control",
            specs,
            scale,
            stoi,
            device,
            batch_size,
        )
        treatment_moves = generate_condition_moves(
            model,
            samples,
            "treatment",
            specs,
            scale,
            stoi,
            itos,
            config.block_size,
            device,
            batch_size,
        )
        control_moves = generate_condition_moves(
            model,
            samples,
            "control",
            specs,
            scale,
            stoi,
            itos,
            config.block_size,
            device,
            batch_size,
        )
        example_rows.extend(
            example_records(
                samples, "treatment", treatment_moves, treatment_geometry, scale
            )
        )
        example_rows.extend(
            example_records(
                samples, "control", control_moves, control_geometry, scale
            )
        )
        summary_rows.extend(
            summarize_paired(
                samples,
                treatment_moves,
                control_moves,
                treatment_geometry,
                control_geometry,
                scale,
            )
        )

    print_summary(summary_rows)
    paths = save_results(example_rows, summary_rows, output_dir)
    for path in paths.values():
        print(f"Saved: {path}")
    return example_rows, summary_rows


def main():
    parser = argparse.ArgumentParser(
        description="Run all-layer causal empty interventions"
    )
    parser.add_argument("--per-piece", type=int, default=400)
    parser.add_argument("--candidate-pool", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--discovery-batch-size", type=int, default=2048)
    parser.add_argument("--min-centroid-support", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run_interventions(
        per_piece=args.per_piece,
        candidate_pool=args.candidate_pool,
        batch_size=args.batch_size,
        discovery_batch_size=args.discovery_batch_size,
        min_centroid_support=args.min_centroid_support,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
