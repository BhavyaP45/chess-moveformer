from pathlib import Path

import chess
import torch


MAX_MOVE_CHARS = 16
BOS_TOKEN = "\1"
PAD_TOKEN = "\0"


def _generate_moves(model, contexts, stoi, itos, block_size, device, k_values):
    pad_index = stoi[PAD_TOKEN]
    bos_index = stoi[BOS_TOKEN]
    expanded_contexts = []
    board_indices = []
    row_k_values = []

    for board_index, context in enumerate(contexts):
        for k in k_values:
            expanded_contexts.append(context)
            board_indices.append(board_index)
            row_k_values.append(k)

    row_count = len(expanded_contexts)
    tokens = torch.full((row_count, block_size), pad_index, dtype=torch.long, device=device)
    lengths = torch.tensor([len(context) for context in expanded_contexts], device=device)
    k_tensor = torch.tensor(row_k_values, device=device)

    for row, context in enumerate(expanded_contexts):
        tokens[row, :len(context)] = torch.tensor(context, dtype=torch.long, device=device)

    active = torch.ones(row_count, dtype=torch.bool, device=device)
    terminated = torch.zeros(row_count, dtype=torch.bool, device=device)
    generated_moves = [""] * row_count
    max_k = max(k_values)
    device_type = torch.device(device).type
    use_bf16 = device_type == "cuda" and torch.cuda.is_bf16_supported()

    for _ in range(MAX_MOVE_CHARS):
        active_rows = torch.nonzero(active, as_tuple=False).squeeze(1)
        if active_rows.numel() == 0:
            break

        active_lengths = lengths[active_rows]
        current_width = int(active_lengths.max().item())
        model_input = tokens[active_rows, :current_width]

        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):
            logits, _ = model(model_input)

        batch_rows = torch.arange(active_rows.numel(), device=device)
        next_logits = logits[batch_rows, active_lengths - 1, :].float()
        next_logits[:, pad_index] = float("-inf")
        next_logits[:, bos_index] = float("-inf")

        top_values, top_indices = torch.topk(next_logits, max_k, dim=-1)
        ranks = torch.arange(max_k, device=device).unsqueeze(0)
        allowed_ranks = ranks < k_tensor[active_rows].unsqueeze(1)
        top_values = top_values.masked_fill(~allowed_ranks, float("-inf"))
        probabilities = torch.softmax(top_values, dim=-1)
        sampled_ranks = torch.multinomial(probabilities, num_samples=1)
        next_tokens = top_indices.gather(-1, sampled_ranks).squeeze(1)

        active_row_list = active_rows.tolist()
        next_token_list = next_tokens.tolist()

        for row, next_token in zip(active_row_list, next_token_list):
            next_character = itos[next_token]
            if next_character == " ":
                active[row] = False
                terminated[row] = True
                continue

            generated_moves[row] += next_character
            length = int(lengths[row].item())
            if length == block_size:
                tokens[row, :-1] = tokens[row, 1:].clone()
                tokens[row, -1] = next_token
            else:
                tokens[row, length] = next_token
                lengths[row] += 1

    return generated_moves, terminated.tolist(), board_indices, row_k_values


@torch.no_grad()
def evaluate_legality(model, val_path, stoi, itos, block_size, device, k_values=[1, 5, 10], n_batches=100, batch_size=64, ply_values=[6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]):
    """Measure how often top-k character sampling produces a legal SAN move."""
    with Path(val_path).open("r", encoding="utf-8") as file:
        games = tuple(game.split() for line in file if (game := line.rstrip("\r\n")))

    eligible_games = {
        ply: tuple(game for game in games if len(game) >= ply)
        for ply in ply_values
    }
    legal_counts = {ply: {k: 0 for k in k_values} for ply in ply_values}
    evaluated_counts = {ply: 0 for ply in ply_values}
    was_training = model.training
    model.eval()

    try:
        for _ in range(n_batches):
            boards = []
            contexts = []
            context_plies = []

            for target_ply in ply_values:
                data = eligible_games[target_ply]
                game_indices = torch.randint(len(data), (batch_size,))

                for game_index in game_indices:
                    moves = data[game_index.item()]
                    board = chess.Board()
                    for san in moves[:target_ply]:
                        board.push_san(san)

                    context = BOS_TOKEN + " ".join(moves[:target_ply]) + " "
                    context_ids = [stoi[char] for char in context][-block_size:]
                    boards.append(board)
                    contexts.append(context_ids)
                    context_plies.append(target_ply)
                    evaluated_counts[target_ply] += 1

            generated_moves, terminated, board_indices, row_k_values = _generate_moves(
                model, contexts, stoi, itos, block_size, device, k_values
            )

            for generated_move, did_terminate, board_index, k in zip(
                generated_moves, terminated, board_indices, row_k_values
            ):
                if not did_terminate:
                    continue

                try:
                    boards[board_index].parse_san(generated_move)
                    ply = context_plies[board_index]
                    legal_counts[ply][k] += 1
                except ValueError:
                    pass
    finally:
        model.train(was_training)

    legality_rates = {
        ply: {
            k: 100.0 * legal_counts[ply][k] / evaluated_counts[ply]
            if evaluated_counts[ply]
            else 0.0
            for k in k_values
        }
        for ply in ply_values
    }

    print("Top-k move legality by context length")
    print(f"{'Plies':<8}{'k':<6}{'Legal':<10}{'Evaluated':<12}{'Rate':>8}")
    for ply in ply_values:
        for k in k_values:
            print(
                f"{ply:<8}{k:<6}{legal_counts[ply][k]:<10}"
                f"{evaluated_counts[ply]:<12}{legality_rates[ply][k]:>7.2f}%"
            )

    return legality_rates
