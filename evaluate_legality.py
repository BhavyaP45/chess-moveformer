from pathlib import Path

import chess
import torch


MAX_MOVE_CHARS = 16
BOS_TOKEN = "\1"
PAD_TOKEN = "\0"


@torch.no_grad()
def evaluate_legality(model, val_path, stoi, itos, block_size, device, k_values=[1, 5, 10], n_batches=100, batch_size=64):
    """Measure how often top-k character sampling produces a legal SAN move."""
    with Path(val_path).open("r", encoding="utf-8") as file:
        games = tuple(game for line in file if (game := line.rstrip("\r\n")))

    legal_counts = {k: 0 for k in k_values}
    evaluated = 0
    skipped = 0
    was_training = model.training
    model.eval()

    try:
        for _ in range(n_batches):
            game_indices = torch.randint(len(games), (batch_size,))

            for game_index in game_indices:
                moves = games[game_index.item()].split()
                if len(moves) < 10:
                    skipped += 1
                    continue

                board = chess.Board()
                for san in moves[:10]:
                    board.push_san(san)

                context = BOS_TOKEN + " ".join(moves[:10]) + " "
                context_ids = [stoi[char] for char in context][-block_size:]
                context_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
                evaluated += 1

                for k in k_values:
                    generated = context_tensor
                    generated_move = ""
                    terminated = False

                    for _ in range(MAX_MOVE_CHARS):
                        logits, _ = model(generated[:, -block_size:])
                        next_logits = logits[:, -1, :].clone()
                        next_logits[:, stoi[PAD_TOKEN]] = float("-inf")
                        next_logits[:, stoi[BOS_TOKEN]] = float("-inf")

                        top_values, top_indices = torch.topk(next_logits, k, dim=-1)
                        probabilities = torch.softmax(top_values, dim=-1)
                        sampled_rank = torch.multinomial(probabilities, num_samples=1)
                        next_token = top_indices.gather(-1, sampled_rank)
                        next_character = itos[next_token.item()]

                        if next_character == " ":
                            terminated = True
                            break

                        generated_move += next_character
                        generated = torch.cat((generated, next_token), dim=1)

                    if not terminated:
                        continue

                    try:
                        board.parse_san(generated_move)
                        legal_counts[k] += 1
                    except ValueError:
                        pass
    finally:
        model.train(was_training)

    legality_rates = {
        k: 100.0 * legal_counts[k] / evaluated if evaluated else 0.0
        for k in k_values
    }

    print("Top-k move legality")
    print(f"{'k':<6}{'Legal':<10}{'Evaluated':<12}{'Skipped':<10}{'Rate':>8}")
    for k in k_values:
        print(f"{k:<6}{legal_counts[k]:<10}{evaluated:<12}{skipped:<10}{legality_rates[k]:>7.2f}%")

    return legality_rates
