import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from evaluate_legality import evaluate_legality


CHARS = [
    "\0", "\1", " ", "#", "+", "-", "1", "2", "3", "4", "5", "6",
    "7", "8", "=", "B", "K", "N", "O", "Q", "R", "a", "b", "c",
    "d", "e", "f", "g", "h", "x",
]
STOI = {char: index for index, char in enumerate(CHARS)}
ITOS = {index: char for char, index in STOI.items()}
LONG_GAME = "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1"
LONG_GAME_2 = "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O Bd6 Re1"
MULTI_DEPTH_GAME = (
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 "
    "c3 O-O h3 Nb8 d4 Nbd7 c4 c6 Nc3"
)
SHORT_GAME = "e4 e5 Nf3"


class ScriptedModel(nn.Module):
    def __init__(self, move, never_terminate: bool = False):
        super().__init__()
        self.move = move
        self.never_terminate = never_terminate
        self.forward_calls = 0
        self.batch_sizes = []

    def forward(self, idx):
        self.forward_calls += 1
        self.batch_sizes.append(idx.shape[0])
        logits = torch.full(
            (idx.shape[0], idx.shape[1], len(CHARS)),
            -1000.0,
            device=idx.device,
        )

        for row in range(idx.shape[0]):
            real_tokens = [token for token in idx[row].tolist() if token != STOI["\0"]]
            text = "".join(ITOS[token] for token in real_tokens)
            context, suffix = text.rsplit(" ", 1)
            context_ply = len(context.lstrip("\1").split())
            move = self.move[context_ply] if isinstance(self.move, dict) else self.move
            next_character = "R" if self.never_terminate else move[len(suffix)]
            logits[row, len(real_tokens) - 1, STOI[next_character]] = 1000.0

        return logits, None


class EvaluateLegalityTests(unittest.TestCase):
    def evaluate(self, contents, model, k_values=(1, 5, 10), batch_size=1, ply_values=(10,), block_size=64):
        with tempfile.TemporaryDirectory() as temp_directory:
            val_path = Path(temp_directory) / "val.txt"
            val_path.write_text(contents, encoding="utf-8")

            output = io.StringIO()
            def sample_indices(high, size):
                return torch.arange(size[0]) % high

            with patch("evaluate_legality.torch.randint", side_effect=sample_indices):
                with redirect_stdout(output):
                    rates = evaluate_legality(
                        model,
                        val_path,
                        STOI,
                        ITOS,
                        block_size=block_size,
                        device="cpu",
                        k_values=list(k_values),
                        n_batches=1,
                        batch_size=batch_size,
                        ply_values=list(ply_values),
                    )
            return rates, output.getvalue()

    def test_reports_legal_move_for_every_k_and_restores_training_mode(self):
        model = ScriptedModel("Re1 ")
        model.train()

        rates, output = self.evaluate(
            LONG_GAME + "\n" + LONG_GAME_2 + "\n",
            model,
            batch_size=2,
        )

        self.assertEqual(rates, {10: {1: 100.0, 5: 100.0, 10: 100.0}})
        self.assertTrue(model.training)
        self.assertEqual(model.forward_calls, 4)
        self.assertEqual(model.batch_sizes, [6, 6, 6, 6])
        self.assertIn("Top-k move legality", output)
        self.assertIn("100.00%", output)

    def test_illegal_and_unterminated_moves_score_zero(self):
        illegal_rates, _ = self.evaluate(LONG_GAME + "\n", ScriptedModel("e5 "))
        unterminated_rates, _ = self.evaluate(
            LONG_GAME + "\n",
            ScriptedModel("R", never_terminate=True),
        )

        self.assertEqual(illegal_rates, {10: {1: 0.0, 5: 0.0, 10: 0.0}})
        self.assertEqual(unterminated_rates, {10: {1: 0.0, 5: 0.0, 10: 0.0}})

    def test_short_games_are_prefiltered_without_reducing_sample_count(self):
        model = ScriptedModel("Re1 ")
        rates, output = self.evaluate(
            SHORT_GAME + "\n" + LONG_GAME + "\n",
            model,
            k_values=(1,),
            batch_size=2,
        )

        self.assertEqual(rates, {10: {1: 100.0}})
        self.assertIn("10      1     2         2", output)
        self.assertNotIn("Skipped", output)

    def test_evaluates_all_default_ply_depths(self):
        model = ScriptedModel({6: "Ba4 ", 10: "Re1 ", 14: "c3 ", 18: "d4 ", 22: "Nc3 "})

        rates, output = self.evaluate(
            MULTI_DEPTH_GAME + "\n",
            model,
            k_values=(1,),
            ply_values=(6, 10, 14, 18, 22),
            block_size=256,
        )

        self.assertEqual(
            rates,
            {
                6: {1: 100.0},
                10: {1: 100.0},
                14: {1: 100.0},
                18: {1: 100.0},
                22: {1: 100.0},
            },
        )
        self.assertIn("22      1", output)


if __name__ == "__main__":
    unittest.main()
