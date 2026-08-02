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
SHORT_GAME = "e4 e5 Nf3"


class ScriptedModel(nn.Module):
    def __init__(self, move: str, never_terminate: bool = False):
        super().__init__()
        self.move = move
        self.never_terminate = never_terminate

    def forward(self, idx):
        text = "".join(ITOS[token] for token in idx[0].tolist())
        suffix = text.rsplit(" ", 1)[-1]
        next_character = "R" if self.never_terminate else self.move[len(suffix)]
        logits = torch.full((1, idx.shape[1], len(CHARS)), -1000.0, device=idx.device)
        logits[0, -1, STOI[next_character]] = 1000.0
        return logits, None


class EvaluateLegalityTests(unittest.TestCase):
    def evaluate(self, contents, model, k_values=(1, 5, 10), batch_size=1):
        with tempfile.TemporaryDirectory() as temp_directory:
            val_path = Path(temp_directory) / "val.txt"
            val_path.write_text(contents, encoding="utf-8")
            sampled_indices = torch.arange(batch_size) % len(contents.splitlines())

            output = io.StringIO()
            with patch("evaluate_legality.torch.randint", return_value=sampled_indices):
                with redirect_stdout(output):
                    rates = evaluate_legality(
                        model,
                        val_path,
                        STOI,
                        ITOS,
                        block_size=64,
                        device="cpu",
                        k_values=list(k_values),
                        n_batches=1,
                        batch_size=batch_size,
                    )
            return rates, output.getvalue()

    def test_reports_legal_move_for_every_k_and_restores_training_mode(self):
        model = ScriptedModel("Re1 ")
        model.train()

        rates, output = self.evaluate(LONG_GAME + "\n", model)

        self.assertEqual(rates, {1: 100.0, 5: 100.0, 10: 100.0})
        self.assertTrue(model.training)
        self.assertIn("Top-k move legality", output)
        self.assertIn("100.00%", output)

    def test_illegal_and_unterminated_moves_score_zero(self):
        illegal_rates, _ = self.evaluate(LONG_GAME + "\n", ScriptedModel("e5 "))
        unterminated_rates, _ = self.evaluate(
            LONG_GAME + "\n",
            ScriptedModel("R", never_terminate=True),
        )

        self.assertEqual(illegal_rates, {1: 0.0, 5: 0.0, 10: 0.0})
        self.assertEqual(unterminated_rates, {1: 0.0, 5: 0.0, 10: 0.0})

    def test_short_games_are_excluded_and_reported(self):
        model = ScriptedModel("Re1 ")
        rates, output = self.evaluate(
            SHORT_GAME + "\n" + LONG_GAME + "\n",
            model,
            k_values=(1,),
            batch_size=2,
        )

        self.assertEqual(rates, {1: 100.0})
        self.assertIn("1         1           1", output)


if __name__ == "__main__":
    unittest.main()
