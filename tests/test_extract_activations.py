import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import chess
import numpy as np
import torch

import extract_activations
from model import MoveFormerConfig, MoveFormerModel


CHARS = [
    "\0", "\1", " ", "#", "+", "-", "1", "2", "3", "4", "5", "6",
    "7", "8", "=", "B", "K", "N", "O", "Q", "R", "a", "b", "c",
    "d", "e", "f", "g", "h", "x",
]
STOI = {char: index for index, char in enumerate(CHARS)}


class ExtractActivationsTests(unittest.TestCase):
    def test_board_labels_follow_documented_piece_classes(self):
        labels = extract_activations._board_labels(chess.Board())

        self.assertEqual(labels[chess.A1], 4)
        self.assertEqual(labels[chess.B1], 2)
        self.assertEqual(labels[chess.E1], 6)
        self.assertEqual(labels[chess.A2], 1)
        self.assertEqual(labels[chess.A7], 7)
        self.assertEqual(labels[chess.E8], 12)
        self.assertEqual(labels[chess.E4], 0)

    def test_prepare_game_aligns_space_boundaries_with_board_states(self):
        prepared = extract_activations._prepare_game("e4 e5 Nf3", 17, STOI, 768)

        self.assertIsNotNone(prepared)
        self.assertEqual(prepared.boundary_positions.tolist(), [3, 6])
        self.assertEqual(prepared.plies.tolist(), [1, 2])
        self.assertEqual(prepared.game_id, 17)
        self.assertEqual(prepared.labels[0, chess.E4], 1)
        self.assertEqual(prepared.labels[0, chess.E2], 0)
        self.assertEqual(prepared.labels[1, chess.E5], 7)
        self.assertIsNone(
            extract_activations._prepare_game("e4 e5 e5", 18, STOI, 768)
        )

    def test_extracts_expected_arrays_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            checkpoint_dir = root / "checkpoints"
            data_dir = root / "data"
            checkpoint_dir.mkdir()
            data_dir.mkdir()
            (data_dir / "val.txt").write_text(
                "e4 e5 Nf3 Nc6\nd4 d5 c4 e6\n",
                encoding="utf-8",
            )

            config = MoveFormerConfig(
                block_size=16,
                n_layer=2,
                n_head=2,
                n_embd=8,
                dropout=0.0,
                vocab_size=len(CHARS),
            )
            model = MoveFormerModel(
                config.block_size,
                config.n_layer,
                config.n_head,
                config.n_embd,
                config.dropout,
                config.vocab_size,
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {"model": asdict(config), "stoi": STOI},
                },
                checkpoint_dir / "ckpt_step10_valloss1.2500.pt",
            )

            with patch.dict(
                os.environ,
                {"ROOT_DIR": str(root), "CHECKPOINT_DIR": "checkpoints"},
            ):
                with patch.object(extract_activations, "N_GAMES", 2):
                    with patch.object(extract_activations, "BATCH_SIZE", 2):
                        with patch.object(extract_activations, "MIN_GAME_PLIES", 0):
                            with patch.object(extract_activations, "CONTEXT_LENGTH", 16):
                                with redirect_stdout(io.StringIO()):
                                    output_path = extract_activations.extract_activations()

            with np.load(output_path) as arrays:
                self.assertEqual(arrays["activations"].shape, (6, 2, 8))
                self.assertEqual(arrays["labels"].shape, (6, 64))
                self.assertEqual(arrays["plies"].shape, (6,))
                self.assertEqual(arrays["game_ids"].shape, (6,))
                self.assertEqual(arrays["activations"].dtype, np.float16)
                self.assertEqual(arrays["labels"].dtype, np.int8)
                self.assertEqual(arrays["plies"].dtype, np.int16)
                self.assertEqual(arrays["game_ids"].dtype, np.int32)


if __name__ == "__main__":
    unittest.main()
