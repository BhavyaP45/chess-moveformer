import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project_utils import (
    find_best_checkpoint,
    find_latest_checkpoint,
    find_step_0_checkpoint,
    load_games,
)


class ProjectUtilsTests(unittest.TestCase):
    def test_checkpoint_selection_uses_shared_filename_parser(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            paths = [
                checkpoint_dir / "ckpt_step0_valloss2.0000.pt",
                checkpoint_dir / "ckpt_step10_valloss0.9000.pt",
                checkpoint_dir / "ckpt_step20_valloss1.1000.pt",
            ]
            for path in paths:
                path.touch()

            with patch.dict(
                os.environ,
                {"ROOT_DIR": str(root), "CHECKPOINT_DIR": "checkpoints"},
            ):
                self.assertEqual(find_step_0_checkpoint(), paths[0])
                self.assertEqual(find_best_checkpoint(), paths[1])
                self.assertEqual(find_latest_checkpoint(), paths[2])

    def test_load_games_filters_by_minimum_ply_count(self):
        with tempfile.TemporaryDirectory() as temp_directory:
            data_path = Path(temp_directory) / "games.txt"
            fifty_plies = " ".join(["e4"] * 50)
            fifty_one_plies = " ".join(["d4"] * 51)
            data_path.write_text(
                fifty_plies + "\n" + fifty_one_plies + "\n\n",
                encoding="utf-8",
            )

            self.assertEqual(load_games(data_path, min_plies=51), (fifty_one_plies,))


if __name__ == "__main__":
    unittest.main()
