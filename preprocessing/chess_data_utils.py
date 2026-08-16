import os
from pathlib import Path

import chess
import numpy as np
from dotenv import load_dotenv


load_dotenv()
ROOT_DIR = Path(os.environ["ROOT_DIR"])
DATA_PATH = ROOT_DIR / "data" / "train.txt"


def compute_block_size(txt_path: str | Path, percentile: float = 95.0):
    with Path(txt_path).open("r", encoding="utf-8") as file:
        lengths = np.array([len(line.rstrip("\r\n")) for line in file if line.strip()])

    raw = int(np.percentile(lengths, percentile))
    recommended = 1
    while recommended < raw:
        recommended *= 2

    print(f"Games                  : {len(lengths):,}")
    print(f"p50 / p90 / p95 / p99 : {np.percentile(lengths, [50, 90, 95, 99]).astype(int)}")
    print(f"Raw p{percentile:.0f}                : {raw}")
    print(f"Recommended            : {recommended}  (next power of 2)")
    print(f"Coverage               : {np.mean(lengths <= recommended) * 100:.1f}% of games fit fully")


def inspect_vocabulary(txt_path: str | Path):
    with Path(txt_path).open("r", encoding="utf-8") as file:
        chars = sorted(set(file.read()))

    print(chars, len(chars))
    return chars


def validate_game(game_text: str) -> tuple[bool, int | None, str | None, str | None]:
    """Validate space-separated SAN moves from the standard starting position."""
    board = chess.Board()

    for ply, san in enumerate(game_text.split(), start=1):
        try:
            board.push_san(san)
        except ValueError as error:
            return False, ply, san, str(error)

    return True, None, None, None


def main():
    compute_block_size(DATA_PATH)
    inspect_vocabulary(DATA_PATH)


if __name__ == "__main__":
    main()
