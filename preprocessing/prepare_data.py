# Requirements: zstandard, python-chess, tqdm

"""Convert Lichess PGN exports into a line-oriented SAN move corpus."""

from __future__ import annotations

import argparse
import io
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

import chess.pgn
import zstandard
from tqdm import tqdm


MIN_ELO = 1400
MIN_BASE_SECONDS = 600  # 10 min — excludes bullet and blitz
MIN_MOVES = 20  # half-moves

LOGGER = logging.getLogger(__name__)


@dataclass
class Stats:
    """Counters collected while processing input games."""

    total: int = 0
    kept: int = 0
    dropped_elo: int = 0
    dropped_result: int = 0
    dropped_time_control: int = 0
    dropped_too_short: int = 0
    output_lines: int = 0


@contextmanager
def open_pgn(path: Path) -> Iterator[TextIO]:
    """Open a plain or Zstandard-compressed PGN as a UTF-8 text stream."""

    if path.name.lower().endswith(".zst"):
        with path.open("rb") as compressed:
            decompressor = zstandard.ZstdDecompressor()
            with decompressor.stream_reader(compressed) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text:
                    yield text
    else:
        with path.open("r", encoding="utf-8") as text:
            yield text


def parse_rating(value: str | None) -> int | None:
    """Return a non-negative integer rating, or None for invalid input."""

    if value is None:
        return None
    try:
        rating = int(value)
    except (TypeError, ValueError):
        return None
    return rating if rating >= 0 else None


def parse_base_seconds(value: str | None) -> int | None:
    """Parse the base component of a BaseSeconds+IncrementSeconds header."""

    if value is None:
        return None
    parts = value.split("+")
    if len(parts) != 2:
        return None
    try:
        base_seconds, increment_seconds = (int(part) for part in parts)
    except ValueError:
        return None
    if base_seconds < 0 or increment_seconds < 0:
        return None
    return base_seconds


def game_identity(game: chess.pgn.Game | None) -> tuple[str, str]:
    """Return Event and Site values suitable for a warning message."""

    if game is None:
        return "?", "?"
    return game.headers.get("Event", "?"), game.headers.get("Site", "?")


def warn_parse_error(game: chess.pgn.Game | None, error: BaseException) -> None:
    event, site = game_identity(game)
    LOGGER.warning(
        "Skipping game after PGN parse error (Event=%r, Site=%r): %s",
        event,
        site,
        error,
    )


def game_to_line(game: chess.pgn.Game, stats: Stats) -> str | None:
    """Filter one game and return its space-separated SAN moves if accepted."""

    white_elo = parse_rating(game.headers.get("WhiteElo"))
    black_elo = parse_rating(game.headers.get("BlackElo"))
    if (
        white_elo is None
        or black_elo is None
        or white_elo < MIN_ELO
        or black_elo < MIN_ELO
    ):
        stats.dropped_elo += 1
        return None

    if game.headers.get("Result") not in {"1-0", "0-1"}:
        stats.dropped_result += 1
        return None

    base_seconds = parse_base_seconds(game.headers.get("TimeControl"))
    if base_seconds is None or base_seconds < MIN_BASE_SECONDS:
        stats.dropped_time_control += 1
        return None

    board = game.board()
    san_moves: list[str] = []
    for move in game.mainline_moves():
        san_moves.append(board.san(move))
        board.push(move)

    if len(san_moves) < MIN_MOVES:
        stats.dropped_too_short += 1
        return None

    return " ".join(san_moves)


def process_stream(
    stream: TextIO,
    output: TextIO,
    stats: Stats,
    progress: tqdm,
    max_games: int | None = None,
) -> bool:
    """Process one PGN stream, returning True when the keep limit is reached."""

    while True:
        if max_games is not None and stats.kept >= max_games:
            return True

        game: chess.pgn.Game | None = None
        try:
            game = chess.pgn.read_game(stream)
        except zstandard.ZstdError:
            raise
        except Exception as error:
            stats.total += 1
            progress.update(1)
            possible_game = getattr(error, "game", None)
            warn_parse_error(
                possible_game if isinstance(possible_game, chess.pgn.Game) else None,
                error,
            )
            continue

        if game is None:
            return False

        stats.total += 1
        progress.update(1)

        if game.errors:
            warn_parse_error(game, game.errors[0])
            continue

        try:
            line = game_to_line(game, stats)
        except Exception as error:
            warn_parse_error(game, error)
            continue

        if line is not None:
            output.write(line)
            output.write("\n")
            stats.kept += 1
            stats.output_lines += 1
            if max_games is not None and stats.kept >= max_games:
                return True


def prepare_data(
    input_paths: list[Path],
    output_path: Path,
    max_games: int | None = None,
) -> Stats:
    """Convert all input PGNs into one overwritten output corpus."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = Stats()

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        with tqdm(desc="Games processed", unit="game") as progress:
            for input_path in input_paths:
                try:
                    with open_pgn(input_path) as stream:
                        limit_reached = process_stream(
                            stream,
                            output,
                            stats,
                            progress,
                            max_games,
                        )
                except zstandard.ZstdError as error:
                    raise SystemExit(
                        f"Failed to decompress Zstandard file "
                        f"{input_path}: {error}"
                    ) from error
                except OSError as error:
                    raise SystemExit(
                        f"Failed to read input file {input_path}: {error}"
                    ) from error
                if limit_reached:
                    break

    return stats


def print_summary(stats: Stats) -> None:
    """Print the requested processing summary to stdout."""

    print(f"Total games seen:     {stats.total}")
    print(f"Games kept:           {stats.kept}")
    print(f"Dropped (Elo):        {stats.dropped_elo}")
    print(f"Dropped (result):     {stats.dropped_result}")
    print(f"Dropped (time ctrl):  {stats.dropped_time_control}")
    print(f"Dropped (too short):  {stats.dropped_too_short}")
    print(f"Output lines written: {stats.output_lines}")


def positive_int(value: str) -> int:
    """Parse a positive integer for an argparse option."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Lichess PGN files to a SAN move text corpus."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input .pgn or .pgn.zst files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output UTF-8 text corpus",
    )
    parser.add_argument(
        "--max-games",
        type=positive_int,
        default=None,
        metavar="N",
        help="Stop after keeping N games",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args()
    stats = prepare_data(args.inputs, args.out, args.max_games)
    print_summary(stats)


if __name__ == "__main__":
    main()
