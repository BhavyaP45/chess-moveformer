import io
import tempfile
import unittest
from pathlib import Path

import chess.pgn
import zstandard
from tqdm import tqdm

import prepare_data


LONG_GAME = (
    "1. e4 { [%clk 0:10:00] } e5 $1 2. Nf3 Nc6 3. Bb5 a6 "
    "4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O "
    "9. h3 Nb8 10. d4 Nbd7 1-0"
)
EXPECTED_LINE = (
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 "
    "Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7"
)


def make_pgn(
    movetext: str = LONG_GAME,
    *,
    event: str = "Test",
    site: str = "https://lichess.org/test",
    white_elo: str | None = "1600",
    black_elo: str | None = "1700",
    result: str = "1-0",
    time_control: str | None = "600+5",
) -> str:
    headers = [
        ("Event", event),
        ("Site", site),
        ("Date", "2026.07.26"),
        ("Round", "-"),
        ("White", "White"),
        ("Black", "Black"),
        ("Result", result),
    ]
    if white_elo is not None:
        headers.append(("WhiteElo", white_elo))
    if black_elo is not None:
        headers.append(("BlackElo", black_elo))
    if time_control is not None:
        headers.append(("TimeControl", time_control))

    header_text = "\n".join(f'[{key} "{value}"]' for key, value in headers)
    return f"{header_text}\n\n{movetext}\n\n"


class PrepareDataTests(unittest.TestCase):
    def test_filters_games_and_writes_clean_san(self) -> None:
        pgn = "".join(
            [
                make_pgn(event="Kept"),
                make_pgn(event="Low Elo", white_elo="1399"),
                make_pgn(
                    event="Draw",
                    result="1/2-1/2",
                    movetext=LONG_GAME.replace("1-0", "1/2-1/2"),
                ),
                make_pgn(event="Blitz", time_control="300+0"),
                make_pgn(
                    event="Short",
                    movetext="1. e4 e5 2. Nf3 Nc6 1-0",
                ),
            ]
        )
        output = io.StringIO()
        stats = prepare_data.Stats()

        with tqdm(disable=True) as progress:
            prepare_data.process_stream(io.StringIO(pgn), output, stats, progress)

        self.assertEqual(output.getvalue(), EXPECTED_LINE + "\n")
        self.assertEqual(stats.total, 5)
        self.assertEqual(stats.kept, 1)
        self.assertEqual(stats.dropped_elo, 1)
        self.assertEqual(stats.dropped_result, 1)
        self.assertEqual(stats.dropped_time_control, 1)
        self.assertEqual(stats.dropped_too_short, 1)
        self.assertEqual(stats.output_lines, 1)

    def test_reads_zstandard_and_overwrites_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            input_path = root / "game.pgn.zst"
            output_path = root / "nested" / "train.txt"
            input_path.write_bytes(
                zstandard.ZstdCompressor().compress(make_pgn().encode("utf-8"))
            )
            output_path.parent.mkdir()
            output_path.write_text("old data\n", encoding="utf-8")

            stats = prepare_data.prepare_data([input_path], output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                EXPECTED_LINE + "\n",
            )
            self.assertEqual(stats.kept, 1)
            self.assertEqual(stats.output_lines, 1)

    def test_max_games_counts_kept_games_and_stops_early(self) -> None:
        pgn = "".join(
            [
                make_pgn(event="Dropped", white_elo="1200"),
                make_pgn(event="First kept"),
                make_pgn(event="Must not be seen"),
            ]
        )
        output = io.StringIO()
        stats = prepare_data.Stats()

        with tqdm(disable=True) as progress:
            limit_reached = prepare_data.process_stream(
                io.StringIO(pgn),
                output,
                stats,
                progress,
                max_games=1,
            )

        self.assertTrue(limit_reached)
        self.assertEqual(output.getvalue(), EXPECTED_LINE + "\n")
        self.assertEqual(stats.total, 2)
        self.assertEqual(stats.kept, 1)
        self.assertEqual(stats.dropped_elo, 1)
        self.assertEqual(stats.output_lines, 1)

    def test_missing_and_non_numeric_headers_are_rejected(self) -> None:
        for field, value in [
            ("white_elo", None),
            ("black_elo", "not-a-number"),
            ("time_control", None),
            ("time_control", "600"),
            ("time_control", "600+unknown"),
        ]:
            with self.subTest(field=field, value=value):
                arguments = {field: value}
                game = chess.pgn.read_game(io.StringIO(make_pgn(**arguments)))
                self.assertIsNotNone(game)
                stats = prepare_data.Stats()

                assert game is not None
                self.assertIsNone(prepare_data.game_to_line(game, stats))
                if field in {"white_elo", "black_elo"}:
                    self.assertEqual(stats.dropped_elo, 1)
                else:
                    self.assertEqual(stats.dropped_time_control, 1)

    def test_parser_errors_log_game_identity_and_skip_game(self) -> None:
        invalid_pgn = make_pgn(
            event="Broken Game",
            site="https://lichess.org/broken",
            movetext="1. e4 e5 2. e5 1-0",
        )
        output = io.StringIO()
        stats = prepare_data.Stats()

        with self.assertLogs(prepare_data.LOGGER, level="WARNING") as logs:
            with tqdm(disable=True) as progress:
                prepare_data.process_stream(
                    io.StringIO(invalid_pgn), output, stats, progress
                )

        self.assertEqual(output.getvalue(), "")
        self.assertEqual(stats.total, 1)
        self.assertIn("Broken Game", logs.output[0])
        self.assertIn("https://lichess.org/broken", logs.output[0])

    def test_bad_zstandard_file_exits_with_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_directory:
            root = Path(temp_directory)
            input_path = root / "bad.pgn.zst"
            input_path.write_bytes(b"not zstandard data")

            with self.assertRaisesRegex(
                SystemExit, "Failed to decompress Zstandard file"
            ):
                prepare_data.prepare_data([input_path], root / "output.txt")


if __name__ == "__main__":
    unittest.main()
