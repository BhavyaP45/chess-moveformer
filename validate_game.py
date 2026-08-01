import chess


def validate_game(game_text: str) -> tuple[bool, int | None, str | None, str | None]:
    """Validate space-separated SAN moves from the standard starting position."""
    board = chess.Board()

    for ply, san in enumerate(game_text.split(), start=1):
        try:
            board.push_san(san)
        except ValueError as error:
            return False, ply, san, str(error)

    return True, None, None, None

