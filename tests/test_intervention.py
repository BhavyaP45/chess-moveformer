import io
from contextlib import redirect_stdout

import chess
import numpy as np
import torch
import torch.nn as nn

import intervention
from intervention import (
    CandidatePosition,
    CentroidSpec,
    GeometryDiagnostics,
    InterventionPosition,
    MultiLayerCentroidIntervention,
    apply_centroid_edit,
    capture_condition_geometry,
    centroid_progress,
    estimate_centroid_specs,
    example_records,
    generate_greedy_moves,
    group_indices,
    identify_intervention_positions,
    move_involves_square,
    sample_candidate_positions,
    save_results,
    select_control_square,
    select_supported_positions,
    summarize_paired,
)


CHARS = [
    "\0", "\1", " ", "#", "+", "-", "1", "2", "3", "4", "5", "6",
    "7", "8", "=", "B", "K", "N", "O", "Q", "R", "a", "b", "c",
    "d", "e", "f", "g", "h", "x",
]
STOI = {character: index for index, character in enumerate(CHARS)}
ITOS = {index: character for character, index in STOI.items()}
LONG_GAME = (
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 "
    "c3 O-O h3 Nb8 d4 Nbd7 c4 c6 Nc3"
)


class ResidualModel(nn.Module):
    def __init__(self, n_features=2, n_layers=2):
        super().__init__()
        self.n_features = n_features
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(n_layers)])

    def forward(self, idx, targets=None):
        del targets
        residual = torch.ones(
            (idx.shape[0], idx.shape[1], self.n_features),
            device=idx.device,
        )
        for block in self.blocks:
            residual = block(residual)
        return residual, None


class ScriptedMoveModel(nn.Module):
    def __init__(self, move="e4 "):
        super().__init__()
        self.move = move
        self.blocks = nn.ModuleList([nn.Identity()])

    def forward(self, idx, targets=None):
        del targets
        residual = torch.zeros((idx.shape[0], idx.shape[1], 2), device=idx.device)
        self.blocks[0](residual)
        logits = torch.full(
            (idx.shape[0], idx.shape[1], len(CHARS)),
            -1000.0,
            device=idx.device,
        )
        for row in range(idx.shape[0]):
            real_tokens = [token for token in idx[row].tolist() if token != STOI["\0"]]
            text = "".join(ITOS[token] for token in real_tokens)
            suffix = text.rsplit(" ", 1)[1]
            logits[row, len(real_tokens) - 1, STOI[self.move[len(suffix)]]] = 1000.0
        return logits, None


def _spec(layer, direction, piece_target=0.0, empty_target=2.0):
    gap = empty_target - piece_target
    return CentroidSpec(
        layer=layer,
        square=chess.E2,
        piece_class=chess.PAWN,
        direction=torch.tensor(direction, dtype=torch.float32),
        piece_target=torch.tensor(piece_target),
        empty_target=torch.tensor(empty_target),
        gap=torch.tensor(gap),
        piece_support=10,
        empty_support=10,
    )


def _position(
    source_square=chess.E2,
    control_square=chess.D2,
    greedy_san="e4",
    game_id=0,
):
    board = chess.Board()
    greedy_move = board.parse_san(greedy_san)
    return InterventionPosition(
        context=[STOI["\1"], STOI[" "]],
        board=board,
        game_id=game_id,
        ply=15,
        ground_truth_move=board.parse_san("d4"),
        piece_class=chess.PAWN,
        source_square=source_square,
        control_piece_class=chess.PAWN,
        control_square=control_square,
        greedy_move=greedy_move,
    )


def _geometry(n=1, layers=2, treatment=True):
    pre = np.zeros((n, layers), dtype=np.float32)
    post = np.ones((n, layers), dtype=np.float32)
    reference_post = post if treatment else np.full((n, layers), 0.1, dtype=np.float32)
    return GeometryDiagnostics(
        pre_progress=pre,
        post_progress=post,
        edit_fraction=np.ones((n, layers), dtype=np.float32),
        step_norm=np.ones((n, layers), dtype=np.float32),
        reference_pre_progress=pre,
        reference_post_progress=reference_post,
    )


def test_centroid_edit_moves_only_along_axis_at_requested_scale():
    activation = torch.tensor([[0.0, 7.0]])
    spec = _spec(0, [1.0, 0.0])

    half = apply_centroid_edit(activation, spec, 0.5)
    exact = apply_centroid_edit(activation, spec, 1.0)
    overshoot = apply_centroid_edit(activation, spec, 2.0)

    assert torch.allclose(half, torch.tensor([[1.0, 7.0]]))
    assert torch.allclose(exact, torch.tensor([[2.0, 7.0]]))
    assert torch.allclose(overshoot, torch.tensor([[4.0, 7.0]]))
    assert torch.allclose(centroid_progress(exact, spec), torch.tensor([1.0]))


def test_centroid_specs_use_specific_piece_not_all_occupied():
    activations = np.array(
        [
            [[2.0, 0.0], [4.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[100.0, 0.0], [100.0, 0.0]],
        ],
        dtype=np.float32,
    )
    labels = np.array([[0], [0], [1], [1], [2]], dtype=np.int8)
    specs = estimate_centroid_specs(
        activations,
        labels,
        {(1, 0), (2, 0)},
        device="cpu",
        min_support=2,
    )

    assert (1, 0) in specs
    assert (2, 0) not in specs
    assert torch.allclose(specs[(1, 0)][0].direction, torch.tensor([1.0, 0.0]))
    assert torch.isclose(specs[(1, 0)][1].empty_target, torch.tensor(4.0))


def test_multilayer_wrapper_edits_every_layer_and_cleans_hooks():
    model = ResidualModel()
    specs = (_spec(0, [1.0, 0.0]), _spec(1, [0.0, 1.0], 0.0, 3.0))
    wrapper = MultiLayerCentroidIntervention(
        model, specs, 1.0, STOI[" "], record_geometry=True
    )
    tokens = torch.tensor([[STOI["\1"], STOI[" "], STOI["e"]]])

    with wrapper:
        output, _ = wrapper(tokens)
        assert torch.allclose(output[0, 1], torch.tensor([2.0, 3.0]))
        assert all(len(block._forward_hooks) == 1 for block in model.blocks)
        assert torch.isclose(wrapper.geometry[0]["post_progress"][0], torch.tensor(1.0))
        assert torch.isclose(wrapper.geometry[1]["post_progress"][0], torch.tensor(1.0))

    assert all(len(block._forward_hooks) == 0 for block in model.blocks)


def test_geometry_capture_reports_each_layer_and_control_source_cross_effect():
    model = ResidualModel()
    sample = _position()
    own_specs = (_spec(0, [1.0, 0.0]), _spec(1, [0.0, 1.0]))
    source_specs = (_spec(0, [0.0, 1.0]), _spec(1, [1.0, 0.0]))
    specs = {
        (sample.piece_class, sample.source_square): source_specs,
        (sample.control_piece_class, sample.control_square): own_specs,
    }
    diagnostics = capture_condition_geometry(
        model,
        [sample],
        "control",
        specs,
        1.0,
        STOI,
        "cpu",
        batch_size=1,
    )

    assert diagnostics.post_progress.shape == (1, 2)
    assert np.allclose(diagnostics.post_progress, 1.0)
    assert np.isfinite(diagnostics.reference_post_progress).all()
    assert all(len(block._forward_hooks) == 0 for block in model.blocks)


def test_control_selection_is_matched_deterministic_and_not_in_move():
    board = chess.Board()
    move = board.parse_san("e4")
    first = select_control_square(board, move, game_id=3, ply=15)
    second = select_control_square(board, move, game_id=3, ply=15)

    assert first == second
    square, piece_class = first
    assert piece_class == chess.PAWN
    assert not move_involves_square(board, move, square)


def test_control_groups_include_source_reference_key():
    first = _position(source_square=chess.E2, control_square=chess.A2, game_id=1)
    second = _position(
        source_square=chess.D2,
        control_square=chess.A2,
        greedy_san="d4",
        game_id=2,
    )
    groups = group_indices([first, second], "control")
    assert len(groups) == 2


def test_identification_excludes_kings_and_adds_control(monkeypatch):
    board = chess.Board()
    pawn_move = board.parse_san("e4")
    candidate = CandidatePosition(
        context=[STOI["\1"], STOI[" "]],
        board=board,
        game_id=0,
        ply=15,
        ground_truth_move=board.parse_san("d4"),
    )
    monkeypatch.setattr(
        intervention,
        "generate_greedy_moves",
        lambda *_args, **_kwargs: [pawn_move],
    )
    positions = identify_intervention_positions(
        [candidate], None, STOI, ITOS, 16, "cpu"
    )
    assert len(positions) == 1
    assert positions[0].source_square == chess.E2
    assert positions[0].control_square != chess.E2


def test_supported_selection_balances_piece_classes_or_reports_shortage():
    positions = [_position()]
    source_key = (chess.PAWN, chess.E2)
    control_key = (chess.PAWN, chess.D2)
    specs = {source_key: (), control_key: ()}
    try:
        select_supported_positions(positions, specs, per_piece=1)
    except RuntimeError as error:
        assert "white knight" in str(error)
    else:
        raise AssertionError("Expected shortages for missing piece classes")


def test_paired_summary_uses_legal_denominator_and_specificity():
    sample = _position()
    treatment_move = sample.board.parse_san("d4")
    control_move = sample.greedy_move
    rows = summarize_paired(
        [sample],
        [treatment_move],
        [control_move],
        _geometry(),
        _geometry(treatment=False),
        scale=1.0,
    )
    overall = rows[0]

    assert overall["treatment_plan_retention"] == 0.0
    assert overall["control_plan_retention"] == 100.0
    assert overall["specificity_source_square_usage"] == 100.0
    assert overall["specificity_forced_alternative"] == 100.0
    assert overall["treatment_legality"] == 100.0


def test_example_records_and_versioned_outputs(tmp_path):
    sample = _position()
    rows = example_records(
        [sample], "treatment", [sample.greedy_move], _geometry(), 1.0
    )
    summaries = summarize_paired(
        [sample],
        [sample.greedy_move],
        [sample.greedy_move],
        _geometry(),
        _geometry(treatment=False),
        1.0,
    )
    paths = save_results(rows, summaries, tmp_path)

    assert rows[0]["plan_retention"]
    assert paths["example_npz"].name == "causal_empty_intervention_examples_v4.npz"
    assert paths["summary_csv"].name == "causal_empty_intervention_summary_v4.csv"
    assert all(path.exists() for path in paths.values())
    with np.load(paths["summary_npz"]) as results:
        assert results["scale"][0] == 1.0


def test_candidate_sampling_and_greedy_generation_end_to_end():
    candidates = sample_candidate_positions(
        iter([LONG_GAME]), STOI, block_size=256, pool_size=10
    )
    assert len(candidates) == 1
    assert candidates[0].context[-1] == STOI[" "]

    sample = _position()
    with redirect_stdout(io.StringIO()):
        moves = generate_greedy_moves(
            ScriptedMoveModel(),
            [sample],
            STOI,
            ITOS,
            block_size=16,
            device="cpu",
            label="test",
            batch_size=1,
        )
    assert moves == [sample.board.parse_san("e4")]
