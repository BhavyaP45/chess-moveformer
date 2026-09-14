# Chess MoveFormer

Chess MoveFormer is an independent machine-learning interpretability project studying the robustness and causal role of the implicit board-state representations learned by a small decoder-only transformer trained only on chess text.

**Contents**

- [AI-assisted code development](#ai-assisted-code-development)
- [Main contributions](#main-contributions)
- [Research status](#research-status)
- [Repository map](#repository-map)
- [How the pieces fit together](#how-the-pieces-fit-together)
- [Setup](#setup)
- [Google Colab workflow](#google-colab-workflow)
- [Testing](#testing)
- [Reproducibility notes](#reproducibility-notes)
- [Research lineage](#research-lineage)
- [Responsible use and current limitations](#responsible-use-and-current-limitations)
- [License and citation](#license-and-citation)

## AI-assisted code development

AI coding assistants were used to generate and revise portions of this codebase. GPT-5.6 Sol was used for the majority of AI-assisted code generation, while Grok 4.6 through Cursor was used to a smaller extent. The author reviewed, modified where necessary, and tested the AI-assisted code, and remains responsible for the final implementation and research results.

The model receives no board tensor, piece list, legal-move generator, or chess engine signal. Its only training objective is next-character prediction over games written as space-separated Standard Algebraic Notation (SAN):

```text
e4 e5 Nf3 Nc6 Bb5 a6
```

Unlike a conventional chess engine such as Stockfish, the model has no explicit board state and does not enforce legal moves by construction. This makes it possible to study what the network learns internally from move sequences alone.

The central research question is:

> How does the model's internalized board-state representation vary as the sequence of moves increases, and is this representation used by the model during inference?

This repository contains the complete experimental pipeline used to investigate that question through three complementary methods:

1. **Move-legality evaluation** measures whether generated SAN continuations are legal at different stages of a game.
2. **Linear probing** tests how accurately piece type and colour on each square can be decoded from the residual stream, and how that decodability changes with ply depth.
3. **Causal intervention** tests whether the decoded representation is used during inference by editing square-specific board-state directions and measuring the resulting behavior.

## Main contributions

- A roughly 10-million-parameter character-level transformer trained from scratch on 800,000 Lichess games represented in SAN.
- A bank of linear probes trained on approximately 660,000 board states to classify piece type and colour from each transformer layer's residual vectors.
- A systematic causal evaluation that intervenes on residual-stream representations and compares source-piece edits against matched controls.

## Research status

This is an active research project being prepared for a paper. The repository documents the model, data pipeline, and experimental methodology, but this README intentionally does **not** report quantitative outcomes or conclusions. Results, figures, and a fuller discussion will be added after publication.

Until then, the code is best read as a transparent description of the experimental design rather than as a results release.

## Repository map

```text
chess-moveformer/
├── model.py                         Transformer architecture and generation
├── train.py                         Corpus encoding, batching, training, evaluation, checkpoints
├── evaluate_legality.py             Vectorized top-k legal-move evaluation
├── extract_activations.py           Residual-stream activation extraction
├── train_probes.py                  Per-layer, per-square linear probe training
├── intervention.py                  Causal board-state intervention experiment
├── plots.py                         Paper-oriented figure generation
├── project_utils.py                 Shared paths, game loading, and checkpoint utilities
├── preprocessing/
│   ├── prepare_data.py              Lichess PGN to clean SAN corpus conversion
│   └── chess_data_utils.py          Corpus inspection and SAN legality utilities
└── tests/                            Unit tests for the data and research pipeline
```

## How the pieces fit together

```text
Lichess PGN exports
        │
        ▼
preprocessing/prepare_data.py
        │
        ▼
SAN text corpus ───────────────► train.py ─────────► model checkpoints
                                      │                     │
                                      │                     ├──► evaluate_legality.py
                                      │                     │
                                      │                     └──► extract_activations.py
                                      │                                  │
                                      │                                  ▼
                                      │                           train_probes.py
                                      │                                  │
                                      └──────────────────────────────────┴──► intervention.py
                                                                             │
                                                                             ▼
                                                                          plots.py
```

### `model.py`: character-level transformer

`MoveFormerModel` is a GPT-style causal transformer implemented in PyTorch. It contains learned token and position embeddings, masked multi-head self-attention, feed-forward layers, residual connections, and layer normalization.

The default architecture is:

- 6 transformer blocks
- 6 attention heads per block
- 384-dimensional residual stream
- 768-character maximum context
- 0.2 dropout
- 30-token vocabulary: 28 SAN characters, PAD, and BOS

The model operates on characters rather than chess moves. For example, `Nf3` is processed as three successive tokens. The causal mask allows each position to attend only to characters at or before that position.

`forward()` returns logits with shape `(batch, time, vocabulary)`. During training, targets are shifted by one character and flattened for cross-entropy. PAD has index `0` and is excluded from the loss with `ignore_index=0`.

`generate()` performs autoregressive character sampling while preventing PAD and BOS from appearing in generated text.

### `preprocessing/prepare_data.py`: corpus construction

This script converts one or more Lichess `.pgn` or `.pgn.zst` exports into the plain-text format used by the model. It parses games with `python-chess`; it does not extract moves with regular expressions.

A game is retained only when:

- both ratings are at least 1400;
- the result is decisive;
- the base time is at least 600 seconds;
- the game contains at least 20 half-moves; and
- the PGN can be parsed successfully.

Comments, clock annotations, NAGs, evaluations, move numbers, and result markers disappear naturally when the main line is reconstructed as SAN. The output contains one game per line with no blank lines.

Example:

```bash
python preprocessing/prepare_data.py lichess_db.pgn.zst --out data/train.txt
```

For a smaller development corpus:

```bash
python preprocessing/prepare_data.py lichess_db.pgn.zst --out data/train.txt --max-games 10000
```

### `train.py`: encoding and language-model training

At startup, the corpus is loaded once. Its character vocabulary is discovered, PAD is assigned index `0`, BOS is assigned index `1`, and every game is encoded once into compact byte storage.

The encoded games are split 90/10 into training and validation sets. `get_batch()` samples complete games with replacement, chooses a random contiguous window for long games, and right-pads short games. It returns `x` and one-character-shifted `y` tensors of shape `(batch_size, block_size)`.

The default physical batch size is 8. Eight gradient-accumulation passes produce an effective batch size of 64 while reducing peak GPU memory. Validation runs periodically, and checkpoints contain model state, optimizer state, step, losses, model/training configuration, and `stoi`.

Checkpoint helpers distinguish three use cases:

- `load_latest_checkpoint()` resumes from the greatest training step.
- `load_best_checkpoint()` selects the checkpoint with the lowest validation loss.
- `load_step_0_checkpoint()` loads the earliest checkpoint for baseline comparisons.

### `evaluate_legality.py`: behavioral evaluation

This experiment measures how robustly the model has learned chess rules as move sequences become longer. For each sampled game, it reconstructs the board with `python-chess`, gives the model the SAN prefix, generates one move character by character, and validates the result with `board.parse_san()`.

The evaluator compares top-k sampling settings and multiple ply depths. Its generation path is vectorized: contexts of different lengths are right-padded into a batch, but logits are gathered from each row's final real token rather than from PAD. Because attention is causal, real context tokens cannot attend to padding on their right.

CUDA inference uses BF16 when supported. Completed candidates leave the active batch, and generation stops at the SAN-delimiting space or a fixed character limit.

### `extract_activations.py`: residual-stream dataset

This script loads the best transformer checkpoint, replays held-out games, and records board labels at SAN move boundaries. Forward hooks capture the output of each transformer block after its residual update.

For every retained boundary it stores:

- one residual-stream vector per layer;
- a 64-square board label vector;
- the ply number; and
- a game identifier.

Square labels use 13 classes: empty plus six white and six black piece types. Activations are stored in reduced precision in `activations.npz` to control disk usage.

### `train_probes.py`: linear decoding

The probe experiment asks whether board contents are linearly recoverable from hidden states and whether that representation changes with game depth. It trains probes for each transformer layer and each board square, mapping the residual vector to one of 13 square-state classes: empty, six white piece types, or six black piece types.

The train/test split is made by game rather than by individual position, preventing positions from the same game from leaking across the split. The script records ordinary accuracy, balanced accuracy, per-class behavior, ply-bucket behavior, support counts, and a shuffled-label baseline. Probe weights and the exact split are saved for later analysis.

### `intervention.py`: causal testing

Linear decodability alone does not establish that the model uses the decoded representation. The intervention experiment therefore estimates an occupied-piece-to-empty direction for a square and pushes the move-boundary residual activation toward the empty centroid through forward hooks. It compares editing the source piece selected by the model's greedy move against editing a matched control piece.

The code handles candidate discovery, centroid estimation, multi-layer edits, control-square selection, paired generation, behavioral metrics, and CSV/NPZ output. This experiment requires CUDA.

### `plots.py`: research figures

The plotting module turns saved checkpoints and experiment artifacts into publication-oriented PDF and PNG figures. It includes training-loss, legality, probe-by-layer/ply, and intervention visualizations.

### `project_utils.py`: shared infrastructure

Shared helpers centralize:

- `.env`-based root and checkpoint paths;
- checkpoint naming, parsing, saving, and selection;
- safe checkpoint restoration; and
- streaming or eager loading of one-game-per-line corpora.

## Setup

The project is designed for Python 3.10+ and PyTorch. Install its dependencies in a virtual environment or Colab runtime:

```bash
pip install torch numpy matplotlib python-chess zstandard tqdm python-dotenv pytest
```

Create a local `.env` file. It is gitignored and should not be committed:

```env
ROOT_DIR=/path/to/chess-moveformer
CHECKPOINT_DIR=checkpoints
```

`CHECKPOINT_DIR` may be absolute or relative. A relative value is resolved beneath `ROOT_DIR`.

Expected research data layout:

```text
$ROOT_DIR/
├── data/
│   ├── train.txt
│   └── val.txt
├── checkpoints/
├── activations.npz                 # produced by activation extraction
├── probe_weights.pt                # produced by probe training
├── probe_split.npz
└── figures/                         # produced by plotting
```

Large corpora, checkpoints, activations, result directories, and `.env` are excluded by the current `.gitignore`. Other generated probe and figure artifacts should also be reviewed before committing.

## Google Colab workflow

Mount Drive before importing modules that resolve project paths:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then run from the repository root:

```python
%cd /path/to/chess-moveformer
```

Train the language model:

```bash
python train.py
```

Extract residual-stream activations:

```bash
python extract_activations.py
```

Train the probes:

```bash
python train_probes.py --device cuda
```

Run the intervention experiment:

```bash
python intervention.py
```

Generate available figures:

```bash
python plots.py
```

The legality evaluator is a callable utility rather than a command-line script. After constructing the model and loading the checkpoint you want to evaluate:

```python
from evaluate_legality import evaluate_legality

rates = evaluate_legality(
    model=model,
    val_path=ROOT_DIR / "data" / "val.txt",
    stoi=stoi,
    itos=itos,
    block_size=model_config.block_size,
    device=DEVICE,
    n_batches=100,
    batch_size=64,
)
```

## Testing

Run the complete test suite from the repository root:

```bash
python -m pytest
```

The tests cover corpus preparation, checkpoint selection, legality evaluation, activation extraction, probe training, interventions, and plotting utilities.

## Reproducibility notes

- SAN legality and board reconstruction are delegated to `python-chess`.
- The language-model validation split is deterministic by corpus order; stochastic batches are sampled during evaluation.
- Probe splitting is performed at game level with a fixed random seed.
- Checkpoint filenames encode both step and validation loss.
- `ROOT_DIR` controls all research artifact paths, which keeps the same code usable locally and in Google Drive.
- Evaluation and intervention scripts expect their inputs to have been produced from compatible model checkpoints and vocabulary mappings.

## Research lineage

The project builds on evidence that autoregressive sequence models can learn latent game state without receiving an explicit board. Li et al. demonstrated this behavior in Othello-GPT using probes over model activations, while Karvonen found linearly decodable board representations in language models trained on chess PGN strings. Chess MoveFormer extends that line of work by asking how robust the representation remains as games progress and whether it has a causal role in the model's decisions.

The transformer implementation follows the compact, educational style popularized by nanoGPT.

A formal bibliography, complete experimental specification, quantitative results, and analysis will accompany the paper release.

## Responsible use and current limitations

This repository is research code rather than a chess engine or production training framework. Generated SAN can be syntactically malformed or illegal, saved artifacts can be large, and several experiments are GPU-oriented. Review configuration values and run small smoke tests before launching full Colab jobs.

## License and citation

Licensing and citation information will be finalized with the paper release. If you use or build on this repository before then, please contact the author before redistributing experimental results.
