# Catan Coach — how to run

Recommends the **best opening settlement** for any Catan board, using a trained
value network. See [Findings](#findings) below for the write-up and results.

All commands run from this folder with the project's virtual environment:

```bash
cd /Users/benjysteinberg/Downloads/CS153/catan_coach
```

---

## 🖱️ Interactive visual board (recommended)

Build the board and place pieces by **clicking** — no typing.

```bash
./.venv/bin/python webapp.py
```

Then open **http://127.0.0.1:8000** in your browser. Stop the server with `Ctrl-C`.

How to use it:
1. **Paint terrain** — pick a resource (and number) in the panel, then click a hex to set it. The board starts pre-filled so you can just tweak it.
2. **Place pieces** — pick a player color and Settlement/City, then click a corner to place it (click again to remove). Illegal spots (too close together) are rejected with a message.
3. **Advise** — choose the player to advise and click **Get advice**. The top 5 spots are numbered on the board, with a one-line "why" for each.

The board geometry is computed from Catanatron's own node coordinates, so every
clickable corner is a real engine node ID.

---

## ⌨️ Command-line tools

**Generated board** (no input needed):
```bash
./.venv/bin/python opening_advisor.py --seed 7          # advice on board #7
./.venv/bin/python opening_advisor.py --seed 7 --eval   # + value-net vs heuristic agreement
```

**Your own board from a file** (the text version of the web app):
```bash
./.venv/bin/python advise.py --example > board.json     # starter file to edit
./.venv/bin/python advise.py board.json --show          # prints board + top-5 spots
```
`board.json` lists the 19 hexes (resource + number, standard order), the pieces
already placed per color, and `advise_for`. `--show` prints each hex with its
surrounding node IDs so you know which numbers to use.

---

## What's under the hood

| File | Role |
|---|---|
| `webapp.py` + `board_ui.html` | the interactive UI (stdlib server, no extra installs) |
| `advise.py` | build a custom board + placements, rank the next settlement |
| `opening_advisor.py` | rank opening spots on a generated board (+ agreement eval) |
| `features.py`, `model.py`, `train.py`, `selfplay.py`, `evaluate.py`, `agent.py` | the value-net pipeline |
| `compare_models.py` | fair held-out comparison of `valuenet_v1` vs `valuenet_v2` |
| `valuenet_v2.pt` | the trained value network (**deployed** — loaded by the app and CLI) |
| `valuenet_v1.pt` | the earlier model, kept for the v1-vs-v2 comparison |

> Note: win-probabilities are a **ranking signal**, not calibrated odds — the
> ordering is what's meaningful (see [Findings](#findings)).

---

## Findings

- **The value net is a strong position evaluator.** On held-out positions it
  predicts the eventual winner at **~96% accuracy** (winners average ~0.89
  predicted win-prob vs ~0.04 for losers, against a 25% base rate). The deployed
  **v2** model improves calibration (log-loss **1.15 vs 1.48** for v1) and opening
  agreement (top-1 match **65% vs 52%**) on a fresh held-out test set — reproduce
  with `python compare_models.py`.
- **Greedy 1-ply value-maximization fails as a _policy_.** Used as a player it
  wins only **~13% vs random** (below the 25% chance baseline): the net learned
  that winners correlate with holding resources/VP, so one-ply "build" (which
  spends) scores lower than "end turn" (which hoards), and the agent never
  converts resources into points. This is why the product uses the net as an
  **evaluator/ranker**, not as a greedy player.
- **The opening advisor recovers human Catan intuition.** Its ranking agrees with
  the classic "most pips" heuristic at **Spearman ~0.92**, while picking a
  _different_ #1 spot about **half the time** — trading a pip or two for better
  resource balance or port access.

---

## AI usage disclosure

This project was developed with the assistance of AI coding tools (Claude /
Claude Code) for code scaffolding, refactoring, debugging, and documentation.
All AI-assisted code was reviewed, run, and validated by the author, and the
experimental design, findings, and analysis are the author's own.

## Citations / acknowledgements

- **Catanatron** — the open-source Catan simulator this project is built on:
  [bcollazo/catanatron](https://github.com/bcollazo/catanatron). The board
  geometry, game rules, legality checks, and baseline bots
  (`RandomPlayer`, `WeightedRandomPlayer`, `VictoryPointPlayer`) come from it.
- **PyTorch** — used to define and train the value network.
- **NumPy** — feature arrays and dataset handling.
- *Settlers of Catan* is a trademark of its respective owners; this is a
  non-commercial educational project for CS 153.
