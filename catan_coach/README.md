# Catan Coach — how to run

Recommends the **best opening settlement** for any Catan board, using a trained
value network. See `../RESULTS.md` for the write-up and findings.

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
| `valuenet_v1.pt` | the trained value network |

> Note: win-probabilities are a **ranking signal**, not calibrated odds — the
> ordering is what's meaningful (see `../RESULTS.md`).
