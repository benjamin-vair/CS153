"""Advise the best next settlement on a user-supplied board + placements.

You describe a real Catan board and the pieces already on it in a JSON file, and
this tool tells you where to place your next settlement. It reuses the trained
value net as a position evaluator (ranking) and the interpretable Catan
heuristics for the explanation (see opening_advisor.py for the rationale).

Board file format (see `--example` to dump a starter):

    {
      "tiles": [                 # exactly 19 hexes, in standard topology order
        {"resource": "ORE",   "number": 10},
        {"resource": "WHEAT", "number": 2},
        {"resource": null,    "number": null},   # the desert
        ... (19 total)
      ],
      "ports": ["WOOD","BRICK","SHEEP","WHEAT","ORE",null,null,null,null],  # optional, 9 entries; null = 3:1
      "placements": {
        "RED":  {"settlements": [12], "cities": [], "roads": [[12, 13]]},
        "BLUE": {"settlements": [29], "roads": [[29, 30]]}
      },
      "advise_for": "WHITE"
    }

Run:
    python advise.py board.json                 # top-5 next settlements for advise_for
    python advise.py board.json --show          # also print the board + node map
    python advise.py --example > board.json     # write a starter file
"""

import sys
import json
import argparse

import numpy as np
import torch

from catanatron import Game, Color, RandomPlayer
from catanatron import state_functions as SF
from catanatron.models.map import CatanMap, BASE_MAP_TEMPLATE, initialize_tiles, LandTile

import features as F
from model import load_checkpoint
from opening_advisor import _node_stats, _why

COLOR_BY_NAME = {c.value: c for c in [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]}
SEAT_COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
VALID_RES = set(F.RESOURCES)

# Land-tile coordinates in the order tiles[] are read (standard base topology).
LAND_COORDS = [c for c, t in BASE_MAP_TEMPLATE.topology.items() if t == LandTile]


# --------------------------------------------------------------------------- #
# Board construction
# --------------------------------------------------------------------------- #
def build_custom_map(tiles, ports=None):
    """Build a CatanMap from an ordered list of 19 {resource, number} hexes."""
    if len(tiles) != len(LAND_COORDS):
        raise ValueError(f"Expected {len(LAND_COORDS)} tiles, got {len(tiles)}.")

    resources, numbers, n_desert = [], [], 0
    for i, t in enumerate(tiles):
        r = t.get("resource")
        if r is None:
            n_desert += 1
            resources.append(None)
        else:
            r = r.upper()
            if r not in VALID_RES:
                raise ValueError(f"tile {i}: bad resource {t['resource']!r}")
            resources.append(r)
            if t.get("number") is None:
                raise ValueError(f"tile {i} ({r}) needs a number token.")
            numbers.append(int(t["number"]))
    if n_desert != 1:
        print(f"[warn] {n_desert} desert tiles (standard board has 1).", file=sys.stderr)

    if ports is None:
        port_resources = list(BASE_MAP_TEMPLATE.port_resources)
    else:
        port_resources = [None if p in (None, "3:1") else p.upper() for p in ports]

    # The engine pops from the END of each list, so reverse to honor our order.
    tiles_map = initialize_tiles(
        BASE_MAP_TEMPLATE,
        shuffled_numbers_param=list(reversed(numbers)),
        shuffled_tile_resources_param=list(reversed(resources)),
        shuffled_port_resources_param=list(reversed(port_resources)),
    )
    return CatanMap.from_tiles(tiles_map)


def _place(state, color, node, kind):
    """Apply one already-decided placement to both the board graph and state."""
    if kind == "settlement":
        try:
            state.board.build_settlement(color, node, initial_build_phase=True)
        except ValueError:
            raise ValueError(
                f"Illegal settlement for {color.value} at node {node}: it is "
                f"occupied or adjacent to another settlement (distance rule).")
        SF.build_settlement(state, color, node, is_free=True)
    elif kind == "city":
        # a city implies the settlement is already there; ensure it then upgrade
        if node not in state.buildings_by_color[color].get("SETTLEMENT", []):
            state.board.build_settlement(color, node, initial_build_phase=True)
            SF.build_settlement(state, color, node, is_free=True)
        state.board.build_city(color, node)
        SF.build_city(state, color, node)


def build_game(spec):
    cmap = build_custom_map(spec["tiles"], spec.get("ports"))
    game = Game([RandomPlayer(c) for c in SEAT_COLORS], seed=0, catan_map=cmap)
    st = game.state
    for cname, pieces in spec.get("placements", {}).items():
        color = COLOR_BY_NAME[cname.upper()]
        for node in pieces.get("settlements", []):
            _place(st, color, node, "settlement")
        for node in pieces.get("cities", []):
            _place(st, color, node, "city")
        for edge in pieces.get("roads", []):
            color_edge = tuple(edge)
            try:
                st.board.build_road(color, color_edge)
                SF.build_road(st, color, color_edge, is_free=True)
            except ValueError:
                # Roads don't affect opening-settlement advice; warn and skip
                # rather than crash on a non-existent or disconnected edge.
                print(f"[warn] skipping invalid road {color_edge} for {cname}.",
                      file=sys.stderr)
    return game


# --------------------------------------------------------------------------- #
# Advice
# --------------------------------------------------------------------------- #
def advise(game, advise_for, model, mean, std, top_k=5, device="cpu"):
    color = COLOR_BY_NAME[advise_for.upper()]
    candidates = game.state.board.buildable_node_ids(color, initial_build_phase=True)
    if not candidates:
        raise ValueError(f"No legal settlement spots for {advise_for}.")

    vecs = np.empty((len(candidates), F.FEATURE_SIZE), dtype=np.float32)
    for i, node in enumerate(candidates):
        gc = game.copy()
        gc.state.board.build_settlement(color, node, initial_build_phase=True)
        SF.build_settlement(gc.state, color, node, is_free=True)
        vecs[i] = F.perspective_vector(gc.state, color)
    x = (torch.from_numpy(vecs).to(device) - mean) / std
    with torch.no_grad():
        win_probs = torch.sigmoid(model(x)).cpu().numpy()

    bmap = game.state.board.map
    rows = []
    for node, wp in zip(candidates, win_probs):
        stats = _node_stats(bmap, node)
        rows.append({"node": node, "win_prob": float(wp), "why": _why(stats, float(wp))})
    rows.sort(key=lambda r: r["win_prob"], reverse=True)
    return rows[:top_k]


def show_board(game):
    st = game.state
    bmap = st.board.map
    print("\nBoard")
    print("-" * 60)
    for i, coord in enumerate(LAND_COORDS):
        tile = bmap.tiles[coord]
        res = tile.resource or "DESERT"
        num = getattr(tile, "number", None)
        nodes = sorted(tile.nodes.values())
        print(f"  hex {i:>2}: {res:<7}{('#'+str(num)) if num else '':<4} "
              f"nodes {nodes}")
    # ports
    ports = {}
    for r, nodes in bmap.port_nodes.items():
        ports.setdefault("3:1" if r is None else r, set()).update(nodes)
    print("  ports:", {k: sorted(v) for k, v in ports.items()})
    # placements
    print("  placed:")
    for color in SEAT_COLORS:
        b = st.buildings_by_color[color]
        if b.get("SETTLEMENT") or b.get("CITY"):
            print(f"    {color.value}: settlements={b.get('SETTLEMENT', [])} "
                  f"cities={b.get('CITY', [])}")
    print()


EXAMPLE = {
    "tiles": [
        {"resource": "WOOD", "number": 11}, {"resource": "SHEEP", "number": 12},
        {"resource": "WHEAT", "number": 9}, {"resource": "BRICK", "number": 4},
        {"resource": "ORE", "number": 6}, {"resource": "BRICK", "number": 5},
        {"resource": "SHEEP", "number": 10}, {"resource": "WHEAT", "number": 3},
        {"resource": "WOOD", "number": 11}, {"resource": None, "number": None},
        {"resource": "WOOD", "number": 8}, {"resource": "ORE", "number": 3},
        {"resource": "ORE", "number": 4}, {"resource": "SHEEP", "number": 5},
        {"resource": "WHEAT", "number": 9}, {"resource": "BRICK", "number": 6},
        {"resource": "WHEAT", "number": 10}, {"resource": "SHEEP", "number": 8},
        {"resource": "WOOD", "number": 2},
    ],
    "placements": {
        "RED": {"settlements": [], "roads": []},
        "BLUE": {"settlements": [], "roads": []},
    },
    "advise_for": "RED",
}


def main():
    ap = argparse.ArgumentParser(description="Best next settlement on a custom board.")
    ap.add_argument("board", nargs="?", help="path to board JSON file")
    ap.add_argument("--model", default="valuenet_v2.pt")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--show", action="store_true", help="print the parsed board first")
    ap.add_argument("--example", action="store_true", help="print a starter board JSON and exit")
    args = ap.parse_args()

    if args.example:
        print(json.dumps(EXAMPLE, indent=2))
        return
    if not args.board:
        ap.error("provide a board JSON file (or use --example)")

    with open(args.board) as f:
        spec = json.load(f)

    game = build_game(spec)
    if args.show:
        show_board(game)

    model, mean, std = load_checkpoint(args.model)
    advise_for = spec["advise_for"]
    recs = advise(game, advise_for, model, mean, std, top_k=args.top)

    print(f"Best next settlement for {advise_for.upper()}")
    print("=" * 60)
    for rank, r in enumerate(recs, 1):
        print(f"{rank}. node {r['node']:>2}  —  {r['why']}")
    print()


if __name__ == "__main__":
    main()
