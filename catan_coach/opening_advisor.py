"""Opening-placement advisor: rank the best opening settlement spots.

This is the project's primary deliverable. For the player currently on the clock
in the initial build phase, it enumerates every legal settlement vertex, scores
each one, and returns the top spots with a plain-English reason.

Two signals are combined:

  * Value net (learned) -- we simulate placing the settlement and ask the trained
    value network "what is this player's win probability now?". The net predicts
    winners on held-out positions at ~96% accuracy, so it is a strong *evaluator*
    (we use it here only to score positions, never as a greedy player -- 1-ply
    value-greedy play hoards resources and under-builds; see the writeup).
  * Interpretable Catan heuristics -- expected production (pips), resource
    diversity, and port access, pulled straight from the board. These make every
    recommendation explainable and are what the "why" line reports.

We rank by the value net and explain with the heuristics; `agreement()` reports
how well the two corroborate, which doubles as an eval result.
"""

import argparse

import numpy as np
import torch

from catanatron import Game, Color, RandomPlayer
from catanatron.models.enums import ActionType

import features as F
from model import load_checkpoint

RESOURCES = F.RESOURCES


def _port_at(board_map, node):
    """Return the port resource at `node` ('3:1' for any-port, None if no port)."""
    for resource, nodes in board_map.port_nodes.items():
        if node in nodes:
            return "3:1" if resource is None else resource
    return None


def _node_stats(board_map, node):
    """Interpretable features for a settlement on `node`."""
    prod = board_map.node_production[node]  # resource -> expected count per roll
    by_resource = {r: float(prod.get(r, 0.0)) for r in RESOURCES}
    total = sum(by_resource.values())
    return {
        "by_resource": by_resource,
        "production": total,            # expected resources per roll
        "pips": int(round(total * 36)),  # classic Catan "pip" count
        "diversity": sum(1 for v in by_resource.values() if v > 0),
        "port": _port_at(board_map, node),
    }


def _why(stats, win_prob):
    """One-line, human-readable justification for a spot."""
    res = stats["by_resource"]
    top = sorted((v, r) for r, v in res.items() if v > 0)[::-1][:3]
    res_str = ", ".join(f"{r.lower()} {v*36:.0f}" for v, r in top) or "no production"
    parts = [f"{stats['pips']} pips ({res_str})",
             f"{stats['diversity']}/5 resources"]
    if stats["port"]:
        parts.append(f"{stats['port']} port")
    parts.append(f"win-prob {win_prob:.0%}")
    return "; ".join(parts)


def recommend(game, model, mean, std, top_k=5, device="cpu"):
    """Rank legal opening settlement spots for the player on the clock.

    Returns a list of dicts (best first), each with node, win_prob, the
    interpretable stats, and a 'why' string.
    """
    state = game.state
    color = state.current_color()
    board_map = state.board.map

    candidates = [a for a in state.playable_actions
                  if a.action_type == ActionType.BUILD_SETTLEMENT]
    if not candidates:
        raise ValueError("No settlement placements available — not an opening decision.")

    # Score each candidate position with the value net (one batched forward pass).
    vecs = np.empty((len(candidates), F.FEATURE_SIZE), dtype=np.float32)
    for i, action in enumerate(candidates):
        gc = game.copy()
        gc.execute(action)
        vecs[i] = F.perspective_vector(gc.state, color)
    x = (torch.from_numpy(vecs).to(device) - mean) / std
    with torch.no_grad():
        win_probs = torch.sigmoid(model(x)).cpu().numpy()

    rows = []
    for action, wp in zip(candidates, win_probs):
        node = action.value
        stats = _node_stats(board_map, node)
        rows.append({
            "node": node,
            "win_prob": float(wp),
            "production": stats["production"],
            "pips": stats["pips"],
            "diversity": stats["diversity"],
            "port": stats["port"],
            "by_resource": stats["by_resource"],
            "why": _why(stats, float(wp)),
        })
    rows.sort(key=lambda r: r["win_prob"], reverse=True)
    return rows[:top_k]


def agreement(model, mean, std, n_boards=50, device="cpu"):
    """Eval: how well does the learned value net's ranking agree with the
    classic 'most pips' heuristic at the opening? Reports mean Spearman
    correlation and top-1 match rate across `n_boards` fresh boards."""
    import random
    corrs, top1 = [], 0
    for s in range(n_boards):
        random.seed(s)
        g = Game([RandomPlayer(c) for c in
                  [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]], seed=s)
        rows = recommend(g, model, mean, std, top_k=10_000, device=device)
        wp = np.array([r["win_prob"] for r in rows])
        pip = np.array([r["production"] for r in rows])
        # Spearman = Pearson on ranks
        rwp = np.argsort(np.argsort(wp))
        rpip = np.argsort(np.argsort(pip))
        if rwp.std() > 0 and rpip.std() > 0:
            corrs.append(float(np.corrcoef(rwp, rpip)[0, 1]))
        if rows[0]["node"] == max(rows, key=lambda r: r["production"])["node"]:
            top1 += 1
    return float(np.mean(corrs)), top1 / n_boards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="valuenet_v2.pt")
    ap.add_argument("--seed", type=int, default=0, help="board to advise on")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--eval", action="store_true",
                    help="report agreement between value net and pip heuristic")
    args = ap.parse_args()

    model, mean, std = load_checkpoint(args.model)

    g = Game([RandomPlayer(c) for c in
              [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]], seed=args.seed)
    color = g.state.current_color()
    recs = recommend(g, model, mean, std, top_k=args.top)

    print(f"\nOpening placement advice  (board seed={args.seed}, player={color})")
    print("=" * 64)
    for rank, r in enumerate(recs, 1):
        print(f"{rank}. node {r['node']:>2}  —  {r['why']}")
    print()

    if args.eval:
        corr, t1 = agreement(model, mean, std)
        print(f"[eval] value-net vs pip-heuristic over 50 boards: "
              f"Spearman={corr:.2f}, top-1 match={t1:.0%}")


if __name__ == "__main__":
    main()
