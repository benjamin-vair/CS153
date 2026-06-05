"""Benchmark a trained value-net agent against baseline bots.

Plays N games in which one seat is the value-net agent and the other three are a
chosen baseline. The agent's color is rotated across seats to remove first-player
bias. Reports win rate (random 4-player baseline = 25%).
"""

import argparse
import random
import time
import multiprocessing as mp

import numpy as np

from catanatron import Game, Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer

COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
BASELINES = {
    "random": RandomPlayer,
    "weighted": WeightedRandomPlayer,
    "victorypoint": VictoryPointPlayer,
}


def _worker(args):
    n_games, baseline, model_path, seed0 = args
    from agent import load_value_player  # lazy import per process
    import torch
    torch.set_num_threads(1)
    agent_color = Color.RED
    base_cls = BASELINES[baseline]
    wins = draws = 0
    for i in range(n_games):
        seed = seed0 + i
        random.seed(seed)
        players = [load_value_player(agent_color, model_path)] + \
                  [base_cls(c) for c in COLORS if c != agent_color]
        random.shuffle(players)  # rotate agent's seat
        game = Game(players, seed=seed)
        try:
            w = game.play()
        except Exception:
            continue
        if w is None:
            draws += 1
        elif w == agent_color:
            wins += 1
    return wins, draws, n_games


def evaluate(model_path, baseline="weighted", games=200, workers=None, seed0=10_000):
    workers = workers or max(1, mp.cpu_count() - 1)
    per = [games // workers] * workers
    for i in range(games - sum(per)):
        per[i] += 1
    jobs = [(per[i], baseline, model_path, seed0 + sum(per[:i]))
            for i in range(workers) if per[i] > 0]
    t0 = time.time()
    with mp.Pool(len(jobs)) as pool:
        res = pool.map(_worker, jobs)
    wins = sum(r[0] for r in res)
    draws = sum(r[1] for r in res)
    total = sum(r[2] for r in res)
    decided = total - draws
    wr = wins / total if total else 0.0
    se = (wr * (1 - wr) / total) ** 0.5 if total else 0.0
    dt = time.time() - t0
    print(f"[eval] vs {baseline:12s}: win={wins}/{total} = {wr:.1%} "
          f"(±{1.96*se:.1%} 95%CI; draws={draws}; chance=25%) in {dt:.0f}s")
    return wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="valuenet.pt")
    ap.add_argument("--baseline", default="all",
                    choices=["random", "weighted", "victorypoint", "all"])
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    baselines = ["random", "weighted", "victorypoint"] if args.baseline == "all" else [args.baseline]
    for b in baselines:
        evaluate(args.model, baseline=b, games=args.games, workers=(args.workers or None))


if __name__ == "__main__":
    main()
