"""Generate (state -> winner) training data via self-play.

We play full games with some policy (random / weighted / victorypoint / a trained
value net), and record board positions sampled along the way. Each recorded
position is stored once in seat order together with the seat index of the
eventual winner. At training time every position is expanded into 4 perspective
samples (label = 1 if that seat won) -- see `expand_samples`.
"""

import os
import random
import argparse
import time
import multiprocessing as mp

import numpy as np

from catanatron import Game, GameAccumulator, Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer

import features as F

COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


class Recorder(GameAccumulator):
    """Records seat-order feature vectors for sampled pre-action states."""

    def __init__(self, sample_prob=0.12):
        self.sample_prob = sample_prob

    def before(self, game):
        self.vecs = []

    def step(self, game, action):
        st = game.state
        # Always keep the (few, important) opening-placement states; subsample rest.
        if st.is_initial_build_phase or random.random() < self.sample_prob:
            self.vecs.append(F.seat_order_vector(st))

    def after(self, game):
        w = game.winning_color()
        self.result = None if w is None else (self.vecs, game.state.color_to_index[w])


def make_players(kind, model_path=None):
    if kind == "random":
        return [RandomPlayer(c) for c in COLORS]
    if kind == "weighted":
        return [WeightedRandomPlayer(c) for c in COLORS]
    if kind == "victorypoint":
        return [VictoryPointPlayer(c) for c in COLORS]
    if kind == "valuenet":
        from agent import load_value_player  # lazy import (torch)
        return [load_value_player(c, model_path) for c in COLORS]
    raise ValueError(kind)


def _worker(args):
    n_games, kind, seed0, model_path, sample_prob = args
    Xs, ys = [], []
    for i in range(n_games):
        seed = seed0 + i
        random.seed(seed)
        players = make_players(kind, model_path)
        random.shuffle(players)  # vary which color sits in which seat
        game = Game(players, seed=seed)
        rec = Recorder(sample_prob=sample_prob)
        try:
            game.play(accumulators=[rec])
        except Exception:
            continue
        if rec.result is None:
            continue
        vecs, wseat = rec.result
        for v in vecs:
            Xs.append(v)
            ys.append(wseat)
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int8)


def generate(n_games, kind="victorypoint", workers=None, seed0=0,
             model_path=None, sample_prob=0.12):
    workers = workers or max(1, mp.cpu_count() - 1)
    per = [n_games // workers] * workers
    for i in range(n_games - sum(per)):
        per[i] += 1
    seeds = [seed0 + sum(per[:i]) for i in range(workers)]
    jobs = [(per[i], kind, seeds[i], model_path, sample_prob) for i in range(workers) if per[i] > 0]
    t0 = time.time()
    if workers == 1:
        results = [_worker(jobs[0])]
    else:
        with mp.Pool(len(jobs)) as pool:
            results = pool.map(_worker, jobs)
    Xs = np.concatenate([r[0] for r in results if len(r[0])])
    ys = np.concatenate([r[1] for r in results if len(r[1])])
    dt = time.time() - t0
    print(f"[selfplay] {kind}: {n_games} games -> {len(Xs)} states "
          f"in {dt:.1f}s ({n_games/dt:.1f} games/s, {len(jobs)} workers)")
    return Xs, ys


def expand_samples(Xs, ys):
    """Expand each seat-order state into 4 perspective training samples."""
    n = len(Xs)
    X = np.empty((n * F.NUM_PLAYERS, F.FEATURE_SIZE), dtype=np.float32)
    Y = np.empty((n * F.NUM_PLAYERS,), dtype=np.float32)
    k = 0
    for v, wseat in zip(Xs, ys):
        for seat in range(F.NUM_PLAYERS):
            X[k] = F.roll_perspective(v, seat)
            Y[k] = 1.0 if wseat == seat else 0.0
            k += 1
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--kind", default="victorypoint",
                    choices=["random", "weighted", "victorypoint", "valuenet"])
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--model", default=None, help="value-net checkpoint (for kind=valuenet)")
    ap.add_argument("--sample-prob", type=float, default=0.12)
    ap.add_argument("--out", default="data.npz")
    args = ap.parse_args()

    Xs, ys = generate(args.games, kind=args.kind,
                      workers=(args.workers or None), seed0=args.seed0,
                      model_path=args.model, sample_prob=args.sample_prob)
    np.savez_compressed(args.out, X=Xs, y=ys)
    print(f"[selfplay] saved {args.out}  (states={len(Xs)})")


if __name__ == "__main__":
    main()
