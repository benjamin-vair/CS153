"""Fair comparison of valuenet_v1 vs v2 on a fresh held-out test set."""
import numpy as np, torch
import features as F
from selfplay import generate, expand_samples
from model import load_checkpoint
from opening_advisor import agreement


def main():
    # Fresh games neither model trained on (v1: seeds ~0, v2: + ~100000). Use 900000+.
    print("generating fresh test set (400 games)...")
    Xs, ys = generate(400, kind="victorypoint", seed0=900_000)
    X, Y = expand_samples(Xs, ys)
    print(f"test: {len(Xs)} states -> {len(X)} samples (base win-rate {Y.mean():.3f})\n")

    def eval_model(path):
        model, mean, std = load_checkpoint(path)
        xt = (torch.from_numpy(X) - mean) / std
        with torch.no_grad():
            p = torch.sigmoid(model(xt)).numpy()
        acc = ((p > 0.5).astype(np.float32) == Y).mean()
        eps = 1e-7; pc = np.clip(p, eps, 1 - eps)
        ll = -(Y * np.log(pc) + (1 - Y) * np.log(1 - pc)).mean()
        sep = p[Y == 1].mean() - p[Y == 0].mean()
        corr, top1 = agreement(model, mean, std, n_boards=60)
        return acc, ll, sep, corr, top1

    print(f"{'model':<16}{'test-acc':>9}{'logloss':>9}{'win-sep':>9}{'open-corr':>11}{'open-top1':>11}")
    for path in ["valuenet_v1.pt", "valuenet_v2.pt"]:
        acc, ll, sep, corr, t1 = eval_model(path)
        print(f"{path:<16}{acc:>9.3f}{ll:>9.3f}{sep:>9.3f}{corr:>11.2f}{t1:>11.0%}")


if __name__ == "__main__":
    main()
