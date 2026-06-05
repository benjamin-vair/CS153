"""Train the value network from a self-play dataset (.npz of seat-order states)."""

import argparse
import numpy as np
import torch
import torch.nn as nn

import features as F
from selfplay import expand_samples
from model import ValueNet, save_checkpoint


def train(data_paths, out="valuenet.pt", hidden=256, depth=3, epochs=12,
          batch=4096, lr=1e-3, val_frac=0.05, device=None, seed=0):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load + concatenate datasets, expand to perspective samples
    Xs, ys = [], []
    for p in data_paths:
        d = np.load(p)
        Xs.append(d["X"]); ys.append(d["y"])
    Xs = np.concatenate(Xs); ys = np.concatenate(ys)
    X, Y = expand_samples(Xs, ys)
    print(f"[train] {len(Xs)} states -> {len(X)} samples, win-rate={Y.mean():.3f}")

    # shuffle + split
    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]
    n_val = int(len(X) * val_frac)
    Xtr, Ytr = X[n_val:], Y[n_val:]
    Xva, Yva = X[:n_val], Y[:n_val]

    # standardize using training stats
    mean = Xtr.mean(0)
    std = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xva = (Xva - mean) / std

    Xtr = torch.as_tensor(Xtr, device=device)
    Ytr = torch.as_tensor(Ytr, device=device)
    Xva = torch.as_tensor(Xva, device=device)
    Yva = torch.as_tensor(Yva, device=device)

    model = ValueNet(F.FEATURE_SIZE, hidden, depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            logits = model(Xtr[b])
            loss = lossf(logits, Ytr[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        # validation
        model.eval()
        with torch.no_grad():
            vlogit = model(Xva)
            vloss = lossf(vlogit, Yva).item()
            vpred = (torch.sigmoid(vlogit) > 0.5).float()
            vacc = (vpred == Yva).float().mean().item()
        print(f"[train] epoch {ep+1:2d}/{epochs}  train_loss={tot/n:.4f}  "
              f"val_loss={vloss:.4f}  val_acc={vacc:.3f}")

    save_checkpoint(out, model, mean, std, F.FEATURE_SIZE, hidden, depth)
    print(f"[train] saved {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+", help="one or more .npz datasets")
    ap.add_argument("--out", default="valuenet.pt")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    train(args.data, out=args.out, hidden=args.hidden, depth=args.depth,
          epochs=args.epochs, batch=args.batch, lr=args.lr)


if __name__ == "__main__":
    main()
