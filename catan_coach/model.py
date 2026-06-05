"""The value network: board features -> probability the perspective player wins."""

import torch
import torch.nn as nn


class ValueNet(nn.Module):
    def __init__(self, in_dim, hidden=256, depth=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def save_checkpoint(path, model, mean, std, in_dim, hidden, depth):
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean, "std": std,
        "in_dim": in_dim, "hidden": hidden, "depth": depth,
    }, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ValueNet(ckpt["in_dim"], ckpt["hidden"], ckpt["depth"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mean = torch.as_tensor(ckpt["mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(ckpt["std"], dtype=torch.float32, device=device)
    return model, mean, std
