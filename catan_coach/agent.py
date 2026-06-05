"""ValueNetPlayer: chooses moves by 1-ply lookahead under the learned value net.

For each legal action it simulates the resulting position and scores it with the
network from its own perspective, then plays the highest-scoring action. All
candidate positions for a decision are scored in a single batched forward pass.
"""

import numpy as np
import torch

from catanatron import Player
import features as F
from model import load_checkpoint


class ValueNetPlayer(Player):
    def __init__(self, color, model, mean, std, device="cpu", is_bot=True):
        super().__init__(color, is_bot)
        self.model = model
        self.mean = mean
        self.std = std
        self.device = device

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        vecs = np.empty((len(playable_actions), F.FEATURE_SIZE), dtype=np.float32)
        for i, action in enumerate(playable_actions):
            gc = game.copy()
            gc.execute(action)
            vecs[i] = F.perspective_vector(gc.state, self.color)
        x = (torch.from_numpy(vecs).to(self.device) - self.mean) / self.std
        with torch.no_grad():
            scores = torch.sigmoid(self.model(x))
        return playable_actions[int(torch.argmax(scores).item())]


def load_value_player(color, model_path, device="cpu"):
    model, mean, std = load_checkpoint(model_path, device)
    return ValueNetPlayer(color, model, mean, std, device)
