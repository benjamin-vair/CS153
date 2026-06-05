"""Feature extraction for the Catan value network.

A board position is turned into a fixed-length float vector. The vector is laid
out as 4 per-player blocks followed by a small block of global features:

    [ block(seat0) | block(seat1) | block(seat2) | block(seat3) | global ]

Each per-player block contains public game-state counts PLUS the player's
expected dice production (pips per resource) derived from the board. Because the
blocks are seat-ordered and equal length, we can produce any player's
"perspective" (their block first) just by rolling the four blocks -- see
`perspective_vector`. This lets one network evaluate "does the player in slot 0
win?" and be reused for every seat.
"""

import numpy as np
from catanatron.state_functions import get_player_buildings
from catanatron.models.enums import SETTLEMENT, CITY

RESOURCES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]

# Public-ish per-player scalars pulled straight from state.player_state.
# (We use ACTUAL_VICTORY_POINTS so the value fn sees hidden VP cards -- it is a
#  heuristic evaluator, not a player with hidden information.)
PER_PLAYER_STATE_KEYS = [
    "ACTUAL_VICTORY_POINTS",
    "ROADS_AVAILABLE", "SETTLEMENTS_AVAILABLE", "CITIES_AVAILABLE",
    "HAS_ROAD", "HAS_ARMY", "LONGEST_ROAD_LENGTH",
    "WOOD_IN_HAND", "BRICK_IN_HAND", "SHEEP_IN_HAND", "WHEAT_IN_HAND", "ORE_IN_HAND",
    "KNIGHT_IN_HAND", "YEAR_OF_PLENTY_IN_HAND", "MONOPOLY_IN_HAND",
    "ROAD_BUILDING_IN_HAND", "VICTORY_POINT_IN_HAND",
    "PLAYED_KNIGHT", "PLAYED_YEAR_OF_PLENTY", "PLAYED_MONOPOLY", "PLAYED_ROAD_BUILDING",
]
# production block: 5 resource pips + total + diversity
PROD_SIZE = len(RESOURCES) + 2
PLAYER_BLOCK_SIZE = len(PER_PLAYER_STATE_KEYS) + PROD_SIZE
GLOBAL_SIZE = len(RESOURCES) + 3  # bank(5) + dev_deck_size + num_turns + is_initial_build_phase
NUM_PLAYERS = 4
FEATURE_SIZE = PLAYER_BLOCK_SIZE * NUM_PLAYERS + GLOBAL_SIZE


def player_production(state, color):
    """Expected resources/roll for `color`: settlements x1, cities x2, by resource."""
    prod = {r: 0.0 for r in RESOURCES}
    node_production = state.board.map.node_production
    for node in get_player_buildings(state, color, SETTLEMENT):
        for r, p in node_production[node].items():
            prod[r] += p
    for node in get_player_buildings(state, color, CITY):
        for r, p in node_production[node].items():
            prod[r] += 2.0 * p
    vec = [prod[r] for r in RESOURCES]
    return vec + [sum(vec), float(sum(1 for r in RESOURCES if prod[r] > 0))]


def _player_block(state, seat_index, color):
    ps = state.player_state
    block = [float(ps[f"P{seat_index}_{k}"]) for k in PER_PLAYER_STATE_KEYS]
    block += player_production(state, color)
    return block


def seat_order_vector(state):
    """Canonical vector: blocks in seat order (perspective = seat 0)."""
    feats = []
    for seat, color in enumerate(state.colors):
        feats += _player_block(state, seat, color)
    bank = list(state.resource_freqdeck)[:len(RESOURCES)]
    feats += [float(x) for x in bank]
    feats.append(float(len(state.development_listdeck)))
    feats.append(float(state.num_turns))
    feats.append(1.0 if state.is_initial_build_phase else 0.0)
    return np.asarray(feats, dtype=np.float32)


def perspective_vector(state, color):
    """Vector from `color`'s perspective (their player block first)."""
    idx = state.color_to_index[color]
    return roll_perspective(seat_order_vector(state), idx)


def roll_perspective(vec, seat_index):
    """Given a seat-order vector, reorder the 4 player blocks so `seat_index`
    is first. Global features (the tail) are untouched."""
    if seat_index == 0:
        return vec
    bs = PLAYER_BLOCK_SIZE
    blocks = [vec[i * bs:(i + 1) * bs] for i in range(NUM_PLAYERS)]
    order = [(seat_index + o) % NUM_PLAYERS for o in range(NUM_PLAYERS)]
    rolled = np.concatenate([blocks[o] for o in order] + [vec[bs * NUM_PLAYERS:]])
    return rolled.astype(np.float32)
