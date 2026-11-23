"""Known transition model for HW6; floor rules live in data/maps.json."""

import json
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent / "data" / "maps.json"
DIRECTIONS = {"up": (-1, 0), "right": (0, 1),
              "down": (1, 0), "left": (0, -1)}
ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←"}


def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def coordinate(state):
    return f"({state[0] + 1},{state[1] + 1})"


class GridWorld:
    def __init__(self, config, map_key):
        self.config = config
        self.name = map_key
        self.rows = config[map_key]
        self.rules = config["tiles"]
        self.actions = tuple(config["actions"])
        self.discount = config["discount"]
        self.states = tuple((r, c) for r, row in enumerate(self.rows)
                            for c, tile in enumerate(row)
                            if self.rules[tile].get("traversable", True))
        self.state_set = set(self.states)
        self.nonterminal = tuple(s for s in self.states if not self.terminal(s))
        self.start = next(s for s in self.states if self.tile(s) == "S")

    def tile(self, state):
        return self.rows[state[0]][state[1]]

    def terminal(self, state):
        return self.rules[self.tile(state)].get("terminal", False)

    def transitions(self, state, action):
        """Return (probability, destination, reward) for every movement outcome.

        Keep slip branches separate even if multiple branches hit the same cell.
        Terminal self-loops have zero reward, so G/H values remain zero.
        """
        if state not in self.state_set:
            raise ValueError("A wall or outside coordinate is not a state")
        if action not in self.actions:
            raise ValueError(f"Unknown action: {action}")
        if self.terminal(state):
            return [(1.0, state, 0.0)]
        rule = self.rules[self.tile(state)]
        if rule["movement"] == "slippery":
            index = self.actions.index(action)
            movements = [(rule["intended"], action),
                         (rule["perpendicular_each"], self.actions[(index - 1) % 4]),
                         (rule["perpendicular_each"], self.actions[(index + 1) % 4])]
        else:
            movements = [(1.0, action)]
        outcomes = []
        for probability, direction in movements:
            dr, dc = DIRECTIONS[direction]
            destination = (state[0] + dr, state[1] + dc)
            if destination not in self.state_set:
                destination = state
            reward = self.rules[self.tile(destination)]["reward"]
            outcomes.append((probability, destination, float(reward)))
        return outcomes

    def q(self, state, action, values):
        return sum(p * (reward + self.discount * values[destination])
                   for p, destination, reward in self.transitions(state, action))

    def value_map(self, values):
        lines = []
        for r, row in enumerate(self.rows):
            cells = []
            for c, tile in enumerate(row):
                label = "#" if (r, c) not in self.state_set else f"{values[(r, c)]:.9f}"
                cells.append(f"{label:>14}")
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def policy_map(self, policy):
        return "\n".join(" ".join(
            self.tile((r, c)) if (r, c) not in policy else ARROWS[policy[(r, c)]]
            for c in range(len(row))) for r, row in enumerate(self.rows))
