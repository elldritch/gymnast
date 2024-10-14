from typing import SupportsFloat

import numpy as np


def print_step(action: int, observation: np.ndarray, reward: SupportsFloat):
    action_name = (
        "_"
        if action == 0
        else (
            "<"
            if action == 1
            else "^" if action == 2 else ">" if action == 3 else "ERROR"
        )
    )
    r = "R" if observation[6] == 1.0 else " " if observation[6] == 0.0 else "ERROR"
    l = "L" if observation[7] == 1.0 else " " if observation[7] == 0.0 else "ERROR"
    print(
        f"x: {observation[0]: .3f} y: {observation[1]: .3f} x': {observation[2]: .3f} y': {observation[3]: .3f} θ: {observation[4]: .3f} ω: {observation[5]: .3f} {r} {l} reward: {reward: 8.3f} action: {action_name}"
    )
