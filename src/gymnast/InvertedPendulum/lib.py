from typing import SupportsFloat

import numpy as np


def print_step(
    action: float,
    observation: np.ndarray,
    reward: SupportsFloat,
    synthetic_reward: SupportsFloat | None = None,
):
    [x, theta, v, omega] = observation
    print(
        " ".join(
            (
                [
                    f"x: {x: .3f}",
                    f"v: {v: .3f}",
                    f"θ: {theta: .3f}",
                    f"ω: {omega: .3f}",
                ]
                + (
                    [f"reward': {synthetic_reward: 8.3f}"]
                    if synthetic_reward is not None
                    else []
                )
                + [
                    f"action: {action: .3f}",
                ]
            ),
        )
    )
