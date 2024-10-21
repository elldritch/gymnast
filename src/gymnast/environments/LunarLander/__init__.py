from typing import SupportsFloat

import numpy as np


def render_step(observation: np.ndarray, action: int, reward: SupportsFloat):
    [x, y, dx, dy, theta, omega, r, l] = observation
    d = np.sqrt(x**2 + y**2)
    v = np.sqrt(dx**2 + dy**2)
    action_name = (
        "_"
        if action == 0
        else (
            "<"
            if action == 1
            else "^" if action == 2 else ">" if action == 3 else "ERROR"
        )
    )
    r_show = "R" if r == 1.0 else " " if r == 0.0 else "ERROR"
    l_show = "L" if l == 1.0 else " " if l == 0.0 else "ERROR"
    print(
        " ".join(
            (
                [
                    f"x: {x: .3f}",
                    f"vx: {dx: .3f}",
                    f"y: {y: .3f}",
                    f"vy: {dy: .3f}",
                    f"d: {d: .3f}",
                    f"v: {v: .3f}",
                    f"θ: {theta: .3f}",
                    f"ω: {omega: .3f}",
                    r_show,
                    l_show,
                    f"reward: {reward: 8.3f}",
                    f"action: {action_name}",
                ]
            ),
        )
    )
