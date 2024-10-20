from typing import SupportsFloat
import numpy as np
import torch

from gymnast.algorithms.vpg import gradient
from gymnast.algorithms.vpg.cli import main_base, parse_args_nn_adam
from gymnast.algorithms.vpg.models import ContinuousActionAgent

ENV_ID = "InvertedPendulum-v5"


def main():
    args, agent_args, optimizer_args = parse_args_nn_adam(ENV_ID)
    main_base(
        ENV_ID,
        ContinuousActionAgent,
        torch.optim.Adam,
        gradient.reward_to_go,
        args,
        agent_args,
        optimizer_args,
        render_step,
    )


def render_step(observation: np.ndarray, action: np.ndarray, reward: SupportsFloat):
    [x, theta, v, omega] = observation
    print(
        " ".join(
            (
                [
                    f"x: {x: .3f}",
                    f"v: {v: .3f}",
                    f"θ: {theta: .3f}",
                    f"ω: {omega: .3f}",
                    f"action: {action[0]: .3f}",
                ]
            ),
        )
    )


if __name__ == "__main__":
    main()
