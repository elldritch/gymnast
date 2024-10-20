from typing import SupportsFloat
import numpy as np
import torch

from gymnast.algorithms.vpg import weights
from gymnast.algorithms.vpg.cli import main_base, parse_args_nn_adam
from gymnast.algorithms.vpg.models import DiscreteActionAgent

ENV_ID = "CartPole-v1"


def main():
    args, agent_args, optimizer_args = parse_args_nn_adam(ENV_ID)
    main_base(
        ENV_ID,
        DiscreteActionAgent,
        torch.optim.Adam,
        weights.reward_to_go,
        args,
        agent_args,
        optimizer_args,
        render_step,
    )


def render_step(observation: np.ndarray, action: int, reward: SupportsFloat):
    [x, v, theta, omega] = observation
    action_name = "<" if action == 0 else ">" if action == 1 else "ERROR"
    print(
        " ".join(
            (
                [
                    f"x: {x: .3f}",
                    f"v: {v: .3f}",
                    f"θ: {theta: .3f}",
                    f"ω: {omega: .3f}",
                    f"action: {action_name}",
                ]
            ),
        )
    )


if __name__ == "__main__":
    main()
