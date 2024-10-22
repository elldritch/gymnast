import numpy as np
import torch
from torch import nn

from gymnast.algorithms.vpg import weights
from gymnast.algorithms.vpg.cli import main_base, parse_args_nn_adam
from gymnast.algorithms.vpg.models import DiscreteActionAgent
from gymnast.environments.LunarLander import render_step

ENV_ID = "LunarLander-v3"


# This is a hack to get the lander to actually stop once it lands. Not sure how
# to get it to learn that using VPG.
class LunarLanderAgent(DiscreteActionAgent):
    def __init__(self, layer_sizes: list[int], activation: nn.Module = nn.Tanh()):
        super().__init__(layer_sizes, activation)
        self.landed = False

    # Works great as a post-training add-on, but do NOT train VPG with this,
    # it'll mess things up real bad.
    def sample(self, observation: torch.Tensor) -> int:
        [x, y, dx, dy, theta, omega, r, l] = observation
        if torch.abs(dy) < 0.02 and torch.abs(dx) < 0.02 and r and l:
            self.landed = True
        if self.landed:
            return 0
        return int(self.predict(observation).sample().item())


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


if __name__ == "__main__":
    main()
