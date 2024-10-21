import torch

from gymnast.algorithms.vpg import weights
from gymnast.algorithms.vpg.cli import main_base, parse_args_nn_adam
from gymnast.algorithms.vpg.models import DiscreteActionAgent
from gymnast.environments.LunarLander import render_step

ENV_ID = "LunarLander-v3"


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
