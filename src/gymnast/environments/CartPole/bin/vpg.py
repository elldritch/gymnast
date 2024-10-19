from typing import SupportsFloat
import gymnasium as gym
import numpy as np
import torch

from gymnast.algorithms.vpg import PolicyGradientAgent, argparser, infer, train
from gymnast.algorithms.vpg import gradient
from gymnast.algorithms.vpg.io import save_checkpoint
from gymnast.algorithms.vpg.models import DiscreteActionAgent
from gymnast.utils import dimensions


def main():
    parser, _ = argparser()
    args = parser.parse_args()

    if args.cmd == "train":
        # Parse hyperparameters.
        # TODO: Loading from checkpoint.
        hidden_layers = args.hidden_layers or [64, 64]
        assert isinstance(hidden_layers, list)

        learning_rate = args.learning_rate or 1e-3
        assert isinstance(learning_rate, float)

        seed = args.seed or 42
        assert isinstance(seed, int)

        epochs_to_train = args.epochs or 100
        assert isinstance(epochs_to_train, int)

        epoch_batch_size = args.batch_size or 10000
        assert isinstance(epoch_batch_size, int)

        # TODO: Loading from checkpoint.
        start_epoch = 0

        # Initialize environment.
        env = gym.make("CartPole-v1")
        observation_dim, action_dim = dimensions(env)

        # Initialize model.
        agent_arg = [observation_dim, *hidden_layers, action_dim]
        agent = DiscreteActionAgent(agent_arg).to("cuda")
        optimizer_arg = learning_rate
        optimizer = torch.optim.Adam(agent.parameters(), optimizer_arg)
        gradient_fn = gradient.reward_to_go

        def save_progress(agent: PolicyGradientAgent, current_epoch: int):
            save_checkpoint(
                "CartPole-v1",
                agent,
                [agent_arg],
                optimizer,
                [optimizer_arg],
                gradient_fn,
                seed,
                epoch_batch_size,
                current_epoch,
                args.checkpoint_folder,
                args.save_id,
            )
            pass

        # Train.
        print(
            " ".join(
                [
                    "Initializing with parameters:",
                    f"seed={seed}",
                    f"start_epoch={start_epoch}",
                    f"epochs_to_train={epochs_to_train}",
                    f"epoch_batch_size={epoch_batch_size}",
                    f"learning_rate={learning_rate}",
                    f"hidden_layers={hidden_layers}",
                ]
            )
        )
        train(
            "CartPole-v1",
            agent,
            optimizer,
            gradient_fn,
            None,
            save_progress,
            start_epoch,
            epochs_to_train,
            seed,
            epoch_batch_size,
        )
    elif args.cmd == "infer":
        # infer(args.load_from, args.seed)
        pass
    else:
        raise NotImplementedError(f"Unknown subcommand: {args.cmd}")
    pass


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
