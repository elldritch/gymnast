from dataclasses import asdict
from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
import torch

from gymnast.algorithms.vpg import PolicyGradientAgent, argparser, infer, train
from gymnast.algorithms.vpg import gradient
from gymnast.algorithms.vpg.io import load_checkpoint, save_checkpoint
from gymnast.algorithms.vpg.models import DiscreteActionAgent
from gymnast.utils import dimensions

ENV_ID = "CartPole-v1"


def main():
    parser, _ = argparser()
    args = parser.parse_args()

    if args.cmd == "train":
        # Load from checkpoint.
        checkpoint = None
        if args.load_save_id:
            checkpoint = load_checkpoint(
                args.checkpoint_folder,
                args.load_save_id,
                DiscreteActionAgent,
                torch.optim.Adam,
                gradient.reward_to_go,
            )
            assert checkpoint.env == ENV_ID

        # Parse hyperparameters set only by flags.
        epochs_to_train = args.epochs or 100
        assert isinstance(epochs_to_train, int)

        # Parse hyperparameters that can be loaded from checkpoint.
        seed = args.seed or (checkpoint.seed if checkpoint else 42)
        assert isinstance(seed, int)

        epoch_batch_size = args.batch_size or (
            checkpoint.batch_size if checkpoint else 10000
        )
        assert isinstance(epoch_batch_size, int)

        start_epoch = checkpoint.elapsed_epochs if checkpoint else 0
        assert isinstance(start_epoch, int)

        gradient_fn = checkpoint.gradient_fn if checkpoint else gradient.reward_to_go

        # Parse flags with defaults.
        hidden_layers = args.hidden_layers or [64, 64]
        assert isinstance(hidden_layers, list)

        learning_rate = args.learning_rate or 1e-3
        assert isinstance(learning_rate, float)

        # Initialize environment.
        env = gym.make(ENV_ID)
        observation_dim, action_dim = dimensions(env)

        # Initialize model.
        agent_args: list[Any] = (
            checkpoint.agent_args
            if checkpoint
            else [[observation_dim, *hidden_layers, action_dim]]
        )
        agent = DiscreteActionAgent(*agent_args).to("cuda")
        if checkpoint:
            agent.load_state_dict(checkpoint.agent_state_dict)

        # Initialize optimizer.
        optimizer_args: list[Any] = (
            checkpoint.optimizer_args if checkpoint else [learning_rate]
        )
        optimizer = torch.optim.Adam(agent.parameters(), *optimizer_args)
        if checkpoint:
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        def save_progress(agent: PolicyGradientAgent, current_epoch: int):
            save_checkpoint(
                ENV_ID,
                agent,
                agent_args,
                optimizer,
                optimizer_args,
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
                    f"agent_args={agent_args}",
                    f"optimizer_args={optimizer_args}",
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
