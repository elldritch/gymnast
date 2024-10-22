from argparse import ArgumentParser, Namespace
from typing import Any, Callable, SupportsFloat
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import torch

from gymnast.algorithms.vpg import (
    PolicyGradientAgent,
    explore_one_episode,
    train,
    Observation,
    Action,
)
from gymnast.algorithms.vpg.io import load_checkpoint, save_checkpoint
from gymnast.utils import dimensions, set_seeds


def argparser_base() -> tuple[ArgumentParser, ArgumentParser]:
    # Parse hyperparameters.
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    trainP = subparsers.add_parser("train")
    trainP.add_argument("--seed", type=int)
    trainP.add_argument("--epochs", type=int)
    trainP.add_argument("--batch_size", type=int)
    trainP.add_argument("--checkpoint_folder", type=str, required=True)
    trainP.add_argument("--load_save_id", type=str)
    trainP.add_argument("--save_id", type=str, required=True)

    inferP = subparsers.add_parser("infer")
    inferP.add_argument("--seed", type=int)
    inferP.add_argument("--checkpoint_folder", type=str, required=True)
    inferP.add_argument("--save_id", type=str, required=True)
    inferP.add_argument("--video_folder", type=str, default=None)

    return (parser, trainP)


def parse_args_nn_adam(
    env_id: str,
) -> tuple[Namespace, list[Any], list[Any]]:
    parser, trainP = argparser_base()
    trainP.add_argument("--learning_rate", type=float)
    trainP.add_argument("--hidden_layers", type=int, nargs="+")

    args = parser.parse_args()

    if args.cmd == "train":
        hidden_layers = args.hidden_layers or [64, 64]
        assert isinstance(hidden_layers, list)

        learning_rate = args.learning_rate or 1e-3
        assert isinstance(learning_rate, float)

        env = gym.make(env_id)
        observation_dim, action_dim = dimensions(env)

        return args, [[observation_dim, *hidden_layers, action_dim]], [learning_rate]
    else:
        return args, [], []


def main_base(
    env_id: str,
    agent_class: type[PolicyGradientAgent],
    optimizer_class: type[torch.optim.Optimizer],
    weights_fn: Callable[[list[np.float32]], list[float]],
    args: Namespace,
    agent_args_cli: list[Any],
    optimizer_args_cli: list[Any],
    render_step: Callable[[Observation, Action, np.float32], None],
):
    if args.cmd == "train":
        # Load from checkpoint.
        checkpoint = None
        if args.load_save_id:
            checkpoint = load_checkpoint(
                args.checkpoint_folder,
                args.load_save_id,
                agent_class,
                optimizer_class,
                weights_fn,
            )
            assert checkpoint.env == env_id

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

        weights_fn = checkpoint.weights_fn if checkpoint else weights_fn

        # Initialize environment and set seeds. We must seed before initializing
        # the agent and optimizer so that those are reproducibly initialized
        # when training from scratch.
        env = gym.make(env_id, render_mode=None)
        set_seeds(seed, start_epoch, env)

        # Initialize model.
        agent_args: list[Any] = checkpoint.agent_args if checkpoint else agent_args_cli
        agent = agent_class(*agent_args).to("cuda")
        if checkpoint:
            agent.load_state_dict(checkpoint.agent_state_dict)
        agent.train()

        # Initialize optimizer.
        optimizer_args: list[Any] = (
            checkpoint.optimizer_args if checkpoint else optimizer_args_cli
        )
        optimizer = optimizer_class(agent.parameters(), *optimizer_args)
        if checkpoint:
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        # Reset seeds after initialization. Otherwise, runs that use newly
        # initialized models will be inconsistent from runs that load
        # checkpoints.
        set_seeds(seed, start_epoch, env)

        def save_progress(agent: PolicyGradientAgent, current_epoch: int):
            save_checkpoint(
                env_id,
                agent,
                agent_args,
                optimizer,
                optimizer_args,
                weights_fn,
                seed,
                epoch_batch_size,
                current_epoch,
                args.checkpoint_folder,
                args.save_id,
            )

        # Train.
        print(
            " ".join(
                [
                    "Initializing training with parameters:",
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
            env,
            agent,
            optimizer,
            weights_fn,
            None,
            save_progress,
            start_epoch,
            epochs_to_train,
            epoch_batch_size,
        )

        # Run final model.
        agent.eval()
        env = gym.make(env_id, render_mode="human")
        explore_one_episode(env, agent, render_step)
        env.close()
    elif args.cmd == "infer":
        # Load from checkpoint.
        checkpoint = load_checkpoint(
            args.checkpoint_folder,
            args.save_id,
            agent_class,
            optimizer_class,
            weights_fn,
        )
        assert checkpoint.env == env_id

        # Parse hyperparameters that can be loaded from checkpoint.
        seed = args.seed or checkpoint.seed
        assert isinstance(seed, int)

        # Initialize environment.
        env = gym.make(env_id, render_mode="human")
        if args.video_folder:
            env = gym.wrappers.RecordVideo(
                gym.make(env_id, render_mode="rgb_array"),
                args.video_folder,
                name_prefix=f"{env_id}_{checkpoint.elapsed_epochs}_{seed}",
            )
        set_seeds(seed, 0, env)

        # Initialize model.
        agent = agent_class(*checkpoint.agent_args).to("cuda")
        agent.load_state_dict(checkpoint.agent_state_dict)
        agent.eval()

        # Run inference.
        print(
            " ".join(
                [
                    "Initializing inference with parameters:",
                    f"seed={seed}",
                    f"agent_args={checkpoint.agent_args}",
                    f"elapsed_epochs={checkpoint.elapsed_epochs}",
                ]
            )
        )
        # Run inference.
        observations, actions, rewards = explore_one_episode(env, agent, render_step)
        print(f"Return: {sum(rewards)}")
        env.close()
    else:
        raise NotImplementedError(f"Unknown subcommand: {args.cmd}")
