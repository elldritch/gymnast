from abc import abstractmethod
from collections.abc import Callable
from typing import SupportsFloat, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from gymnast.cli import InputListener
from gymnast.utils import set_seeds


class PolicyGradientAgent(nn.Module):
    """
    Base class for agents that use policy gradient descent. Policy gradient
    agents expose an action distribution for calculating gradient.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, observation: torch.Tensor) -> torch.distributions.Distribution:
        """
        Given an observation, return an action distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, observation: torch.Tensor) -> int | NDArray[np.float32]:
        """
        Given an observation, return an action.
        """
        raise NotImplementedError


def explore_one_episode[
    Observation: NDArray[np.float32] | NDArray[np.float64],
    Action: int | NDArray[np.float32],
    Reward: SupportsFloat,
](
    env: gym.Env[Observation, Action],
    agent: PolicyGradientAgent,
    render_step: Callable[[Observation, Action, Reward], None] | None,
) -> list[tuple[Observation, Action, Reward]]:
    steps: list[tuple[Observation, Action, Reward]] = []

    observation, _ = env.reset()
    episode_over = False
    while not episode_over:
        observation = observation.astype(np.float32)
        action = agent.sample(torch.from_numpy(observation.copy()).to("cuda"))
        action = cast(Action, action)

        next_observation, reward, terminated, truncated, _ = env.step(action)
        reward = cast(Reward, reward)

        steps.append((observation.copy(), action, reward))
        observation = next_observation

        if render_step is not None:
            render_step(observation, action, reward)
            if terminated:
                print("Terminated")
            if truncated:
                print("Truncated")

        episode_over = terminated or truncated

    return steps


def train_one_epoch[
    Observation: NDArray[np.float32] | NDArray[np.float64],
    Action: int | NDArray[np.float32],
    Reward: SupportsFloat,
](
    env: gym.Env[Observation, Action],
    agent: PolicyGradientAgent,
    optimizer: torch.optim.Optimizer,
    gradient_fn: Callable[
        [PolicyGradientAgent, list[list[tuple[Observation, Action, Reward]]]],
        torch.Tensor,
    ],
    epoch_batch_size: int,
    render_step: Callable[[Observation, Action, Reward], None] | None = None,
) -> tuple[torch.Tensor, list[list[tuple[Observation, Action, Reward]]]]:
    episodes: list[list[tuple[Observation, Action, Reward]]] = []
    steps_count = 0

    while steps_count < epoch_batch_size:
        episode = explore_one_episode(env, agent, render_step)
        episodes.append(episode)
        steps_count += len(episode)

    optimizer.zero_grad()
    gradient = gradient_fn(agent, episodes)
    gradient.backward()
    optimizer.step()

    return (gradient, episodes)


def train[
    Observation: NDArray[np.float32] | NDArray[np.float64],
    Action: int | NDArray[np.float32],
    Reward: SupportsFloat,
](
    env_id: str,
    agent: PolicyGradientAgent,
    optimizer: torch.optim.Optimizer,
    gradient_fn: Callable[
        [PolicyGradientAgent, list[list[tuple[Observation, Action, Reward]]]],
        torch.Tensor,
    ],
    render_step: Callable[[Observation, Action, Reward], None] | None,
    save_progress: Callable[[PolicyGradientAgent, int], None],
    start_epoch: int,
    epochs_to_train: int,
    seed: int,
    epoch_batch_size: int,
):
    # Initialize environment.
    env: gym.Env = gym.make(env_id, render_mode=None)
    set_seeds(seed, start_epoch, env)
    agent.train()
    current_epoch = start_epoch

    # Save checkpoint on keypress.
    save_listener = InputListener(lambda: save_progress(agent, current_epoch))

    # Run training.
    print(f" Epoch | {"Gradient":>12} | {"Return":>12} | {"Duration":>12}")
    save_listener.start()
    for i in range(epochs_to_train):
        gradient, episodes = train_one_epoch(
            env, agent, optimizer, gradient_fn, epoch_batch_size, render_step
        )
        returns = [
            sum([float(reward) for (_, _, reward) in episode]) for episode in episodes
        ]
        durations = [len(episode) for episode in episodes]
        current_epoch = i + start_epoch
        print(
            " | ".join(
                [
                    f" {current_epoch:5d}",
                    f"{gradient:>12.3f}",
                    f"{np.mean(returns):>12.3f}",
                    f"{np.mean(durations):>12.3f}",
                ]
            )
        )
        if current_epoch % 200 == 0 and current_epoch > 0:
            save_progress(agent, current_epoch)
    print("Done")
    save_listener.stop()
    env.close()

    # Save results.
    save_progress(agent, current_epoch)

    # Run final model.
    env = gym.make(env_id, render_mode="human")
    explore_one_episode(env, agent, render_step)
    env.close()


def infer[
    Observation: NDArray[np.float32] | NDArray[np.float64],
    Action: int | NDArray[np.float32],
    Reward: SupportsFloat,
](
    env_id: str,
    agent: PolicyGradientAgent,
    seed: int,
    render_step: Callable[[Observation, Action, Reward], None] | None,
):
    # Initialize environment.
    env: gym.Env[Observation, Action] = gym.make(env_id, render_mode="human")
    set_seeds(seed, 0, env)
    agent.eval()

    # Run inference.
    steps = explore_one_episode(env, agent, render_step)
    print(f"Return: {sum([float(reward) for (_, _, reward) in steps])}")
    env.close()
