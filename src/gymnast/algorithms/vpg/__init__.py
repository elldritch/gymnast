from abc import abstractmethod
from collections.abc import Callable
from typing import SupportsFloat, TypeVar, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from gymnast.cli import InputListener
from gymnast.utils import dimensions


Observation = TypeVar("Observation", bound=NDArray[np.float32] | NDArray[np.float64])
Action = TypeVar("Action", bound=int | NDArray[np.float32])
Reward = TypeVar("Reward", bound=SupportsFloat)


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
    def sample(
        self, observation: torch.Tensor
    ) -> Action:  # pyright: ignore [reportInvalidTypeVarUse]
        """
        Given an observation, return an action.
        """
        raise NotImplementedError


def explore_one_episode(
    env: gym.Env[Observation, Action],
    agent: PolicyGradientAgent,
    render_step: Callable[[Observation, Action, np.float32], None] | None,
) -> tuple[list[NDArray[np.float32]], list[np.float32], list[np.float32]]:
    observations_dim, _ = dimensions(env)

    observations: list[NDArray[np.float32]] = []
    actions: list[np.float32] = []
    rewards: list[np.float32] = []

    observation, _ = env.reset()
    episode_over = False
    while not episode_over:
        observation = observation.astype(np.float32)
        assert (observations_dim,) == observation.shape
        observations.append(observation.copy())

        action = agent.sample(torch.from_numpy(observation).to("cuda"))
        if isinstance(action, np.ndarray):
            assert len(action) == 1
            item = action[0]
            assert type(item) == np.float32
            actions.append(item)
        elif isinstance(action, int):
            actions.append(np.float32(action))
        else:
            raise ValueError(f"Unknown action type: {type(action)}")
        action = cast(Action, action)

        observation, reward, terminated, truncated, _ = env.step(action)
        reward = np.float32(reward)
        rewards.append(reward)

        if render_step is not None:
            render_step(observation, action, reward)
            if terminated:
                print("Terminated")
            if truncated:
                print("Truncated")

        episode_over = terminated or truncated

    assert len(observations) == len(actions) == len(rewards)
    return observations, actions, rewards


def train_one_epoch(
    env: gym.Env[Observation, Action],
    agent: PolicyGradientAgent,
    optimizer: torch.optim.Optimizer,
    weights_fn: Callable[[list[np.float32]], list[float]],
    epoch_batch_size: int,
    render_step: Callable[[Observation, Action, np.float32], None] | None = None,
) -> tuple[
    torch.Tensor, list[list[tuple[NDArray[np.float32], np.float32, np.float32]]]
]:
    episodes: list[list[tuple[NDArray[np.float32], np.float32, np.float32]]] = []
    epoch_observations: list[NDArray[np.float32]] = []
    epoch_actions: list[np.float32] = []
    epoch_weights: list[float] = []

    while len(epoch_observations) < epoch_batch_size:
        episode_observations, episode_actions, episode_rewards = explore_one_episode(
            env, agent, render_step
        )
        episodes.append(
            list(zip(episode_observations, episode_actions, episode_rewards))
        )

        epoch_observations += episode_observations
        epoch_actions += episode_actions
        epoch_weights += weights_fn(episode_rewards)

    observations_tensor = torch.from_numpy(np.stack(epoch_observations, axis=0)).to(
        "cuda"
    )
    actions_tensor = torch.as_tensor(epoch_actions).to("cuda")
    weights_tensor = torch.as_tensor(epoch_weights).to("cuda")

    optimizer.zero_grad()
    gradient = -(
        agent.predict(observations_tensor).log_prob(actions_tensor) * weights_tensor
    ).mean()
    gradient.backward()
    optimizer.step()

    return (gradient, episodes)


def train(
    env: gym.Env[Observation, Action],
    agent: PolicyGradientAgent,
    optimizer: torch.optim.Optimizer,
    weight_fn: Callable[[list[np.float32]], list[float]],
    render_step: Callable[[Observation, Action, np.float32], None] | None,
    save_progress: Callable[[PolicyGradientAgent, int], None],
    start_epoch: int,
    epochs_to_train: int,
    epoch_batch_size: int,
):
    # Save checkpoint on keypress.
    save_listener = InputListener(lambda: save_progress(agent, current_epoch))

    # Run training.
    print(f" Epoch | {"Gradient":>12} | {"Return":>12} | {"Duration":>12}")
    current_epoch = start_epoch
    save_listener.start()
    for i in range(epochs_to_train):
        gradient, episodes = train_one_epoch(
            env, agent, optimizer, weight_fn, epoch_batch_size, render_step
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
        if current_epoch % 20 == 0 and current_epoch > 0:
            save_progress(agent, current_epoch)
    print("Done")
    save_listener.stop()
    env.close()

    # Save results.
    save_progress(agent, current_epoch)
