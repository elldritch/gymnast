from typing import SupportsFloat

import numpy as np
import torch

from gymnast.algorithms.vpg import PolicyGradientAgent


def simple[
    Observation: np.ndarray, Action: int | float, Reward: SupportsFloat
](
    agent: PolicyGradientAgent, episodes: list[list[tuple[Observation, Action, Reward]]]
) -> torch.Tensor:
    observations: list[Observation] = []
    actions: list[Action] = []
    weights: list[float] = []

    for episode in episodes:
        episode_return = 0
        for observation, action, reward in episode:
            observations.append(observation)
            actions.append(action)
            episode_return += float(reward)
        weights += [episode_return] * len(episode)

    observations_tensor = torch.from_numpy(np.stack(observations, axis=0)).to("cuda")
    actions_tensor = torch.as_tensor(actions).to("cuda")
    weights_tensor = torch.as_tensor(weights).to("cuda")

    return -(
        agent.act(observations_tensor).log_prob(actions_tensor) * weights_tensor
    ).mean()


def reward_to_go[
    Observation: np.ndarray, Action: int | float, Reward: SupportsFloat
](
    agent: PolicyGradientAgent, episodes: list[list[tuple[Observation, Action, Reward]]]
) -> torch.Tensor:
    observations: list[Observation] = []
    actions: list[Action] = []
    weights: list[float] = []

    for episode in episodes:
        future_return = 0.0
        episode_observations = []
        episode_actions = []
        episode_weights = []
        for observation, action, reward in episode[::-1]:
            episode_observations.append(observation)
            episode_actions.append(action)
            future_return += float(reward)
            episode_weights.append(future_return)

        observations += episode_observations[::-1]
        actions += episode_actions[::-1]
        weights += episode_weights[::-1]

    observations_tensor = torch.from_numpy(np.stack(observations, axis=0)).to("cuda")
    actions_tensor = torch.as_tensor(actions).to("cuda")
    weights_tensor = torch.as_tensor(weights).to("cuda")

    return -(
        agent.act(observations_tensor).log_prob(actions_tensor) * weights_tensor
    ).mean()
