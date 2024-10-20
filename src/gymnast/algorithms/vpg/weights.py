import numpy as np


def simple(episode_rewards: list[np.float32]) -> list[float]:
    episode_return = sum(map(float, episode_rewards))
    return [episode_return] * len(episode_rewards)


def reward_to_go(episode_rewards: list[np.float32]) -> list[float]:
    weights: list[float] = []

    future_return = 0
    weights = []
    for reward in episode_rewards[::-1]:
        future_return += float(reward)
        weights.append(future_return)

    return weights[::-1]
