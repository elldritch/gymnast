import gymnasium as gym
import numpy as np
import torch


def dimensions(env: gym.Env) -> tuple[int, int]:
    observations_dim = _space_dimensions(env.observation_space)
    actions_dim = _space_dimensions(env.action_space)
    return (observations_dim, actions_dim)


def _space_dimensions(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    elif isinstance(space, gym.spaces.Box):
        return space.shape[0]
    else:
        raise NotImplementedError("Space type not supported.")


def set_seeds(seed: int, start_epoch: int, env: gym.Env):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)
    # Otherwise, using the same seed will cause us to replay previously seen
    # examples when we load from checkpoint.
    for _ in range(start_epoch):
        env.reset()