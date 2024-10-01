import argparse

import gymnasium as gym
import numpy as np


def set_seeds(seed: int, env: gym.Env):
    np.random.seed(seed)
    env.reset(seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    env: gym.Env = gym.make("LunarLander-v3", render_mode="human")

    seed = args.seed if args.seed is not None else 42
    set_seeds(seed, env)

    observation: np.ndarray
    observation, _ = env.reset()
    episode_over = False
    while not episode_over:
        action = int(input())
        observation, reward, terminated, truncated, _ = env.step(action)

        action_name = (
            "_"
            if action == 0
            else (
                "<"
                if action == 1
                else "^" if action == 2 else ">" if action == 3 else "ERROR"
            )
        )
        r = "R" if observation[6] == 1.0 else " " if observation[6] == 0.0 else "ERROR"
        l = "L" if observation[7] == 1.0 else " " if observation[7] == 0.0 else "ERROR"
        print(
            f"x: {observation[0]: .3f} y: {observation[1]: .3f} x': {observation[2]: .3f} y': {observation[3]: .3f} θ: {observation[4]: .3f} ω: {observation[5]: .3f} {r} {l} reward: {reward: 8.3f} action: {action_name}"
        )

        episode_over = terminated or truncated
