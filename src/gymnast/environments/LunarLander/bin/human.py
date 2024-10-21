import argparse

import gymnasium as gym
import numpy as np

from gymnast.environments.LunarLander import render_step


def set_seeds(seed: int, env: gym.Env):
    np.random.seed(seed)
    env.reset(seed=seed)


def main():
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
        print("Input action (0 for none, 1 for left, 2 for up, 3 for right): ", end="")
        action = input()
        if action not in ["0", "1", "2", "3"]:
            print(f"Invalid action: {action}")
            continue
        action = int(action)

        observation, reward, terminated, truncated, _ = env.step(action)
        episode_over = terminated or truncated

        render_step(observation, action, reward)


if __name__ == "__main__":
    main()
