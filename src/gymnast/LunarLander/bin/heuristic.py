import argparse

import gymnasium as gym
import numpy as np

from gymnast.LunarLander.lib import print_step


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
    episode_return = 0
    while not episode_over:
        [x, y, dx, dy, theta, omega, r, l] = observation

        # angle should point towards center
        angle_target = x * 0.5 + dx * 1.0
        # more than 0.4 radians (22 degrees) is bad
        if angle_target > 0.4:
            angle_target = 0.4
        if angle_target < -0.4:
            angle_target = -0.4
        # target y should be proportional to horizontal offset
        hover_target = 0.75 * np.abs(x)

        angle_todo = (angle_target - theta) * 0.5 - omega * 1.0
        hover_todo = (hover_target - y) * 0.5 - dy * 0.5

        if r or l:  # legs have contact
            angle_todo = 0
            # override to reduce fall speed, that's all we need after contact
            hover_todo = -dy * 0.5

        action = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            action = 2
        elif angle_todo < -0.05:
            action = 3
        elif angle_todo > 0.05:
            action = 1
        elif y < 0.01 and np.abs(dx) > 0.08:
            if x < -0.1 and dx < 0:
                action = 3
            elif x > 0.1 and dx > 0:
                action = 1

        observation, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)
        episode_over = terminated or truncated

        print_step(action, observation, reward)

    print(f"Return: {episode_return}")


if __name__ == "__main__":
    main()
