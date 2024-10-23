import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", render_mode=None)

alpha = 0.1
epsilon = 0.1
gamma = 0.9
q_table = np.zeros((env.observation_space.n, env.action_space.n))
epochs = 500000

for epoch in range(epochs):
    if epoch % 2000 == 0:
        print(q_table)
    observation, _ = env.reset()
    episode_over = False
    while not episode_over:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[observation])

        next_observation, reward, terminated, truncated, _ = env.step(action)
        # print(action, observation, reward)

        q_table[observation, action] = q_table[observation, action] + alpha * (
            reward
            + gamma * np.max(q_table[next_observation])
            - q_table[observation, action]
        )
        observation = next_observation

        episode_over = terminated or truncated
        # if terminated:
        #     print("Terminated")
        # if truncated:
        #     print("Truncated")

env.close()
print(q_table)

env = gym.make("FrozenLake-v1", render_mode="human")
input("Ready to run evaluation?")
observation, _ = env.reset()
episode_over = False
while not episode_over:
    action = np.argmax(q_table[observation])

    next_observation, reward, terminated, truncated, _ = env.step(action)
    print(action, observation, reward)
    observation = next_observation

    episode_over = terminated or truncated
    if terminated:
        print("Terminated")
    if truncated:
        print("Truncated")
