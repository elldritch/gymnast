import time
import gymnasium as gym

import numpy as np
import torch
from torch import nn


class NeuralNet(nn.Module):
    def __init__(self, layer_sizes: list[int], activation: nn.Module = nn.Tanh()):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), activation]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    def predict(self, x: torch.Tensor) -> torch.distributions.Categorical:
        prediction: torch.Tensor = self(x)
        return torch.distributions.Categorical(logits=prediction)


def explore_one_episode(
    env: gym.Env, model: NeuralNet
) -> tuple[list[np.ndarray], list[int], list[np.float64]]:
    observations: list[np.ndarray] = []
    actions: list = []
    rewards: list = []

    observation: np.ndarray
    observation, _ = env.reset()
    episode_over = False
    while not episode_over:
        observations.append(observation.copy())

        action = model.predict(torch.from_numpy(observation).to("cuda")).sample().item()
        actions.append(action)

        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        episode_over = terminated or truncated

    return (observations, actions, rewards)


def train_one_epoch(
    env: gym.Env,
    model: NeuralNet,
    optimizer: torch.optim.Optimizer,  # pyright: ignore [reportPrivateImportUsage]
    epoch_batch_size=5000,
):
    observations: list[np.ndarray] = []
    actions: list[int] = []
    returns: list = []
    durations: list = []
    weights: list = []

    while len(observations) < epoch_batch_size:
        episode_observations, episode_actions, episode_rewards = explore_one_episode(
            env, model
        )
        episode_return = sum(episode_rewards)
        duration = len(episode_rewards)

        observations += episode_observations
        actions += episode_actions
        returns.append(episode_return)
        durations.append(duration)
        weights += [episode_return] * duration

    observations_tensor: torch.Tensor = torch.from_numpy(
        np.stack(observations, axis=0)
    ).to("cuda")
    actions_tensor: torch.Tensor = torch.as_tensor(actions).to("cuda")
    weights_tensor: torch.Tensor = torch.as_tensor(weights).to("cuda")

    optimizer.zero_grad()
    loss: torch.Tensor = -(model.predict(observations_tensor).log_prob(actions_tensor) * weights_tensor).mean()
    loss.backward()
    optimizer.step()

    return (loss, returns, durations)


if __name__ == "__main__":
    # Hyperparameters.
    env_id = "LunarLander-v3"
    epochs = 50
    batch_size = 5000
    learning_rate = 1e-2

    # Initialize environment.
    env: gym.Env = gym.make(env_id, render_mode=None)
    observation_dim: int = (
        env.observation_space.shape  # pyright: ignore [reportOptionalSubscript]
    )[0]
    actions_dim: int = (
        env.action_space.n  # pyright: ignore [reportAttributeAccessIssue]
    )

    # Initialize model and optimizer.
    model = NeuralNet([observation_dim, 32, actions_dim]).to("cuda")
    optimizer = torch.optim.Adam(  # pyright: ignore [reportPrivateImportUsage]
        model.parameters(), lr=1e-3
    )

    # Run training.
    print(f" Epoch | {"Loss":>8} | {"Return":>8} | Duration")
    for i in range(epochs):
        loss, returns, durations = train_one_epoch(env, model, optimizer, batch_size)
        print(f" {i:5d} | {loss:>5.3f} | {np.mean(returns):>5.3f} | {np.mean(durations):>5.3f}")
    env.close()

    # Run final model.
    env = gym.make(env_id, render_mode="human")
    explore_one_episode(env, model)
    env.close()

    # Save results.
    torch.save(model.state_dict(), f"model_{time.time()}.pt")
