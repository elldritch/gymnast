import argparse
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
    env: gym.Env, model: NeuralNet, verbose: bool = False
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

        if verbose:
            action_name = (
                "_"
                if action == 0
                else (
                    ">"
                    if action == 1
                    else "^" if action == 2 else "<" if action == 3 else "ERROR"
                )
            )
            l = (
                "L"
                if observation[6] == 1.0
                else " " if observation[6] == 0.0 else "ERROR"
            )
            r = (
                "R"
                if observation[7] == 1.0
                else " " if observation[7] == 0.0 else "ERROR"
            )
            print(
                f"x: {observation[0]: .3f} y: {observation[1]: .3f} x': {observation[2]: .3f} y': {observation[3]: .3f} θ: {observation[4]: .3f} ω: {observation[5]: .3f} {l} {r} reward: {reward: 8.3f} action: {action_name}"
            )

        episode_over = terminated or truncated

    return (observations, actions, rewards)


def train_one_epoch(
    env: gym.Env,
    model: NeuralNet,
    optimizer: torch.optim.Optimizer,  # pyright: ignore [reportPrivateImportUsage]
    epoch_batch_size: int = 5000,
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
    loss: torch.Tensor = -(
        model.predict(observations_tensor).log_prob(actions_tensor) * weights_tensor
    ).mean()
    loss.backward()
    optimizer.step()

    return (loss, returns, durations)


def dimensions(env: gym.Env) -> tuple[int, int]:
    observation_dim: int = (
        env.observation_space.shape  # pyright: ignore [reportOptionalSubscript]
    )[0]
    actions_dim: int = (
        env.action_space.n  # pyright: ignore [reportAttributeAccessIssue]
    )
    return (observation_dim, actions_dim)


def cmd_train(
    env_id: str,
    seed: int | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_layers: list[int],
    save_to: str,
):
    # Initialize environment.
    env: gym.Env = gym.make(env_id, render_mode=None)
    (observation_dim, actions_dim) = dimensions(env)

    # Set seeds.
    torch.use_deterministic_algorithms(True)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.reset(seed=seed)

    # Initialize model and optimizer.
    model = NeuralNet([observation_dim] + hidden_layers + [actions_dim]).to("cuda")
    model.load_state_dict(torch.load("model_1727759757.189204.pt"))
    optimizer = torch.optim.Adam(  # pyright: ignore [reportPrivateImportUsage]
        model.parameters(), lr=learning_rate
    )

    # Run training.
    print(f" Epoch | {"Loss":>8} | {"Return":>8} | Duration")
    for i in range(epochs):
        loss, returns, durations = train_one_epoch(env, model, optimizer, batch_size)
        print(
            f" {i: 5d} | {loss: >5.3f} | {np.mean(returns): >5.3f} | {np.mean(durations): >5.3f}"
        )
    env.close()

    # Run final model.
    env = gym.make(env_id, render_mode="human")
    explore_one_episode(env, model)
    env.close()

    # Save results.
    torch.save(model.state_dict(), save_to)


def cmd_infer(env_id: str, load_from: str):
    env: gym.Env = gym.make(env_id, render_mode="human")

    (observation_dim, actions_dim) = dimensions(env)
    # TODO: Load model parameters too.
    hidden_layers: list[int] = [64, 64]
    model = NeuralNet([observation_dim] + hidden_layers + [actions_dim]).to("cuda")
    model.load_state_dict(torch.load(load_from, weights_only=True))
    model.eval()

    _, _, rewards = explore_one_episode(env, model, verbose=True)
    print(f"Return: {sum(rewards)}")

    env.close()


if __name__ == "__main__":
    # Parse hyperparameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    subparsers = parser.add_subparsers(dest="cmd")

    trainP = subparsers.add_parser("train")
    trainP.add_argument("--epochs", type=int, default=600)
    trainP.add_argument("--batch_size", type=int, default=5000)
    trainP.add_argument("--learning_rate", type=float, default=1e-3)
    trainP.add_argument("--hidden_layers", type=int, nargs="+", default=[64, 64])
    trainP.add_argument("--save_to", type=str, required=True)
    # TODO: Implement resuming training from a saved checkpoint.

    inferP = subparsers.add_parser("infer")
    inferP.add_argument("--load_from", type=str, required=True)

    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(
            args.env,
            args.seed,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.hidden_layers,
            args.save_to,
        )
    elif args.cmd == "infer":
        cmd_infer(args.env, args.load_from)
    else:
        raise NotImplementedError(f"Unknown subcommand: {args.cmd}")
