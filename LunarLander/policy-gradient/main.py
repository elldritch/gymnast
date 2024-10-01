import argparse
from collections.abc import Callable
import os
import sys
import threading
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class InputListener(threading.Thread):
    def __init__(self, callback: Callable):
        super().__init__(daemon=True)
        self.callback = callback

    def run(self):
        while True:
            sys.stdin.readline()
            self.callback()

    def stop(self):
        raise SystemExit()


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
    actions: list[int] = []
    rewards: list[np.float64] = []

    observation: np.ndarray
    observation, _ = env.reset()
    episode_over = False
    while not episode_over:
        observations.append(observation.copy())

        action = model.predict(torch.from_numpy(observation).to("cuda")).sample().item()
        assert isinstance(action, int)
        actions.append(action)

        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(np.float64(reward))

        if verbose:
            action_name = (
                "_"
                if action == 0
                else (
                    "<"
                    if action == 1
                    else "^" if action == 2 else ">" if action == 3 else "ERROR"
                )
            )
            r = (
                "R"
                if observation[6] == 1.0
                else " " if observation[6] == 0.0 else "ERROR"
            )
            l = (
                "L"
                if observation[7] == 1.0
                else " " if observation[7] == 0.0 else "ERROR"
            )
            print(
                f"x: {observation[0]: .3f} y: {observation[1]: .3f} x': {observation[2]: .3f} y': {observation[3]: .3f} θ: {observation[4]: .3f} ω: {observation[5]: .3f} {r} {l} reward: {reward: 8.3f} action: {action_name}"
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
    returns: list[np.float64] = []
    durations: list[int] = []
    weights: list[np.float64] = []

    while len(observations) < epoch_batch_size:
        episode_observations, episode_actions, episode_rewards = explore_one_episode(
            env, model
        )
        episode_return = sum(episode_rewards)
        assert type(episode_return) == np.float64
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
    assert env.observation_space.shape is not None
    observation_dim: int = env.observation_space.shape[0]
    actions_dim = int(getattr(env.action_space, "n"))
    return (observation_dim, actions_dim)


def set_seeds(seed: int, env: gym.Env):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed)


def save_checkpoint(
    env_id: str,
    model: nn.Module,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_layers: list[int],
    save_to: str,
):
    torch.save(
        {
            "env_id": env_id,
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_layers": hidden_layers,
        },
        save_to,
    )


def cmd_train(
    env_id: str | None,
    seed: int | None,
    epochs: int,
    batch_size: int | None,
    learning_rate: float | None,
    hidden_layers: list[int] | None,
    load_from: str | None,
    save_to: str,
):
    # Defaults that can be overridden by checkpoint.
    checkpoint = None
    start_epoch: int = 0

    # Load hyperparameters from checkpoint.
    if load_from is not None:
        checkpoint = torch.load(load_from, weights_only=True)

    if env_id is None:
        if checkpoint is not None and "env_id" in checkpoint:
            env_id = checkpoint["env_id"]
            assert isinstance(env_id, str)
            print(f"Loaded env_id from checkpoint: {env_id}")
        else:
            raise ValueError("No env_id specified.")
    if seed is None:
        if checkpoint is not None and "seed" in checkpoint:
            seed = checkpoint["seed"]
            assert isinstance(seed, int)
            print(f"Loaded seed from checkpoint: {seed}")
        else:
            seed = 42
    if batch_size is None:
        if checkpoint is not None and "batch_size" in checkpoint:
            batch_size = checkpoint["batch_size"]
            assert isinstance(batch_size, int)
            print(f"Loaded batch_size from checkpoint: {batch_size}")
        else:
            batch_size = 10000
    if learning_rate is None:
        if checkpoint is not None and "learning_rate" in checkpoint:
            learning_rate = checkpoint["learning_rate"]
            assert isinstance(learning_rate, float)
            print(f"Loaded learning_rate from checkpoint: {learning_rate}")
        else:
            learning_rate = 1e-3
    if hidden_layers is None:
        if checkpoint is not None and "hidden_layers" in checkpoint:
            hidden_layers = checkpoint["hidden_layers"]
            assert isinstance(hidden_layers, list)
            print(f"Loaded hidden_layers from checkpoint: {hidden_layers}")
        else:
            hidden_layers = [64, 64]

    if checkpoint is not None and "epochs" in checkpoint:
        start_epoch = checkpoint["epochs"]
        assert isinstance(start_epoch, int)
        print(f"Loaded starting epoch number from checkpoint: {start_epoch}")
    else:
        start_epoch = 0

    print(
        f"Initializing with parameters: seed={seed}, start_epoch={start_epoch}, epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, hidden_layers={hidden_layers}"
    )

    # Initialize environment.
    env: gym.Env = gym.make(env_id, render_mode=None)
    (observation_dim, actions_dim) = dimensions(env)

    # Set seeds.
    set_seeds(seed, env)

    # Initialize model and optimizer.
    model = NeuralNet([observation_dim] + hidden_layers + [actions_dim]).to("cuda")
    optimizer = torch.optim.Adam(  # pyright: ignore [reportPrivateImportUsage]
        model.parameters(), lr=learning_rate
    )

    # Load model from checkpoint.
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        model.train()
        print(f"Loaded weights from {load_from}")

    # Save checkpoint on keypress.
    current_epoch = start_epoch

    def save_progress():
        root, ext = os.path.splitext(save_to)
        save_to_checkpoint_path = f"{root}_checkpoint_{int(time.time())}{ext}"
        print(f"Saving checkpoint to {save_to_checkpoint_path}")
        save_checkpoint(
            env_id,
            model,
            seed,
            current_epoch,
            batch_size,
            learning_rate,
            hidden_layers,
            save_to_checkpoint_path,
        )
        print(f"Checkpoint saved to {save_to_checkpoint_path}")

    save_listener = InputListener(save_progress)

    # Run training.
    print(f" Epoch | {"Loss":>12} | {"Return":>12} | {"Duration":>12}")
    save_listener.start()
    for i in range(epochs):
        current_epoch = i + start_epoch
        loss, returns, durations = train_one_epoch(env, model, optimizer, batch_size)
        print(
            f" {current_epoch:5d} | {loss:>12.3f} | {np.mean(returns):>12.3f} | {np.mean(durations):>12.3f}"
        )
        if current_epoch % 200 == 0:
            save_progress()
    print("Done, stopping")
    save_listener.stop()
    print("Stopped")
    env.close()

    # Run final model.
    env = gym.make(env_id, render_mode="human")
    explore_one_episode(env, model)
    env.close()

    # Save results.
    torch.save(model.state_dict(), save_to)


def cmd_infer(load_from: str, seed: int | None):
    # Load checkpoint.
    checkpoint = torch.load(load_from, weights_only=True)

    # Initialize environment.
    env: gym.Env = gym.make(checkpoint["env_id"], render_mode="human")

    # Set seeds.
    if seed is None:
        seed = checkpoint["seed"]
        assert isinstance(seed, int)
    set_seeds(seed, env)

    # Initialize model.
    (observation_dim, actions_dim) = dimensions(env)
    model = NeuralNet(
        [observation_dim] + checkpoint["hidden_layers"] + [actions_dim]
    ).to("cuda")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, rewards = explore_one_episode(env, model, verbose=True)
    print(f"Return: {sum(rewards)}")

    env.close()


if __name__ == "__main__":
    # Parse hyperparameters.
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    trainP = subparsers.add_parser("train")
    trainP.add_argument("--env", type=str)
    trainP.add_argument("--seed", type=int)
    trainP.add_argument("--epochs", type=int, default=600)
    trainP.add_argument("--batch_size", type=int)
    trainP.add_argument("--learning_rate", type=float)
    trainP.add_argument("--hidden_layers", type=int, nargs="+")
    trainP.add_argument("--load_from", type=str)
    trainP.add_argument("--save_to", type=str, required=True)

    inferP = subparsers.add_parser("infer")
    inferP.add_argument("--load_from", type=str, required=True)
    inferP.add_argument("--seed", type=int)

    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(
            args.env,
            args.seed,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.hidden_layers,
            args.load_from,
            args.save_to,
        )
    elif args.cmd == "infer":
        cmd_infer(args.load_from, args.seed)
    else:
        raise NotImplementedError(f"Unknown subcommand: {args.cmd}")
