import os
import time
from typing import Any, Callable, SupportsFloat

import numpy as np
import torch
import dill
from dill import dump, load

from gymnast.algorithms.vpg import PolicyGradientAgent

CHECKPOINT_FORMAT = 1


def save_checkpoint[
    Observation: np.ndarray, Action: int | float, Reward: SupportsFloat
](
    env_id: str,
    agent: PolicyGradientAgent,
    agent_args: list[Any],
    optimizer: torch.optim.Optimizer,
    optimizer_args: list[Any],
    gradient_fn: Callable[
        [PolicyGradientAgent, list[list[tuple[Observation, Action, Reward]]]],
        torch.Tensor,
    ],
    seed: int,
    batch_size: int,
    current_epoch: int,
    checkpoint_folder: str,
    save_id: str,
):
    checkpoint_id = f"{save_id}_{current_epoch}_{int(time.time())}"
    print(f"Saving checkpoint ID: {checkpoint_id}")

    # Use Torch to save model and optimizer state, because those have optimized
    # picking implementations.
    torch.save(
        {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(checkpoint_folder, f"state_dict_{checkpoint_id}.pt"),
    )
    # Use dill to save other information.
    with open(os.path.join(checkpoint_folder, f"info_{checkpoint_id}.pkl"), "wb") as f:
        dump(
            {
                "checkpoint_format": CHECKPOINT_FORMAT,
                "env": env_id,
                "seed": seed,
                "elapsed_epochs": current_epoch + 1,
                "batch_size": batch_size,
                "agent_class_source": dill.source.getsource(type(agent)),
                "agent_args": agent_args,
                "optimizer_class_source": dill.source.getsource(type(optimizer)),
                "optimizer_args": optimizer_args,
                "gradient_fn": gradient_fn,
                "gradient_fn_source": dill.source.getsource(gradient_fn),
            },
            f,
        )
    print(f"Done saving checkpoint ID: {checkpoint_id}")


def load_checkpoint[
    Observation: np.ndarray, Action: int | float, Reward: SupportsFloat
](
    checkpoint_folder: str,
    checkpoint_id: str,
    current_agent_class: type[PolicyGradientAgent] | None = None,
    current_optimizer_class: type[torch.optim.Optimizer] | None = None,
    current_gradient_fn: (
        Callable[[list[list[tuple[Observation, Action, Reward]]]], torch.Tensor] | None
    ) = None,
) -> tuple[
    str,
    PolicyGradientAgent,
    torch.optim.Optimizer,
    Callable[[list[list[tuple[Observation, Action, Reward]]]], torch.Tensor],
    int,
    int,
    int,
]:
    # Load information from dill pickle.
    with open(os.path.join(checkpoint_folder, f"info_{checkpoint_id}.pkl"), "rb") as f:
        pkl = load(f)

    # Check format.
    assert "checkpoint_format" in pkl
    assert pkl["checkpoint_format"] == CHECKPOINT_FORMAT

    # Load basic parameters.
    assert "env" in pkl
    env = pkl["env"]
    assert isinstance(env, str)
    assert "seed" in pkl
    seed = pkl["seed"]
    assert isinstance(seed, int)
    assert "elapsed_epochs" in pkl
    elapsed_epochs = pkl["elapsed_epochs"]
    assert isinstance(elapsed_epochs, int)
    assert "batch_size" in pkl
    batch_size = pkl["batch_size"]
    assert isinstance(batch_size, int)

    # Deserialize stateful objects.
    assert "agent_class" in pkl
    agent_class = pkl["agent_class"]
    assert "agent_args" in pkl
    agent_args = pkl["agent_args"]
    assert isinstance(agent_args, list)
    if current_agent_class is not None:
        current_agent_class_source = dill.source.getsource(current_agent_class)
        assert "agent_class_source" in pkl
        agent_class_source = pkl["agent_class_source"]
        assert isinstance(agent_class_source, str)
        if current_agent_class_source != agent_class_source:
            print("WARNING: saved agent class does not match current agent class")
            print("CURRENT:")
            print(current_agent_class_source)
            print("SAVED:")
            print(agent_class_source)
    agent = agent_class(*agent_args)

    assert "optimizer_class" in pkl
    optimizer_class = pkl["optimizer_class"]
    assert "optimizer_args" in pkl
    optimizer_args = pkl["optimizer_args"]
    assert isinstance(optimizer_args, list)
    if current_optimizer_class is not None:
        current_optimizer_class_source = dill.source.getsource(current_optimizer_class)
        assert "optimizer_class_source" in pkl
        optimizer_class_source = pkl["optimizer_class_source"]
        assert isinstance(optimizer_class_source, str)
        if current_optimizer_class_source != optimizer_class_source:
            print(
                "WARNING: saved optimizer class does not match current optimizer class"
            )
            print("CURRENT:")
            print(current_optimizer_class_source)
            print("SAVED:")
            print(optimizer_class_source)
    optimizer = optimizer_class(agent.parameters(), *optimizer_args)

    assert "gradient_fn" in pkl
    gradient_fn: Callable[
        [list[list[tuple[Observation, Action, Reward]]]], torch.Tensor
    ] = pkl["gradient_fn"]
    assert callable(gradient_fn)
    if current_gradient_fn is not None:
        current_gradient_fn_source = dill.source.getsource(current_gradient_fn)
        assert "gradient_fn_source" in pkl
        gradient_fn_source = pkl["gradient_fn_source"]
        assert isinstance(gradient_fn_source, str)
        if current_gradient_fn_source != gradient_fn_source:
            print(
                "WARNING: saved gradient function does not match current gradient function"
            )
            print("CURRENT:")
            print(current_gradient_fn_source)
            print("SAVED:")
            print(gradient_fn_source)

    # Load model and optimizer state.
    pt = torch.load(os.path.join(checkpoint_folder, f"state_dict_{checkpoint_id}.pt"))
    assert "checkpoint_format" in pt
    assert pt["checkpoint_format"] == CHECKPOINT_FORMAT
    assert "model_state_dict" in pt
    agent.load_state_dict(pt["model_state_dict"])
    assert "optimizer_state_dict" in pt
    optimizer.load_state_dict(pt["optimizer_state_dict"])

    return (env, agent, optimizer, gradient_fn, seed, elapsed_epochs, batch_size)
