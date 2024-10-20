from dataclasses import dataclass
import os
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.optim.optimizer import StateDict
import dill
from dill import dump, load

from gymnast.algorithms.vpg import PolicyGradientAgent

CHECKPOINT_FORMAT = 1


def save_checkpoint(
    env_id: str,
    agent: PolicyGradientAgent,
    agent_args: list[Any],
    optimizer: torch.optim.Optimizer,
    optimizer_args: list[Any],
    weights_fn: Callable[[list[np.float32]], list[float]],
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
                "weights_fn": weights_fn,
                "weights_fn_source": dill.source.getsource(weights_fn),
            },
            f,
        )
    print(f"Done saving checkpoint ID: {checkpoint_id}")


@dataclass
class Checkpoint:
    env: str
    agent_args: list[Any]
    optimizer_args: list[Any]
    agent_state_dict: Mapping[str, Any]
    optimizer_state_dict: StateDict
    weights_fn: Callable[[list[np.float32]], list[float]]
    seed: int
    elapsed_epochs: int
    batch_size: int


def load_checkpoint(
    checkpoint_folder: str,
    checkpoint_id: str,
    current_agent_class: type[PolicyGradientAgent] | None = None,
    current_optimizer_class: type[torch.optim.Optimizer] | None = None,
    current_weights_fn: Callable[[list[np.float32]], list[float]] | None = None,
) -> Checkpoint:
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

    # Deserialize and check agent and optimizer.
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

    assert "weights_fn" in pkl
    weights_fn: Callable[[list[np.float32]], list[float]] = pkl["weights_fn"]
    assert callable(weights_fn)
    if current_weights_fn is not None:
        current_weights_fn_source = dill.source.getsource(current_weights_fn)
        assert "weights_fn_source" in pkl
        weights_fn_source = pkl["weights_fn_source"]
        assert isinstance(weights_fn_source, str)
        if current_weights_fn_source != weights_fn_source:
            print(
                "WARNING: saved gradient function does not match current gradient function"
            )
            print("CURRENT:")
            print(current_weights_fn_source)
            print("SAVED:")
            print(weights_fn_source)

    # Load model and optimizer state.
    pt = torch.load(
        os.path.join(checkpoint_folder, f"state_dict_{checkpoint_id}.pt"),
        weights_only=True,
    )
    assert "checkpoint_format" in pt
    assert pt["checkpoint_format"] == CHECKPOINT_FORMAT
    assert "model_state_dict" in pt
    assert "optimizer_state_dict" in pt

    return Checkpoint(
        env,
        agent_args,
        optimizer_args,
        pt["model_state_dict"],
        pt["optimizer_state_dict"],
        weights_fn,
        seed,
        elapsed_epochs,
        batch_size,
    )
