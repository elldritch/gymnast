from dataclasses import dataclass
import json
import os
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.optim.optimizer import StateDict
import dill

from gymnast.algorithms.vpg import PolicyGradientAgent

CHECKPOINT_FORMAT = 2
CHECKPOINT_FORMAT_KEY = "checkpoint_format"

WEIGHTS_FILE_NAME = "weights.pt"
AGENT_WEIGHTS_KEY = "model_state_dict"
OPTIMIZER_WEIGHTS_KEY = "optimizer_state_dict"

FUNCTIONS_FILE_NAME = "functions.pkl"
WEIGHTS_FUNCTION_KEY = "weights_fn"

META_FILE_NAME = "meta.json"
ENV_ID_KEY = "env"
SEED_KEY = "seed"
ELAPSED_EPOCHS_KEY = "elapsed_epochs"
BATCH_SIZE_KEY = "batch_size"
AGENT_ARGS_KEY = "agent_args"
OPTIMIZER_ARGS_KEY = "optimizer_args"
AGENT_CLASS_SOURCE_KEY = "agent_class_source"
OPTIMIZER_CLASS_SOURCE_KEY = "optimizer_class_source"
WEIGHTS_FUNCTION_SOURCE_KEY = "weights_fn_source"


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
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_id)

    print(f"Saving checkpoint ID: {checkpoint_id}")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Use Torch to save model and optimizer state, because those have optimized
    # picking implementations.
    torch.save(
        {
            CHECKPOINT_FORMAT_KEY: CHECKPOINT_FORMAT,
            AGENT_WEIGHTS_KEY: agent.state_dict(),
            OPTIMIZER_WEIGHTS_KEY: optimizer.state_dict(),
        },
        os.path.join(checkpoint_path, WEIGHTS_FILE_NAME),
    )
    # Use dill to serialize classes and functions.
    with open(os.path.join(checkpoint_path, FUNCTIONS_FILE_NAME), "wb") as f:
        # TODO: Can we serialize the agent and optimizer classes too? When I
        # tried this earlier, got some sort of serialization error. Can we
        # refactor those classes to be serializable?
        dill.dump(
            {
                CHECKPOINT_FORMAT_KEY: CHECKPOINT_FORMAT,
                WEIGHTS_FUNCTION_KEY: weights_fn,
            },
            f,
        )
    # Use JSON to serialize information that's useful to be human-readable.
    with open(os.path.join(checkpoint_path, META_FILE_NAME), "w") as f:
        json.dump(
            {
                CHECKPOINT_FORMAT_KEY: CHECKPOINT_FORMAT,
                ENV_ID_KEY: env_id,
                SEED_KEY: seed,
                ELAPSED_EPOCHS_KEY: current_epoch + 1,
                BATCH_SIZE_KEY: batch_size,
                AGENT_ARGS_KEY: agent_args,
                OPTIMIZER_ARGS_KEY: optimizer_args,
                AGENT_CLASS_SOURCE_KEY: dill.source.getsource(type(agent)),
                OPTIMIZER_CLASS_SOURCE_KEY: dill.source.getsource(type(optimizer)),
                WEIGHTS_FUNCTION_SOURCE_KEY: dill.source.getsource(weights_fn),
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
    checkpoint_path = os.path.join(checkpoint_folder, checkpoint_id)

    # Load metadata from JSON.
    with open(os.path.join(checkpoint_path, META_FILE_NAME), "r") as f:
        meta = json.load(f)
    assert CHECKPOINT_FORMAT_KEY in meta
    assert meta[CHECKPOINT_FORMAT_KEY] == CHECKPOINT_FORMAT

    assert ENV_ID_KEY in meta
    env = meta[ENV_ID_KEY]
    assert isinstance(env, str)
    assert SEED_KEY in meta
    seed = meta[SEED_KEY]
    assert isinstance(seed, int)
    assert ELAPSED_EPOCHS_KEY in meta
    elapsed_epochs = meta[ELAPSED_EPOCHS_KEY]
    assert isinstance(elapsed_epochs, int)
    assert BATCH_SIZE_KEY in meta
    batch_size = meta[BATCH_SIZE_KEY]
    assert isinstance(batch_size, int)
    assert AGENT_ARGS_KEY in meta
    agent_args = meta[AGENT_ARGS_KEY]
    assert isinstance(agent_args, list)
    assert AGENT_CLASS_SOURCE_KEY in meta
    agent_class_source = meta[AGENT_CLASS_SOURCE_KEY]
    assert isinstance(agent_class_source, str)
    assert OPTIMIZER_ARGS_KEY in meta
    optimizer_args = meta[OPTIMIZER_ARGS_KEY]
    assert isinstance(optimizer_args, list)
    assert OPTIMIZER_CLASS_SOURCE_KEY in meta
    optimizer_class_source = meta[OPTIMIZER_CLASS_SOURCE_KEY]
    assert isinstance(optimizer_class_source, str)
    assert WEIGHTS_FUNCTION_SOURCE_KEY in meta
    weights_fn_source = meta[WEIGHTS_FUNCTION_SOURCE_KEY]
    assert isinstance(weights_fn_source, str)

    # Load functions from pickle.
    with open(os.path.join(checkpoint_path, FUNCTIONS_FILE_NAME), "rb") as f:
        pkl = dill.load(f)
    assert CHECKPOINT_FORMAT_KEY in pkl
    assert pkl[CHECKPOINT_FORMAT_KEY] == CHECKPOINT_FORMAT

    assert WEIGHTS_FUNCTION_KEY in pkl
    weights_fn: Callable[[list[np.float32]], list[float]] = pkl[WEIGHTS_FUNCTION_KEY]
    assert callable(weights_fn)

    # Load weights.
    pt = torch.load(
        os.path.join(checkpoint_path, WEIGHTS_FILE_NAME),
        weights_only=True,
    )
    assert CHECKPOINT_FORMAT_KEY in pt
    assert pt[CHECKPOINT_FORMAT_KEY] == CHECKPOINT_FORMAT

    assert AGENT_WEIGHTS_KEY in pt
    agent_state_dict = pt[AGENT_WEIGHTS_KEY]
    assert OPTIMIZER_WEIGHTS_KEY in pt
    optimizer_state_dict = pt[OPTIMIZER_WEIGHTS_KEY]

    # Check sources.
    if current_agent_class is not None:
        current_agent_class_source = dill.source.getsource(current_agent_class)
        if current_agent_class_source != agent_class_source:
            print("WARNING: saved agent class does not match current agent class")
            print("CURRENT:")
            print(current_agent_class_source)
            print("SAVED:")
            print(agent_class_source)
    if current_optimizer_class is not None:
        current_optimizer_class_source = dill.source.getsource(current_optimizer_class)
        if current_optimizer_class_source != optimizer_class_source:
            print(
                "WARNING: saved optimizer class does not match current optimizer class"
            )
            print("CURRENT:")
            print(current_optimizer_class_source)
            print("SAVED:")
            print(optimizer_class_source)
    if current_weights_fn is not None:
        current_weights_fn_source = dill.source.getsource(current_weights_fn)
        if current_weights_fn_source != weights_fn_source:
            print(
                "WARNING: saved gradient function does not match current gradient function"
            )
            print("CURRENT:")
            print(current_weights_fn_source)
            print("SAVED:")
            print(weights_fn_source)

    return Checkpoint(
        env,
        agent_args,
        optimizer_args,
        agent_state_dict,
        optimizer_state_dict,
        weights_fn,
        seed,
        elapsed_epochs,
        batch_size,
    )
