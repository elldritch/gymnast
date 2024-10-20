import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from gymnast.algorithms.vpg import PolicyGradientAgent


class ContinuousActionAgent(PolicyGradientAgent):
    def __init__(self, layer_sizes: list[int], activation: nn.Module = nn.Tanh()):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers += [
                nn.Linear(
                    layer_sizes[i],
                    # The last layer has twice the number of units, because for
                    # each continuous action we need both a mean and a standard
                    # deviation.
                    layer_sizes[i + 1] * (2 if i + 1 == len(layer_sizes) - 1 else 1),
                ),
                activation,
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    def predict(self, observation: torch.Tensor) -> torch.distributions.Distribution:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        prediction: torch.Tensor = self(observation)
        return torch.distributions.Normal(
            prediction[:, 0], torch.log(1 + torch.exp(prediction[:, 1]))
        )

    def sample(self, observation: torch.Tensor) -> NDArray[np.float32]:
        action = self.predict(observation).sample().cpu().numpy().astype(np.float32)
        return action


class DiscreteActionAgent(PolicyGradientAgent):
    def __init__(self, layer_sizes: list[int], activation: nn.Module = nn.Tanh()):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), activation]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    def predict(self, observation: torch.Tensor) -> torch.distributions.Categorical:
        prediction: torch.Tensor = self(observation)
        return torch.distributions.Categorical(logits=prediction)

    def sample(self, observation: torch.Tensor) -> int:
        return int(self.predict(observation).sample().item())
