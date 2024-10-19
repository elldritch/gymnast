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

    def act(self, observation: torch.Tensor) -> torch.distributions.Distribution:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        action: torch.Tensor = self(observation)
        return torch.distributions.Normal(
            action[:, 0], torch.log(1 + torch.exp(action[:, 1]))
        )


class DiscreteActionAgent(PolicyGradientAgent):
    def __init__(self, layer_sizes: list[int], activation: nn.Module = nn.Tanh()):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(layer_sizes) - 1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), activation]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    def act(self, observation: torch.Tensor) -> torch.distributions.Categorical:
        action: torch.Tensor = self(observation)
        return torch.distributions.Categorical(logits=action)
