import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[64, 64]):
        super().__init__()

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)