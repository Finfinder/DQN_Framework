import torch
import torch.nn as nn


class CNNDQN(nn.Module):
    def __init__(self, input_shape, action_dim, conv_layers=None, hidden_dim=512, dueling=False):
        super().__init__()

        if conv_layers is None:
            conv_layers = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]

        self.action_dim = action_dim
        self.dueling = dueling

        # Build convolutional trunk dynamically from conv_layers list
        conv_modules = []
        in_channels = input_shape[0]

        for out_channels, kernel_size, stride in conv_layers:
            conv_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
            conv_modules.append(nn.ReLU())
            in_channels = out_channels

        self.conv_trunk = nn.Sequential(*conv_modules)

        # Compute flattened size by a dummy forward pass through conv trunk
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flatten_size = self.conv_trunk(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, hidden_dim),
            nn.ReLU(),
        )

        if not dueling:
            self.q_head = nn.Linear(hidden_dim, action_dim)
        else:
            self.value_head = nn.Linear(hidden_dim, 1)
            self.advantage_head = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        relu_gain = nn.init.calculate_gain('relu')
        for m in self.conv_trunk.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=relu_gain)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=relu_gain)
                nn.init.constant_(m.bias, 0)
        if self.dueling:
            nn.init.orthogonal_(self.value_head.weight, gain=1.0)
            nn.init.constant_(self.value_head.bias, 0)
            nn.init.orthogonal_(self.advantage_head.weight, gain=1.0)
            nn.init.constant_(self.advantage_head.bias, 0)
        else:
            nn.init.orthogonal_(self.q_head.weight, gain=1.0)
            nn.init.constant_(self.q_head.bias, 0)

    def forward(self, x):
        conv_out = self.conv_trunk(x)
        fc_out = self.fc(conv_out)

        if not self.dueling:
            return self.q_head(fc_out)
        else:
            value = self.value_head(fc_out)
            advantage = self.advantage_head(fc_out)
            advantage_normalized = advantage - advantage.mean(dim=1, keepdim=True)
            q_values = value + advantage_normalized
            return q_values
