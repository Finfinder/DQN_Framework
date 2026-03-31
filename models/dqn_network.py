import torch
import torch.nn as nn

from models.cnn_dqn_network import CNNDQN


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers=[64, 64], dueling=False):
        super().__init__()
        
        self.action_dim = action_dim
        self.dueling = dueling
        
        # Build shared trunk layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.trunk = nn.Sequential(*layers)
        
        if not dueling:
            # Standard DQN: single output layer for Q-values
            self.q_head = nn.Linear(input_dim, action_dim)
        else:
            # Dueling DQN: separate value and advantage streams
            self.value_head = nn.Linear(input_dim, 1)
            self.advantage_head = nn.Linear(input_dim, action_dim)
    
    def forward(self, x):
        trunk_out = self.trunk(x)
        
        if not self.dueling:
            # Standard DQN: return Q-values directly
            return self.q_head(trunk_out)
        else:
            # Dueling DQN: compute Q(s,a) = V(s) + (A(s,a) - mean(A(s,.)))
            value = self.value_head(trunk_out)
            advantage = self.advantage_head(trunk_out)
            
            # Normalize advantages by subtracting mean
            advantage_normalized = advantage - advantage.mean(dim=1, keepdim=True)
            
            # Combine value and normalized advantages
            q_values = value + advantage_normalized
            return q_values


def create_network(config, state_shape, action_dim):
    if config.network_type == "cnn":
        return CNNDQN(
            state_shape, action_dim,
            conv_layers=config.conv_layers,
            hidden_dim=config.cnn_hidden_dim,
            dueling=config.use_dueling,
        )
    return DQN(
        state_shape[0], action_dim,
        hidden_layers=config.hidden_layers,
        dueling=config.use_dueling,
    )