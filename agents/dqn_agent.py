import random
import torch
import torch.optim as optim
import torch.nn as nn

class DQNAgent:

    def __init__(self, policy_net, target_net, memory, config):

        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.config = config

        self.optimizer = optim.Adam(policy_net.parameters(), lr=config.lr)
        self.criterion = nn.MSELoss()

    def select_action(self, state, epsilon, env):
        if random.random() < epsilon:
            return env.action_space.sample()

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.config.device)
            q_values = self.policy_net(state)

        return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < self.config.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)

        states = torch.FloatTensor(states).to(self.config.device)
        next_states = torch.FloatTensor(next_states).to(self.config.device)
        actions = torch.LongTensor(actions).to(self.config.device)
        rewards = torch.FloatTensor(rewards).to(self.config.device)
        dones = torch.FloatTensor(dones).to(self.config.device)

        q_values = self.policy_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_actions = self.policy_net(next_states).argmax(1)

        with torch.no_grad():

            next_q_values = self.target_net(next_states)
            next_q = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            expected_q = rewards + self.config.gamma * next_q * (1 - dones)

        loss = self.criterion(q_value, expected_q)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),1.0)

        self.optimizer.step()

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.config.tau*param.data + (1-self.config.tau)*target_param.data
            )