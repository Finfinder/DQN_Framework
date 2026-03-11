import random
import torch
import torch.optim as optim

class DQNAgent:

    def __init__(self, policy_net, target_net, memory, config):

        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.config = config

        self.optimizer = optim.Adam(policy_net.parameters(), lr=config.lr)

    def select_action(self, state, epsilon, env):
        if random.random() < epsilon:
            return env.action_space.sample()

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.config.device)
            q_values = self.policy_net(state)

        return q_values.argmax().item()

    def train_step(self, beta=0.4):
        if len(self.memory) < self.config.batch_size:
            return None

        if self.config.use_per:
            states, actions, rewards, next_states, dones, indices, is_weights = self.memory.sample(
                self.config.batch_size,
                beta=beta,
            )
            is_weights = torch.FloatTensor(is_weights).to(self.config.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
            indices = None
            is_weights = None

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

        td_errors = expected_q - q_value
        sample_losses = td_errors.pow(2)

        if self.config.use_per:
            loss = (is_weights * sample_losses).mean()
        else:
            loss = sample_losses.mean()

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),1.0)

        self.optimizer.step()

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.config.tau*param.data + (1-self.config.tau)*target_param.data
            )

        stats = {
            "loss": float(loss.item()),
            "q_mean": float(q_value.mean().item()),
            "target_q_mean": float(expected_q.mean().item()),
            "q_max_mean": float(q_values.max(dim=1).values.mean().item()),
            "td_error_mean": float(td_errors.abs().mean().item()),
        }

        if self.config.use_per:
            stats["indices"] = indices
            stats["td_errors"] = td_errors.detach().abs().cpu().numpy()
            stats["is_weight_mean"] = float(is_weights.mean().item())

        return stats