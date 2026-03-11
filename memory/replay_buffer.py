import random
import numpy as np
from collections import deque

class ReplayBuffer:

    def __init__(self, capacity, use_per=False, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.use_per = use_per
        self.alpha = alpha
        self.eps = eps

        if self.use_per:
            self.memory = [None] * capacity
            self.priorities = np.zeros(capacity, dtype=np.float32)
            self.position = 0
            self.size = 0
            self.max_priority = 1.0
        else:
            self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, td_error=None):
        transition = (state, action, reward, next_state, done)

        if not self.use_per:
            self.memory.append(transition)
            return

        self.memory[self.position] = transition

        if td_error is None:
            priority = self.max_priority
        else:
            priority = abs(float(td_error)) + self.eps

        self.priorities[self.position] = priority
        self.max_priority = max(self.max_priority, priority)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):

        if not self.use_per:
            batch = random.sample(self.memory, batch_size)

            states, actions, rewards, next_states, dones = zip(*batch)

            return (
                np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32)
            )

        valid_priorities = self.priorities[:self.size]
        scaled_priorities = np.power(valid_priorities + self.eps, self.alpha)
        prob_sum = scaled_priorities.sum()

        if prob_sum <= 0:
            probs = np.ones_like(scaled_priorities) / len(scaled_priorities)
        else:
            probs = scaled_priorities / prob_sum

        indices = np.random.choice(self.size, batch_size, p=probs)
        batch = [self.memory[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*batch)

        is_weights = np.power(self.size * probs[indices], -beta)
        is_weights /= is_weights.max()

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(indices),
            np.array(is_weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        if not self.use_per:
            return

        for idx, td_error in zip(indices, td_errors):
            priority = abs(float(td_error)) + self.eps
            self.priorities[int(idx)] = priority
            self.max_priority = max(self.max_priority, priority)

    def mean_priority(self):
        if not self.use_per or self.size == 0:
            return 0.0
        return float(np.mean(self.priorities[:self.size]))

    def __len__(self):
        if self.use_per:
            return self.size
        return len(self.memory)