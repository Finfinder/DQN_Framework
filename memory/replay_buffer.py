import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, td_error=None):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size, beta=0.4):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        pass

    def mean_priority(self):
        return 0.0

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer with importance-sampling weights."""

    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.memory = [None] * capacity
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done, td_error=None):
        self.memory[self.position] = (state, action, reward, next_state, done)

        if td_error is None:
            priority = self.max_priority
        else:
            priority = abs(float(td_error)) + self.eps

        self.priorities[self.position] = priority
        self.max_priority = max(self.max_priority, priority)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
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
        for idx, td_error in zip(indices, td_errors):
            priority = abs(float(td_error)) + self.eps
            self.priorities[int(idx)] = priority
            self.max_priority = max(self.max_priority, priority)

    def mean_priority(self):
        if self.size == 0:
            return 0.0
        return float(np.mean(self.priorities[:self.size]))

    def __len__(self):
        return self.size


class NstepReplayBuffer:
    """N-step return replay buffer with uniform sampling.

    Accumulates n consecutive transitions and stores a single transition
    with the discounted n-step return: R = sum_{k=0}^{n-1} gamma^k * r_k.
    The stored next_state is the state reached after n steps.
    """

    def __init__(self, capacity, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=capacity)
        self._buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done, td_error=None):
        self._buffer.append((state, action, reward, next_state, done))

        if done:
            # Flush all pending transitions on episode end
            while self._buffer:
                self._flush_one()
        elif len(self._buffer) == self.n_step:
            self._flush_one()

    def _flush_one(self):
        """Compute n-step return from buffered transitions and store."""
        n = len(self._buffer)
        first = self._buffer[0]
        state, action = first[0], first[1]

        # Compute discounted n-step return
        nstep_return = 0.0
        for k in range(n):
            nstep_return += (self.gamma ** k) * self._buffer[k][2]

        last = self._buffer[-1]
        next_state, done = last[3], last[4]

        self.memory.append((state, action, nstep_return, next_state, done))
        self._buffer.popleft()

    def sample(self, batch_size, beta=0.4):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        pass

    def mean_priority(self):
        return 0.0

    def __len__(self):
        return len(self.memory)


def create_buffer(config):
    """Factory: create a replay buffer based on config.buffer_type."""
    bt = config.buffer_type
    if bt == "replay":
        return ReplayBuffer(config.memory_size)
    elif bt == "prioritized":
        return PrioritizedReplayBuffer(
            config.memory_size,
            alpha=config.per_alpha,
            eps=config.per_eps,
        )
    elif bt == "nstep":
        return NstepReplayBuffer(
            config.memory_size,
            n_step=config.nstep_n,
            gamma=config.gamma,
        )
    else:
        raise ValueError(
            f"Unknown buffer_type '{bt}'. "
            f"Available: 'replay', 'prioritized', 'nstep'"
        )