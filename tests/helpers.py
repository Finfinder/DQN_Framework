"""Shared helper functions for DQN_Framework unit tests."""
import numpy as np


def make_transitions(n, state_dim=4, action_dim=2):
    """Generate n random (state, action, reward, next_state, done) transitions."""
    rng = np.random.default_rng(0)
    transitions = []
    for i in range(n):
        state = rng.standard_normal(state_dim).astype(np.float32)
        action = int(rng.integers(0, action_dim))
        reward = float(rng.standard_normal())
        next_state = rng.standard_normal(state_dim).astype(np.float32)
        done = bool(i == n - 1)
        transitions.append((state, action, reward, next_state, done))
    return transitions


def fill_buffer(buf, n=20):
    """Push n transitions into buf."""
    for t in make_transitions(n):
        buf.push(*t)
