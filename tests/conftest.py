"""Shared fixtures for DQN_Framework unit tests."""
import pytest
from config.config import Config


@pytest.fixture
def config():
    """Default Config for CartPole-v1."""
    return Config("CartPole-v1")


@pytest.fixture
def small_config():
    """Config with minimal parameters for fast tests."""
    cfg = Config("CartPole-v1")
    cfg.memory_size = 200
    cfg.batch_size = 8
    cfg.min_replay_size = 8
    cfg.num_episodes = 5
    cfg.hidden_layers = [16, 16]
    cfg.use_dueling = False
    cfg.use_per = False
    cfg.buffer_type = "replay"
    cfg.train_every_steps = 1
    cfg.tau = 0.01
    return cfg


@pytest.fixture
def per_config():
    """Config with PER enabled for fast tests."""
    cfg = Config("CartPole-v1")
    cfg.memory_size = 200
    cfg.batch_size = 8
    cfg.min_replay_size = 8
    cfg.use_per = True
    cfg.buffer_type = "prioritized"
    cfg.per_alpha = 0.6
    cfg.per_beta_start = 0.4
    cfg.per_beta_frames = 1000
    cfg.hidden_layers = [16, 16]
    cfg.use_dueling = False
    cfg.train_every_steps = 1
    cfg.tau = 0.01
    return cfg

