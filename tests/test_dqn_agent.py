"""Unit tests for agents.dqn_agent.DQNAgent."""
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock
from agents.dqn_agent import DQNAgent
from memory.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, create_buffer
from models.dqn_network import create_network
from tests.helpers import fill_buffer


def _make_agent(config):
    """Create a DQNAgent with real small networks."""
    state_shape = (4,)
    action_size = 2
    policy_net = create_network(config, state_shape, action_size).to(config.device)
    target_net = create_network(config, state_shape, action_size).to(config.device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    memory = create_buffer(config)
    return DQNAgent(policy_net, target_net, memory, config), memory, policy_net


class TestSelectAction:

    def test_epsilon_one_returns_random(self, small_config):
        agent, _, _ = _make_agent(small_config)
        env = MagicMock()
        env.action_space.sample.return_value = 1
        state = np.zeros(4, dtype=np.float32)
        action = agent.select_action(state, epsilon=1.0, env=env)
        assert action == 1
        env.action_space.sample.assert_called_once()

    def test_epsilon_zero_returns_greedy(self, small_config):
        agent, _, _ = _make_agent(small_config)
        env = MagicMock()
        state = np.zeros(4, dtype=np.float32)
        action = agent.select_action(state, epsilon=0.0, env=env)
        assert action in (0, 1)
        env.action_space.sample.assert_not_called()

    def test_action_within_valid_range(self, small_config):
        agent, _, _ = _make_agent(small_config)
        env = MagicMock()
        env.action_space.sample.return_value = 0
        state = np.zeros(4, dtype=np.float32)
        for eps in [0.0, 0.5, 1.0]:
            action = agent.select_action(state, eps, env)
            assert action in (0, 1)


class TestTrainStep:

    def test_returns_none_when_buffer_too_small(self, small_config):
        agent, _, _ = _make_agent(small_config)
        # Buffer is empty — should return None
        result = agent.train_step()
        assert result is None

    def test_returns_stats_dict_when_ready(self, small_config):
        agent, memory, _ = _make_agent(small_config)
        fill_buffer(memory, small_config.batch_size * 2)
        result = agent.train_step()
        assert result is not None
        assert "loss" in result
        assert "q_mean" in result
        assert "target_q_mean" in result
        assert "q_max_mean" in result
        assert "td_error_mean" in result

    def test_loss_is_positive(self, small_config):
        agent, memory, _ = _make_agent(small_config)
        fill_buffer(memory, small_config.batch_size * 2)
        result = agent.train_step()
        assert result["loss"] >= 0.0

    def test_per_stats_included_when_use_per(self, per_config):
        per_config.use_per = True
        per_config.buffer_type = "prioritized"
        agent, memory, _ = _make_agent(per_config)
        fill_buffer(memory, per_config.batch_size * 2)
        result = agent.train_step(beta=0.4)
        assert result is not None
        assert "indices" in result
        assert "td_errors" in result
        assert "is_weight_mean" in result

    def test_no_per_stats_when_use_per_false(self, small_config):
        small_config.use_per = False
        small_config.buffer_type = "replay"
        agent, memory, _ = _make_agent(small_config)
        fill_buffer(memory, small_config.batch_size * 2)
        result = agent.train_step()
        assert result is not None
        assert "indices" not in result
        assert "td_errors" not in result

    def test_train_step_increments_train_steps(self, small_config):
        agent, memory, _ = _make_agent(small_config)
        fill_buffer(memory, small_config.batch_size * 2)
        assert agent.train_steps == 0
        agent.train_step()
        assert agent.train_steps == 1


class TestWeightDecay:

    def test_weight_decay_zero_by_default(self, small_config):
        agent, _, _ = _make_agent(small_config)
        for group in agent.optimizer.param_groups:
            assert group["weight_decay"] == 0

    def test_weight_decay_passed_from_config(self, small_config):
        small_config.weight_decay = 1e-4
        agent, _, _ = _make_agent(small_config)
        for group in agent.optimizer.param_groups:
            assert group["weight_decay"] == pytest.approx(1e-4)
