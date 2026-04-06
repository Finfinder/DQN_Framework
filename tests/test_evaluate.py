"""Unit tests for utils.evaluate.evaluate_policy."""
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch
from utils.evaluate import evaluate_policy
from models.dqn_network import create_network


def _make_policy_net(config):
    state_shape = (4,)
    action_size = 2
    return create_network(config, state_shape, action_size).to(config.device)


def _make_mock_env(episode_length=5, num_episodes=3):
    """Mock env that runs each episode for episode_length steps."""
    env = MagicMock()
    state = np.zeros(4, dtype=np.float32)
    env.reset.return_value = (state, {})

    def step_side_effect(action):
        nonlocal call_count
        call_count += 1
        terminated = (call_count % episode_length) == 0
        return state, 1.0, terminated, False, {}

    call_count = 0
    env.step.side_effect = step_side_effect
    return env


class TestEvaluatePolicy:

    def test_returns_dict_with_required_keys(self, small_config):
        policy_net = _make_policy_net(small_config)
        mock_env = _make_mock_env()

        with patch("utils.evaluate.make_env", return_value=mock_env), \
             patch("utils.evaluate.wrap_env", return_value=(mock_env, (4,))):
            result = evaluate_policy(policy_net, small_config, num_episodes=3, device=small_config.device)

        assert "mean_reward" in result
        assert "std_reward" in result
        assert "min_reward" in result
        assert "max_reward" in result

    def test_mean_reward_is_float(self, small_config):
        policy_net = _make_policy_net(small_config)
        mock_env = _make_mock_env()

        with patch("utils.evaluate.make_env", return_value=mock_env), \
             patch("utils.evaluate.wrap_env", return_value=(mock_env, (4,))):
            result = evaluate_policy(policy_net, small_config, num_episodes=3, device=small_config.device)

        assert isinstance(result["mean_reward"], float)

    def test_runs_correct_number_of_episodes(self, small_config):
        policy_net = _make_policy_net(small_config)
        mock_env = _make_mock_env(episode_length=3, num_episodes=5)

        with patch("utils.evaluate.make_env", return_value=mock_env), \
             patch("utils.evaluate.wrap_env", return_value=(mock_env, (4,))):
            evaluate_policy(policy_net, small_config, num_episodes=5, device=small_config.device)

        assert mock_env.reset.call_count == 5

    def test_seed_passed_on_first_episode(self, small_config):
        policy_net = _make_policy_net(small_config)
        mock_env = _make_mock_env()

        with patch("utils.evaluate.make_env", return_value=mock_env), \
             patch("utils.evaluate.wrap_env", return_value=(mock_env, (4,))):
            evaluate_policy(policy_net, small_config, num_episodes=2, device=small_config.device, seed=42)

        first_call_kwargs = mock_env.reset.call_args_list[0]
        assert first_call_kwargs == ((), {"seed": 42})

    def test_env_closed_after_evaluation(self, small_config):
        policy_net = _make_policy_net(small_config)
        mock_env = _make_mock_env()

        with patch("utils.evaluate.make_env", return_value=mock_env), \
             patch("utils.evaluate.wrap_env", return_value=(mock_env, (4,))):
            evaluate_policy(policy_net, small_config, num_episodes=2, device=small_config.device)

        mock_env.close.assert_called_once()

    def test_min_max_reward_consistent(self, small_config):
        policy_net = _make_policy_net(small_config)
        mock_env = _make_mock_env()

        with patch("utils.evaluate.make_env", return_value=mock_env), \
             patch("utils.evaluate.wrap_env", return_value=(mock_env, (4,))):
            result = evaluate_policy(policy_net, small_config, num_episodes=3, device=small_config.device)

        assert result["min_reward"] <= result["mean_reward"]
        assert result["mean_reward"] <= result["max_reward"]
