"""Unit tests for utils.training — compute_beta, shape_reward, run_episode, compute_avg100."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from utils.training import compute_beta, shape_reward, run_episode, compute_avg100


class TestComputeBeta:

    def test_returns_one_when_not_using_per(self, small_config):
        small_config.use_per = False
        assert compute_beta(small_config, 0) == pytest.approx(1.0)
        assert compute_beta(small_config, 100000) == pytest.approx(1.0)

    def test_returns_per_beta_start_at_step_zero(self, per_config):
        per_config.per_beta_start = 0.4
        per_config.per_beta_frames = 1000
        result = compute_beta(per_config, 0)
        assert result == pytest.approx(0.4, rel=1e-4)

    def test_returns_one_at_per_beta_frames(self, per_config):
        per_config.per_beta_start = 0.4
        per_config.per_beta_frames = 1000
        result = compute_beta(per_config, 1000)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_capped_at_one_beyond_per_beta_frames(self, per_config):
        per_config.per_beta_start = 0.4
        per_config.per_beta_frames = 1000
        result = compute_beta(per_config, 9999)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_anneals_linearly(self, per_config):
        per_config.per_beta_start = 0.4
        per_config.per_beta_frames = 1000
        beta_mid = compute_beta(per_config, 500)
        assert 0.4 < beta_mid < 1.0
        assert beta_mid == pytest.approx(0.7, rel=1e-4)


class TestShapeReward:

    def test_cartpole_terminal_penalized(self):
        result = shape_reward("CartPole-v1", 1.0, np.zeros(4), terminated=True)
        assert result == pytest.approx(-10.0)

    def test_cartpole_non_terminal_unchanged(self):
        result = shape_reward("CartPole-v1", 1.0, np.zeros(4), terminated=False)
        assert result == pytest.approx(1.0)

    def test_mountaincar_adds_velocity_bonus(self):
        next_state = np.array([0.0, 0.05], dtype=np.float32)
        result = shape_reward("MountainCar-v0", -1.0, next_state, terminated=False)
        expected = -1.0 + 10 * abs(0.05)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_other_env_passthrough(self):
        result = shape_reward("Acrobot-v1", -1.0, np.zeros(6), terminated=False)
        assert result == pytest.approx(-1.0)

    def test_other_env_terminal_passthrough(self):
        result = shape_reward("Acrobot-v1", 0.0, np.zeros(6), terminated=True)
        assert result == pytest.approx(0.0)


class TestComputeAvg100:

    def test_mean_of_all_when_fewer_than_100(self):
        rewards = [1.0, 2.0, 3.0]
        assert compute_avg100(rewards) == pytest.approx(2.0, rel=1e-4)

    def test_last_100_when_more_than_100(self):
        rewards = [0.0] * 900 + [1.0] * 100
        result = compute_avg100(rewards)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_single_element(self):
        assert compute_avg100([5.0]) == pytest.approx(5.0, rel=1e-4)


class TestRunEpisode:

    def _make_mock_env(self, episode_length=3):
        """Make a mock env that runs for episode_length steps."""
        env = MagicMock()
        state = np.zeros(4, dtype=np.float32)
        # reset returns (state, info)
        env.reset.return_value = (state, {})
        # step returns (next_state, reward, terminated, truncated, info)
        # Last step terminates
        side_effects = []
        for i in range(episode_length):
            terminated = i == episode_length - 1
            side_effects.append((state, 1.0, terminated, False, {}))
        env.step.side_effect = side_effects
        return env

    def _make_mock_agent(self):
        agent = MagicMock()
        agent.select_action.return_value = 0
        agent.train_step.return_value = None
        return agent

    def test_returns_dict_with_required_keys(self, small_config):
        env = self._make_mock_env(3)
        agent = self._make_mock_agent()
        memory = MagicMock()
        memory.__len__ = MagicMock(return_value=0)

        result = run_episode(env, agent, memory, small_config, epsilon=0.5, step_count=0)

        assert "total_reward" in result
        assert "step_count" in result
        assert "train_stats_list" in result

    def test_step_count_incremented(self, small_config):
        env = self._make_mock_env(5)
        agent = self._make_mock_agent()
        memory = MagicMock()
        memory.__len__ = MagicMock(return_value=0)

        result = run_episode(env, agent, memory, small_config, epsilon=0.5, step_count=10)
        assert result["step_count"] == 15  # 10 + 5 steps

    def test_total_reward_accumulated(self, small_config):
        env = self._make_mock_env(4)  # 4 steps, each reward=1.0
        agent = self._make_mock_agent()
        memory = MagicMock()
        memory.__len__ = MagicMock(return_value=0)

        result = run_episode(env, agent, memory, small_config, epsilon=0.5, step_count=0)
        assert result["total_reward"] == pytest.approx(4.0, rel=1e-4)

    def test_train_stats_list_empty_when_no_training(self, small_config):
        env = self._make_mock_env(3)
        agent = self._make_mock_agent()
        memory = MagicMock()
        # Buffer always too small to train
        memory.__len__ = MagicMock(return_value=0)

        result = run_episode(env, agent, memory, small_config, epsilon=0.5, step_count=0)
        assert result["train_stats_list"] == []

    def test_train_stats_list_populated_when_training_occurs(self, small_config):
        env = self._make_mock_env(4)
        memory = MagicMock()
        # Buffer large enough to train on every step
        memory.__len__ = MagicMock(return_value=1000)

        fake_stats = {
            "loss": 0.1, "q_mean": 0.5, "q_max_mean": 1.0,
            "target_q_mean": 0.4, "td_error_mean": 0.2,
        }
        agent = MagicMock()
        agent.select_action.return_value = 0
        agent.train_step.return_value = fake_stats

        small_config.min_replay_size = 0
        small_config.train_every_steps = 1
        small_config.use_per = False

        result = run_episode(env, agent, memory, small_config, epsilon=0.5, step_count=0)
        assert len(result["train_stats_list"]) > 0
        ts = result["train_stats_list"][0]
        assert "beta" in ts
        assert "step" in ts

    def test_per_priorities_updated_when_use_per(self, per_config):
        env = self._make_mock_env(4)
        memory = MagicMock()
        memory.__len__ = MagicMock(return_value=1000)

        import numpy as np
        fake_stats = {
            "loss": 0.1, "q_mean": 0.5, "q_max_mean": 1.0,
            "target_q_mean": 0.4, "td_error_mean": 0.2,
            "indices": np.array([0, 1]),
            "td_errors": np.array([0.3, 0.5]),
            "is_weight_mean": 0.8,
        }
        agent = MagicMock()
        agent.select_action.return_value = 0
        agent.train_step.return_value = fake_stats

        per_config.min_replay_size = 0
        per_config.train_every_steps = 1

        run_episode(env, agent, memory, per_config, epsilon=0.5, step_count=0)
        memory.update_priorities.assert_called()

    def test_initial_state_skips_env_reset(self, small_config):
        env = self._make_mock_env(3)
        agent = self._make_mock_agent()
        memory = MagicMock()
        memory.__len__ = MagicMock(return_value=0)
        initial_state = np.zeros(4, dtype=np.float32)

        run_episode(env, agent, memory, small_config, epsilon=0.5, step_count=0, initial_state=initial_state)

        env.reset.assert_not_called()
