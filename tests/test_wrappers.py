"""Unit tests for utils.wrappers — make_env and wrap_env."""
import numpy as np
import pytest
from utils.wrappers import make_env, wrap_env


class TestMakeEnv:

    def test_creates_cartpole_env(self):
        env = make_env("CartPole-v1")
        assert env is not None
        env.close()

    def test_creates_mountaincar_env(self):
        env = make_env("MountainCar-v0")
        assert env is not None
        env.close()

    def test_env_has_action_space(self):
        env = make_env("CartPole-v1")
        assert hasattr(env, "action_space")
        env.close()

    def test_env_has_observation_space(self):
        env = make_env("CartPole-v1")
        assert hasattr(env, "observation_space")
        env.close()

    def test_reset_returns_state_and_info(self):
        env = make_env("CartPole-v1")
        state, info = env.reset()
        assert state is not None
        assert isinstance(info, dict)
        env.close()

    def test_step_returns_five_values(self):
        env = make_env("CartPole-v1")
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        env.close()


class TestWrapEnvMlp:

    def test_mlp_returns_env_and_shape(self, small_config):
        env = make_env("CartPole-v1")
        wrapped_env, state_shape = wrap_env(env, small_config)
        assert state_shape == (4,)
        wrapped_env.close()

    def test_mlp_state_shape_matches_observation_space(self, small_config):
        env = make_env("CartPole-v1")
        wrapped_env, state_shape = wrap_env(env, small_config)
        assert state_shape == wrapped_env.observation_space.shape
        wrapped_env.close()

    def test_mlp_step_returns_flat_state(self, small_config):
        env = make_env("CartPole-v1")
        wrapped_env, state_shape = wrap_env(env, small_config)
        state, _ = wrapped_env.reset()
        assert state.shape == state_shape
        wrapped_env.close()

    def test_mountaincar_mlp_shape(self):
        from config.config import Config
        cfg = Config("MountainCar-v0")
        env = make_env("MountainCar-v0")
        wrapped_env, state_shape = wrap_env(env, cfg)
        assert state_shape == (2,)
        wrapped_env.close()
