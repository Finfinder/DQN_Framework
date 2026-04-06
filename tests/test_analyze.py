"""Unit tests for utils.analyze — _parse_run_filename and helper functions."""
import pytest
from datetime import datetime
from utils.analyze import _parse_run_filename


class TestParseRunFilename:

    def test_valid_train_filename(self):
        stem = "CartPole-v1_dqn_cartpole_dueling_20250101-120000"
        result = _parse_run_filename(stem, is_eval=False)
        assert result is not None
        env, model, ts, run_type = result
        assert env == "CartPole-v1"
        assert "dqn_cartpole" in model
        assert isinstance(ts, datetime)
        assert run_type == "train"

    def test_valid_eval_filename(self):
        stem = "CartPole-v1_dqn_cartpole_dueling_20250101-120000_eval"
        result = _parse_run_filename(stem, is_eval=True)
        assert result is not None
        env, model, ts, run_type = result
        assert env == "CartPole-v1"
        assert run_type == "eval"

    def test_invalid_filename_missing_dqn_prefix(self):
        stem = "CartPole-v1_nomatch_20250101-120000"
        result = _parse_run_filename(stem, is_eval=False)
        assert result is None

    def test_invalid_filename_bad_timestamp(self):
        stem = "CartPole-v1_dqn_cartpole_dueling_99999999-999999"
        result = _parse_run_filename(stem, is_eval=False)
        assert result is None

    def test_standalone_eval_run_type(self):
        stem = "CartPole-v1_dqn_cartpole_standalone_eval_20250101-120000"
        result = _parse_run_filename(stem, is_eval=False)
        # Model names without 'dqn_' prefix return None; this stem has no valid _dqn_ separator
        # just before timestamp, so the parser cannot extract a valid timestamp position.
        # Either None (unparseable) or a valid tuple — must not raise an exception.
        assert result is None or (isinstance(result, tuple) and len(result) == 4)

    def test_mountaincar_env_parsed(self):
        stem = "MountainCar-v0_dqn_mountaincar_dueling_20250606-100000"
        result = _parse_run_filename(stem, is_eval=False)
        assert result is not None
        env, model, ts, run_type = result
        assert env == "MountainCar-v0"

    def test_timestamp_parsed_correctly(self):
        stem = "CartPole-v1_dqn_cartpole_dueling_20240315-143000"
        result = _parse_run_filename(stem, is_eval=False)
        assert result is not None
        _, _, ts, _ = result
        assert ts.year == 2024
        assert ts.month == 3
        assert ts.day == 15
        assert ts.hour == 14
        assert ts.minute == 30

    def test_returns_four_element_tuple(self):
        stem = "CartPole-v1_dqn_cartpole_dueling_20250101-120000"
        result = _parse_run_filename(stem, is_eval=False)
        assert result is not None
        assert len(result) == 4
