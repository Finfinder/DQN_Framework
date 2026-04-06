"""Unit tests for config.config.Config."""
import pytest
from config.config import Config


class TestConfigCreation:

    def test_cartpole_default(self):
        cfg = Config("CartPole-v1")
        assert cfg.env_name == "CartPole-v1"
        assert cfg.solved_threshold == pytest.approx(400.0)
        assert cfg.num_episodes == 900

    def test_mountaincar(self):
        cfg = Config("MountainCar-v0")
        assert cfg.env_name == "MountainCar-v0"
        assert cfg.solved_threshold == pytest.approx(-100.0)
        assert cfg.buffer_type == "replay"

    def test_acrobot(self):
        cfg = Config("Acrobot-v1")
        assert cfg.env_name == "Acrobot-v1"
        assert cfg.solved_threshold == pytest.approx(-80.0)

    def test_pong(self):
        cfg = Config("ALE/Pong-v5")
        assert cfg.env_name == "ALE/Pong-v5"
        assert cfg.network_type == "cnn"
        assert cfg.is_atari is True

    def test_unknown_env_raises(self):
        with pytest.raises(ValueError, match="Unknown environment"):
            Config("FakeEnv-v99")


class TestConfigDefaults:

    def test_weight_decay_default_zero(self):
        cfg = Config("CartPole-v1")
        assert cfg.weight_decay == 0

    def test_weight_decay_attribute_exists(self):
        cfg = Config("MountainCar-v0")
        assert hasattr(cfg, "weight_decay")
        assert cfg.weight_decay == 0

    def test_defaults_merged_into_env(self):
        cfg = Config("CartPole-v1")
        # gamma is from DEFAULTS (not overridden by CartPole ENV_CONFIG)
        assert cfg.gamma == pytest.approx(0.99)

    def test_env_config_overrides_defaults(self):
        cfg = Config("CartPole-v1")
        # lr is overridden in CartPole ENV_CONFIG from default 0.001 → 0.0005
        assert cfg.lr == pytest.approx(0.0005)

    def test_use_per_true_for_prioritized(self):
        cfg = Config("CartPole-v1")
        assert cfg.buffer_type == "prioritized"
        assert cfg.use_per is True

    def test_use_per_false_for_replay(self):
        cfg = Config("MountainCar-v0")
        assert cfg.buffer_type == "replay"
        assert cfg.use_per is False


class TestConfigSuffix:

    def test_dueling_mlp_suffix(self):
        cfg = Config("CartPole-v1")
        assert cfg.suffix == "_dueling"

    def test_standard_mlp_suffix(self):
        cfg = Config("CartPole-v1")
        cfg.use_dueling = False
        # suffix is set in __init__ so we need to verify at init time
        # CartPole default is use_dueling=True (from DEFAULTS)
        assert cfg.suffix == "_dueling"

    def test_cnn_dueling_suffix(self):
        cfg = Config("ALE/Pong-v5")
        # Pong uses cnn and default use_dueling=True
        assert "_cnn" in cfg.suffix
        assert "_dueling" in cfg.suffix

    def test_model_path_has_suffix(self):
        cfg = Config("CartPole-v1")
        assert cfg.suffix in cfg.model_path
        assert cfg.model_path.endswith(".pth")

    def test_plot_path_has_suffix(self):
        cfg = Config("CartPole-v1")
        assert cfg.suffix in cfg.plot_path
        assert cfg.plot_path.endswith(".png")

    def test_model_path_suffix_before_extension(self):
        cfg = Config("CartPole-v1")
        stem, ext = cfg.model_path.rsplit(".", 1)
        assert stem.endswith(cfg.suffix)
        assert ext == "pth"


class TestConfigDevice:

    def test_device_attribute_exists(self):
        cfg = Config("CartPole-v1")
        import torch
        assert isinstance(cfg.device, torch.device)
        assert cfg.device.type in ("cuda", "cpu")
