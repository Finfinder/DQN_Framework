"""Shared fixtures for DQN_Framework unit tests."""
import os
import sys
import warnings

import pytest
import torch

from config.config import Config


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _is_ci():
    return os.environ.get("CI", "").lower() == "true"


def _is_venv():
    return sys.prefix != sys.base_prefix


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_cuda: mark test as requiring CUDA GPU",
    )


def pytest_sessionstart(session):
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    python_ver = sys.version.split()[0]
    print(f"\n{'=' * 60}")
    print("DQN Framework Test Session")
    print(f"Python:         {python_ver}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA device:    {cuda_device}")
    print(f"venv active:    {_is_venv()}")
    print(f"CI:             {_is_ci()}")
    print(f"{'=' * 60}\n")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_no_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "requires_cuda" in item.keywords:
            item.add_marker(skip_no_cuda)


# ---------------------------------------------------------------------------
# Autouse environment validation fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="session")
def validate_environment():
    if _is_ci():
        return
    if not _is_venv():
        warnings.warn(
            "Running outside .venv! CUDA PyTorch may not be available. "
            r"Activate .venv first: .\.venv\Scripts\Activate.ps1",
            stacklevel=1,
        )
    if not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available. Tests will run on CPU (slower). "
            "Ensure .venv is active with CUDA PyTorch installed.",
            stacklevel=1,
        )


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


@pytest.fixture
def cnn_config():
    """Config for CNN tests — minimal parameters for fast CPU execution."""
    cfg = Config("ALE/Pong-v5")
    cfg.cnn_hidden_dim = 64
    cfg.conv_layers = [(8, 4, 2), (16, 3, 1)]
    cfg.frame_size = [32, 32]
    return cfg

