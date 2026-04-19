# Changelog

All notable changes to the DQN Framework project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `tests/test_cnn_dqn_network.py`: 25 new unit tests across 4 test classes (`TestCNNDQNCreation`, `TestCNNDQNForwardStandard`, `TestCNNDQNForwardDueling`, `TestCNNDQNFactory`) covering forward pass of `CNNDQN` — instantiation, output shape/dtype, NaN/Inf checks, advantage normalization, production 84×84 config, and `create_network()` factory integration. Tests are CPU-only, compatible with CI. Coverage of `models/cnn_dqn_network.py` raised from 11.32% to 100%.
- `tests/conftest.py`: `cnn_config` fixture with minimal CNN parameters (`cnn_hidden_dim=64`, `conv_layers=[(8,4,2),(16,3,1)]`, `frame_size=[32,32]`) for fast CPU test execution.

### Added
- `tests/test_analyze.py`: 47 new unit tests across 14 new test classes covering all major functions in `utils/analyze.py` — `_diagnose_trend`, `_diagnose_epsilon`, `_diagnose_td_error`, `_diagnose_eval_vs_train`, `diagnose`, `list_runs`, `load_run`, `load_latest`, `compare_runs`, `run_summary`, `build_summary_report`, `export_summary_report`, `parse_args`, `_print_env_list`, `_print_train_eval_results`, `main`. Tests use `unittest.mock.patch`, in-memory `pd.DataFrame`, `tmp_path` fixture and `capsys` — no real CSV files read from disk. Coverage of `utils/analyze.py` raised from 18.75% to 97%.

### Changed
- `memory/replay_buffer.py`: introduced `BaseReplayBuffer(ABC)` abstract base class with `@abstractmethod push()` and default implementations of `sample()`, `update_priorities()`, `mean_priority()`, and `__len__()`. All three concrete buffer classes (`ReplayBuffer`, `PrioritizedReplayBuffer`, `NstepReplayBuffer`) now inherit from `BaseReplayBuffer`, eliminating ~16 lines of duplicated code between `ReplayBuffer` and `NstepReplayBuffer`. No breaking changes to public API.

### Added
- `tests/test_replay_buffer.py`: `TestBaseReplayBuffer` class — 5 new tests verifying ABC enforcement (`TypeError` on direct instantiation), concrete subclass instantiation, and `isinstance` checks for all three buffer types. Extended `TestCreateBufferFactory` with `test_factory_returns_base_instance` asserting `isinstance(buf, BaseReplayBuffer)`.
- `tests/conftest.py`: session-scoped autouse fixture `validate_environment` that emits `warnings.warn` when `.venv` is not active or CUDA is unavailable (silent in CI).
- `tests/conftest.py`: `pytest_sessionstart` banner displaying Python version, PyTorch version, CUDA availability, CUDA device name, venv status, and CI detection at the start of every test session.
- `tests/conftest.py`: marker `@pytest.mark.requires_cuda` with automatic skip via `pytest_collection_modifyitems` when CUDA is not available; registered programmatically and in `pyproject.toml`.
- `pyproject.toml`: `markers` entry for `requires_cuda` in `[tool.pytest.ini_options]`.

### Removed
- `confirm_test.py`: removed the one-off multi-seed confirmation script — its functionality is fully covered by `tuning_test.py` (which uses shared helpers from `utils/training.py`). The file had 0% test coverage and was not used in CI. Reference to `confirm_test.py` also removed from `sonar.coverage.exclusions` in `sonar-project.properties`.

## [2.0.0] - 2026-04-07

### Added
- Unit tests (`tests/`) — 101 tests covering Config, ReplayBuffer (all variants), DQNAgent, utils/training, utils/evaluate, utils/wrappers; integration with pytest + pytest-cov.
- Module `utils/training.py` — shared training logic: `run_episode()`, `compute_beta()`, `shape_reward()`, `compute_avg100()` extracted from `train.py` and `tuning_test.py`.
- Parameter `weight_decay` in `Config.DEFAULTS` (default `0`) passed to the Adam optimizer in `DQNAgent`.
- Coverage configuration in CI (`ci.yml`) and SonarCloud (`sonar.yml`) — generating `coverage.xml` before scan.
- SonarCloud integration — workflow `.github/workflows/sonar.yml` with analysis on push/PR, pytest code coverage and `sonar-project.properties` configuration.
- Shared SonarQube for IDE binding — `.sonarlint/connectedMode.json` for Connected Mode with SonarCloud.
- `LICENSE` file (MIT, 2025, Finfinder).
- Pre-trained model `dqn_pong_cnn_dueling.pth` for `ALE/Pong-v5`.

### Fixed
- SonarCloud `sonar-project.properties` corrected: removed `sonar.tests=tests` and `sonar.test.inclusions` (tests directory absent in CI workspace at scan time); configured `sonar.coverage.exclusions` to exclude `utils/analyze.py` and `models/cnn_dqn_network.py` (not testable in CI — standalone pandas tool and CNN requiring GPU/Atari ROM).
- 11× S1244 (BUG): float comparisons with `==` replaced with `pytest.approx()` in `test_config.py` and `test_training.py`.
- S6709 (CODE SMELL): `PrioritizedReplayBuffer` — added `seed=None` parameter to `np.random.default_rng()`.
- 5× S1481 (CODE SMELL): unused variables in `test_analyze.py` replaced with `_prefix`.
- Reduced Cognitive Complexity: `list_runs()`, `diagnose()`, `main()` in `utils/analyze.py` (by extracting 7 private helpers); `run_seed()` in `tuning_test.py` (using `run_episode()` from `utils/training.py`).
- Removed unused variables in `utils/analyze.py`: `meta_train`, `meta_eval` → `_`; `final_eps` removed.
- Empty `update_priorities()` methods in `ReplayBuffer` and `NstepReplayBuffer` annotated with `# No-op` comment.
- Unused interface parameters in `ReplayBuffer` and `NstepReplayBuffer`: `td_error` → `_td_error`, `beta` → `_beta`.
- Migrated `PrioritizedReplayBuffer.sample()` from `np.random.choice()` to `numpy.random.Generator` (`self.rng.choice()`).
- Code duplication between `train.py` and `tuning_test.py` — training loop and reward shaping extracted to `utils/training.py`.
- Updated `sonarqube-scan-action` from v5 to v6 (v5 contains a security vulnerability and is deprecated).

### Removed
- File `.github/copilot-instructions.md` — replaced by scoped instructions in `.github/instructions/dqn-framework.instructions.md`.

### Changed
- Section "Virtual Environment" in `.github/instructions/dqn-framework.instructions.md` extended with mandatory venv activation enforcement (CRITICAL for CUDA/GPU).
- README in English — full translation of `README.md` to English with badge bar (Python, PyTorch, Gymnasium, Version, License, CI) in SeqMcpServer style.
- Updated file `.github/instructions/dqn-framework.instructions.md` with scoped project conventions (applyTo: `**`).
- CNN DQN architecture (`models/cnn_dqn_network.py`) with configurable Conv2d layers and Dueling support.
- Factory `create_network(config, state_shape, action_dim)` for automatic MLP or CNN selection.
- Environment wrappers (`utils/wrappers.py`): `make_env()` with `frame_skip`, `wrap_env()` with image preprocessing (Atari + generic).
- `ALE/Pong-v5` preset with dedicated CNN hyperparameters.
- New configuration parameters: `network_type`, `conv_layers`, `cnn_hidden_dim`, `frame_stack`, `frame_size`, `frame_skip`, `is_atari`, `target_update_freq`, `adam_eps`.
- Configurable gradient clipping (`gradient_clip`) — default 1.0, allows per-environment tuning.
- Loss function changed from MSE to Smooth L1 (Huber loss) in `DQNAgent.train_step()`.
- Optional hard target update every `target_update_freq` steps (when > 0) instead of continuous soft update.
- Parameter `adam_eps` added to Adam optimizer.
- Files `train.py`, `evaluate.py`, `play.py` updated to support CNN and environment wrappers.
- CSV metrics flushed after each write for faster live preview.
- CI smoke test updated for the new `make_env`/`wrap_env` API.
- Tuned CartPole-v1 hyperparameters: `hidden_layers=[128,128]`, `epsilon_decay=0.993`, `lr=0.0005`, `tau=0.003`, `batch_size=128`, `memory_size=30000`, `train_every_steps=2`, `per_beta_frames=30000`, `adam_eps=1e-4`, `gradient_clip=0.3`. Achieved 83% success rate (10/12 seeds) at 800 episodes.
- Increased CartPole-v1 `num_episodes` from 800 to 900 — seed 42 (default) required 827 episodes to solve, causing deterministic failure at the 800-episode limit.
- Tuned MountainCar-v0 hyperparameters: `lr=0.001`, `tau=0.001`, `epsilon_decay=0.998`, `buffer_type="replay"`, `eval_every=50`, `eval_episodes=20`. Switch from PER to uniform replay buffer eliminates TD error instability. Achieved 67% success rate (8/12 seeds) at 3500 episodes.
- Reward shaping for MountainCar-v0: `reward + 10 * abs(velocity)` — encourages the agent to build momentum.
- Eval-based early stopping in `train.py` — saves model at best evaluation score and terminates training when eval mean > solved_threshold.
- Fixed `plt.show()` blocking — changed to `plt.show(block=False)` + `plt.close()` to prevent process hanging after training.
- `tuning_test.py` — environment parameterization via `sys.argv[1]`, eval-based early stopping, per-environment reward shaping, extended to 12 seeds.

## [1.0.0] - 2026-03-14

### Added
- DQN training with Double DQN update rule (`train.py`).
- Dueling DQN architecture (`use_dueling` in configuration).
- Three replay buffer variants: uniform (`ReplayBuffer`), Prioritized Experience Replay (`PrioritizedReplayBuffer`), N-step returns (`NstepReplayBuffer`).
- Factory `create_buffer(config)` for automatic buffer creation based on configuration.
- Soft update target network with `tau` parameter.
- Centralized hyperparameter configuration in `config/config.py` with per-environment presets (`CartPole-v1`, `MountainCar-v0`, `Acrobot-v1`).
- Training penalty `-10.0` for terminal transitions in `CartPole-v1`.
- Metrics logging to TensorBoard (`logs/`) and CSV (`metrics/`).
- Separate CSV files for training and evaluation metrics.
- Greedy policy evaluation every `eval_every` episodes during training.
- Standalone model evaluation (`evaluate.py`) with rendering option.
- Visualization of trained agent in `render_mode="human"` mode (`play.py`).
- Automatic `_dueling` / `_standard` suffixes for artifacts.
- `--seed` flag in `train.py` to override the configuration seed.
- Early stopping after exceeding `solved_threshold`.
- Training progress plot with smoothing (moving average).
- Versioning mechanism (`version.py`) and CHANGELOG file.
