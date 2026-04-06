# DQN Framework (PyTorch + Gymnasium)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Environments-0081A5)](https://gymnasium.farama.org/)
[![Version](https://img.shields.io/badge/version-1.0.1-green)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/Finfinder/DQN_Framework/ci.yml?label=CI&logo=github)](https://github.com/Finfinder/DQN_Framework/actions)
[![Quality Gate](https://sonarcloud.io/api/project_badges/measure?project=Finfinder_DQN_Framework&metric=alert_status)](https://sonarcloud.io/summary/overall?id=Finfinder_DQN_Framework)

A lightweight framework for training **Deep Q-Network (DQN)** agents in discrete action space environments using Gymnasium.

Currently supported configurations:

- `CartPole-v1`
- `MountainCar-v0`
- `Acrobot-v1`
- `ALE/Pong-v5` (CNN DQN)

---

## What's Included

- `train.py`: DQN training + best model saving + progress plot.
- `evaluate.py`: trained model evaluation (greedy policy, epsilon=0) with statistics summary.
- `play.py`: running the trained model in `render_mode="human"` mode.
- `agents/dqn_agent.py`: agent logic (epsilon-greedy, training step, soft update target network).
- `models/dqn_network.py`: MLP built dynamically from a hidden layers list + factory `create_network(config, state_shape, action_dim)`.
- `models/cnn_dqn_network.py`: CNN DQN with configurable Conv2d layers and Dueling support.
- `memory/replay_buffer.py`: three replay buffer variants — `ReplayBuffer` (uniform), `PrioritizedReplayBuffer` (PER), `NstepReplayBuffer` (N-step returns) — with factory `create_buffer(config)`.
- `utils/evaluate.py`: shared `evaluate_policy()` function used by `evaluate.py` and `train.py`.
- `utils/wrappers.py`: `make_env()` with `frame_skip`, `wrap_env()` with image preprocessing (Atari + generic).
- `config/config.py`: centralized hyperparameter configuration and per-environment presets.

---

## Requirements

- Python 3.10+
- Python packages:
  - `torch`
  - `gymnasium`
  - `gymnasium[atari]`, `ale-py` (optional, required for Atari environments)
  - `opencv-python` (required by `AtariPreprocessing`)
  - `numpy`
  - `matplotlib`
  - `tensorboard`

Example installation:

```bash
pip install torch gymnasium numpy matplotlib tensorboard

# Optional — for Atari environments (e.g. ALE/Pong-v5):
pip install "gymnasium[atari]" ale-py opencv-python
```

---

## Quick Start

1. Training:

```bash
python train.py
```

2. Training for a specific environment:

```bash
python train.py CartPole-v1
python train.py MountainCar-v0
python train.py Acrobot-v1
python train.py ALE/Pong-v5
```

3. Training with a specific seed (overrides the config value):

```bash
python train.py MountainCar-v0 --seed 123
```

4. Watching the trained agent:

```bash
python play.py
python play.py CartPole-v1
python play.py MountainCar-v0 --play-episodes 10
```

5. Evaluating the trained model (greedy policy, epsilon=0):

```bash
python evaluate.py CartPole-v1
python evaluate.py MountainCar-v0 --episodes 50
python evaluate.py Acrobot-v1 --episodes 100 --render
python evaluate.py CartPole-v1 --render --render-episodes 5
```

6. Running unit tests:

```bash
pytest tests/ -v
```

7. Running tests with coverage report:

```bash
pytest tests/ --cov=config --cov=agents --cov=memory --cov=utils --cov=models --cov-report=term-missing
```

---

## How It Works

- The agent selects actions using **epsilon-greedy**.
- Transitions are stored in a **Replay Buffer** (uniform, PER, or N-step — configurable via `buffer_type`).
- Network updates use the **Double DQN** variant:
  - action selection `argmax` through `policy_net`,
  - evaluation of that action through `target_net`.
- `target_net` is updated via **soft update** with parameter `tau` (or **hard update** every `target_update_freq` steps for CNN).
- For `CartPole-v1`, terminal transitions (`terminated`) receive a training penalty of `-10.0`.
- Loss function: Smooth L1 (Huber loss).
- For image-based environments (e.g. `ALE/Pong-v5`), a **CNN DQN** network with frame preprocessing (grayscale, resize, frame stacking) is used.

---

## Configuration

Configuration is located in `config/config.py` in the `Config` class.

Key fields:

- `gamma`, `lr`, `batch_size`, `memory_size`
- `epsilon`, `epsilon_decay`, `epsilon_min`
- `tau`
- `hidden_layers`
- `num_episodes`, `min_replay_size`, `train_every_steps`
- `solved_threshold`
- `model_path`, `plot_path`, `play_episodes`

Replay Buffer:

- `buffer_type`: buffer type — `"replay"` (uniform), `"prioritized"` (PER), `"nstep"` (N-step returns). Default: `"prioritized"`. Automatically sets `use_per`.
- `nstep_n`: number of N-step return steps (only when `buffer_type: "nstep"`). Default: `3`.

| `buffer_type` | Class | Description |
|---------------|-------|-------------|
| `"replay"` | `ReplayBuffer` | Uniform sampling from deque. Simple and fast. |
| `"prioritized"` | `PrioritizedReplayBuffer` | PER with IS weights. Better for sparse rewards. |
| `"nstep"` | `NstepReplayBuffer` | N-step returns + uniform. Accelerates value propagation. |

PER parameters (active when `buffer_type: "prioritized"`):

- `per_alpha`: prioritization strength (0.0 = uniform sampling).
- `per_beta_start`: initial beta value for IS weights.
- `per_beta_frames`: number of steps to anneal beta to 1.0.
- `per_eps`: small constant added to priority for numerical stability.

Architecture parameters:

- `use_dueling`: enables/disables Dueling DQN (default: `False`). When `True`, the network separates state value and action advantage estimation for better generalization. All artifacts (model, logs, metrics, plots) are stored separately for standard DQN and Dueling DQN using a suffix (`_standard` or `_dueling`).
- `network_type`: network type — `"mlp"` (default) or `"cnn"` (for image-based environments).
- `conv_layers`: list of Conv2d layers as tuples `(out_channels, kernel_size, stride)`. Default: `[(32, 8, 4), (64, 4, 2), (64, 3, 1)]`.
- `cnn_hidden_dim`: hidden layer size after the CNN trunk.
- `frame_stack`: number of frames stacked as observation (default: `4`).
- `frame_size`: target frame size `[H, W]` (default: `[84, 84]`).
- `frame_skip`: number of frames skipped (action repeat, default: `1`).
- `is_atari`: flag controlling the selection of Atari vs generic wrappers.
- `target_update_freq`: hard update target network every N steps (0 = soft update with `tau`).

Evaluation parameters:

- `eval_every`: how often (in training episodes) greedy policy evaluation is run (default: `100`).
- `eval_episodes`: number of evaluation episodes (default: `10`).

By default, `python train.py` runs the preset for `CartPole-v1`.

To run training for another supported environment, pass it as an argument:

```bash
python train.py MountainCar-v0
python train.py Acrobot-v1
```

Note: only environments defined in `Config.ENV_CONFIG` are supported.

---

## Results and Artifacts

During training, the following are saved:

- model weights file (`*.pth`) to `config.model_path`,
- training curve plot (`*.png`) to `config.plot_path`,
- TensorBoard logs to the `logs/<env_name><suffix>_<YYYYMMDD-HHMMSS>/` directory,
- episode metrics to CSV: `metrics/<env_name>_<model_name>_<YYYYMMDD-HHMMSS>.csv`.

Example suffix: `_standard` for standard DQN, `_dueling` for Dueling DQN. The suffix is automatically appended to model and plot names, and consequently also visible in metric file names.

To view metrics during/after training:

```bash
tensorboard --logdir logs
```

Logged metrics include:

- `episode/reward`
- `episode/avg100`
- `episode/epsilon`
- `episode/loss`
- `episode/q_mean`
- `train/loss`
- `train/q_mean`
- `train/q_max_mean`
- `train/target_q_mean`
- `train/td_error_mean`
- `train/beta` (when `buffer_type: "prioritized"`)
- `train/is_weight_mean` (when `buffer_type: "prioritized"`)
- `train/priority_mean` (when `buffer_type: "prioritized"`)
- `eval/mean_reward` (greedy policy)
- `eval/std_reward` (greedy policy)
- `eval/min_reward` (greedy policy)
- `eval/max_reward` (greedy policy)

CSV columns:

- `episode`
- `reward`
- `avg100`
- `epsilon`
- `beta`
- `is_weight_mean`
- `td_error_mean`
- `priority_mean`

A separate evaluation CSV file (`*_eval.csv`) contains columns:

- `episode`
- `mean_reward`
- `std_reward`
- `min_reward`
- `max_reward`

Example artifacts visible in the repository:

- `dqn_cartpole.pth`
- `training_curve_cartpole.png`

---

## Notes

- `play.py` will raise `FileNotFoundError` if the model does not exist.
- For reproducible results, set `seed` in `Config.DEFAULTS` or use the `--seed` flag when calling `train.py` (e.g. `python train.py MountainCar-v0 --seed 42`). The `--seed` flag overrides the config value.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

---

## License

This project is licensed under the [MIT License](LICENSE).
