import argparse
import gymnasium as gym
from pathlib import Path
import torch

from models.dqn_network import DQN
from config.config import Config
from version import __version__


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a trained DQN agent for a supported Gymnasium environment."
    )
    parser.add_argument(
        "env_name",
        nargs="?",
        default="CartPole-v1",
        choices=sorted(Config.ENV_CONFIG.keys()),
        help="Environment name",
    )
    parser.add_argument(
        "--play-episodes",
        type=int,
        default=None,
        help="Number of episodes to play (overrides config default)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser.parse_args()


args = parse_args()
config = Config(env_name=args.env_name)
if args.play_episodes is not None:
    config.play_episodes = args.play_episodes

env = gym.make(config.env_name, render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = DQN(state_size, action_size, hidden_layers=config.hidden_layers, dueling=config.use_dueling).to(config.device)

model_path = Path(config.model_path)
if not model_path.exists():
    raise FileNotFoundError(
        f"Model for '{config.env_name}' not found: {model_path}. "
        f"Run: python train.py {config.env_name}"
    )

model.load_state_dict(
    torch.load(model_path, weights_only=True, map_location=config.device)
)

model.eval()

print(f"DQN Framework v{__version__}")
print(f"Playing in {config.env_name} for {config.play_episodes} episode(s)")

for episode in range(config.play_episodes):

    state, _ = env.reset()

    done = False
    total_reward = 0.0

    while not done:

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(config.device)

        with torch.no_grad():
            q_values = model(state_tensor)

        action = q_values.argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        total_reward += float(reward)

    print(f"Episode {episode + 1}, Reward: {total_reward:.1f}")

env.close()