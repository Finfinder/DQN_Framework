import gymnasium as gym
from pathlib import Path
import torch

from models.dqn_network import DQN
from config.config import Config

config = Config()

env = gym.make(config.env_name,render_mode="human")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = DQN(state_size, action_size, hidden_layers=config.hidden_layers).to(config.device)

model_path = Path(config.model_path)
if not model_path.exists():
    raise FileNotFoundError(
        f"Model file not found: {model_path}. Run train.py first to generate weights."
    )

model.load_state_dict(
    torch.load(model_path, weights_only=True, map_location=config.device)
)

model.eval()

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