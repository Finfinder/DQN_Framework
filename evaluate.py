import argparse
import csv
from datetime import datetime
from pathlib import Path
import gymnasium as gym
import torch

from models.dqn_network import DQN
from config.config import Config
from utils.evaluate import evaluate_policy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent (greedy policy, epsilon=0)."
    )
    parser.add_argument(
        "env_name",
        nargs="?",
        default="CartPole-v1",
        choices=sorted(Config.ENV_CONFIG.keys()),
        help="Environment name",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render a few episodes visually after evaluation",
    )
    parser.add_argument(
        "--render-episodes",
        type=int,
        default=3,
        help="Number of episodes to render when --render is set (default: 3)",
    )
    return parser.parse_args()


def render_episodes(model, config, num_episodes):
    env = gym.make(config.env_name, render_mode="human")
    model.eval()

    try:
        for ep in range(1, num_episodes + 1):
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

            print(f"  Render episode {ep}: reward = {total_reward:.1f}")
    finally:
        env.close()


def main():
    args = parse_args()
    config = Config(env_name=args.env_name)

    env = gym.make(config.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()

    model = DQN(
        state_size, action_size,
        hidden_layers=config.hidden_layers,
        dueling=config.use_dueling,
    ).to(config.device)

    model_path = Path(config.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model for '{config.env_name}' not found: {model_path}. "
            f"Run: python train.py {config.env_name}"
        )

    model.load_state_dict(
        torch.load(model_path, weights_only=True, map_location=config.device)
    )

    print(f"Evaluating {config.env_name} | model: {model_path} | episodes: {args.episodes}")
    print("-" * 60)

    eval_stats = evaluate_policy(
        model, config.env_name, args.episodes, config.device, seed=config.seed,
    )

    print(f"  Mean reward: {eval_stats['mean_reward']:.2f}")
    print(f"  Std reward:  {eval_stats['std_reward']:.2f}")
    print(f"  Min reward:  {eval_stats['min_reward']:.2f}")
    print(f"  Max reward:  {eval_stats['max_reward']:.2f}")
    print("-" * 60)

    # Save evaluation results to CSV
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    safe_env = config.env_name.replace("/", "_")
    safe_model = model_path.stem.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_csv_path = metrics_dir / f"{safe_env}_{safe_model}_standalone_eval_{timestamp}.csv"

    with eval_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mean_reward", "std_reward", "min_reward", "max_reward", "episodes"])
        writer.writerow([
            f"{eval_stats['mean_reward']:.4f}",
            f"{eval_stats['std_reward']:.4f}",
            f"{eval_stats['min_reward']:.4f}",
            f"{eval_stats['max_reward']:.4f}",
            args.episodes,
        ])

    print(f"Saved eval CSV: {eval_csv_path}")

    if args.render:
        print(f"\nRendering {args.render_episodes} episode(s)...\n")
        render_episodes(model, config, args.render_episodes)


if __name__ == "__main__":
    main()
