import argparse
import csv
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from config.config import Config
from models.dqn_network import DQN
from memory.replay_buffer import ReplayBuffer
from agents.dqn_agent import DQNAgent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DQN for a supported Gymnasium environment."
    )
    parser.add_argument(
        "env_name",
        nargs="?",
        default="CartPole-v1",
        help="Environment name (e.g. CartPole-v1, MountainCar-v0, Acrobot-v1)",
    )
    return parser.parse_args()


args = parse_args()
config = Config(env_name=args.env_name)
if config.seed is not None:
    set_seed(config.seed)

env = gym.make(config.env_name)
if config.seed is not None:
    env.action_space.seed(config.seed)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = DQN(state_size, action_size, hidden_layers=config.hidden_layers).to(config.device)
target_net = DQN(state_size, action_size, hidden_layers=config.hidden_layers).to(config.device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

memory = ReplayBuffer(
    config.memory_size,
    use_per=config.use_per,
    alpha=config.per_alpha,
    eps=config.per_eps,
)

agent = DQNAgent(policy_net, target_net, memory, config)

epsilon = config.epsilon
step_count = 0
episode_rewards = []
best_avg_reward = float("-inf")
saved_best = False

run_started_at = datetime.now().strftime("%Y%m%d-%H%M%S")
safe_env_name = config.env_name.replace("/", "_")
safe_model_name = Path(config.model_path).stem.replace("/", "_")
run_log_dir = Path("logs") / f"{safe_env_name}_{run_started_at}"
run_log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=str(run_log_dir))

metrics_dir = Path("metrics")
metrics_dir.mkdir(parents=True, exist_ok=True)
metrics_file = metrics_dir / f"{safe_env_name}_{safe_model_name}_{run_started_at}.csv"
metrics_fp = metrics_file.open("w", newline="", encoding="utf-8")
metrics_writer = csv.writer(metrics_fp)
metrics_writer.writerow([
    "episode",
    "reward",
    "avg100",
    "epsilon",
    "beta",
    "is_weight_mean",
    "td_error_mean",
    "priority_mean",
])

print(f"TensorBoard logs: {run_log_dir}")
print(f"CSV metrics: {metrics_file}")

try:
    for episode in range(1, config.num_episodes + 1):

        if episode == 1 and config.seed is not None:
            state, _ = env.reset(seed=config.seed)
        else:
            state, _ = env.reset()

        done = False
        total_reward = 0.0
        episode_losses = []
        episode_q_means = []
        episode_is_weight_means = []
        episode_td_error_means = []
        episode_betas = []

        while not done:

            action = agent.select_action(state, epsilon, env)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            train_reward = reward
            if config.env_name == "CartPole-v1" and terminated:
                # Penalize failure transitions to improve value separation.
                train_reward = -10.0

            memory.push(state, action, train_reward, next_state, done)

            state = next_state
            total_reward += float(reward)
            step_count += 1

            if len(memory) >= config.min_replay_size and step_count % config.train_every_steps == 0:
                beta = 1.0
                if config.use_per:
                    progress = min(1.0, step_count / max(1, config.per_beta_frames))
                    beta = config.per_beta_start + progress * (1.0 - config.per_beta_start)

                train_stats = agent.train_step(beta=beta)
                if train_stats is not None:
                    episode_losses.append(train_stats["loss"])
                    episode_q_means.append(train_stats["q_mean"])
                    writer.add_scalar("train/loss", train_stats["loss"], step_count)
                    writer.add_scalar("train/q_mean", train_stats["q_mean"], step_count)
                    writer.add_scalar("train/q_max_mean", train_stats["q_max_mean"], step_count)
                    writer.add_scalar("train/target_q_mean", train_stats["target_q_mean"], step_count)
                    writer.add_scalar("train/td_error_mean", train_stats["td_error_mean"], step_count)

                    if config.use_per and "indices" in train_stats:
                        memory.update_priorities(train_stats["indices"], train_stats["td_errors"])

                        priority_mean = memory.mean_priority()
                        is_weight_mean = float(train_stats.get("is_weight_mean", 0.0))

                        episode_betas.append(beta)
                        episode_is_weight_means.append(is_weight_mean)
                        episode_td_error_means.append(train_stats["td_error_mean"])

                        writer.add_scalar("train/beta", beta, step_count)
                        writer.add_scalar("train/is_weight_mean", is_weight_mean, step_count)
                        writer.add_scalar("train/priority_mean", priority_mean, step_count)

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)

        episode_rewards.append(total_reward)
        avg_reward_100 = float(np.mean(episode_rewards[-100:]))

        writer.add_scalar("episode/reward", total_reward, episode)
        writer.add_scalar("episode/avg100", avg_reward_100, episode)
        writer.add_scalar("episode/epsilon", epsilon, episode)
        episode_beta = float(np.mean(episode_betas)) if episode_betas else 1.0
        episode_is_weight = float(np.mean(episode_is_weight_means)) if episode_is_weight_means else 1.0
        episode_td_error = float(np.mean(episode_td_error_means)) if episode_td_error_means else 0.0
        episode_priority = memory.mean_priority() if config.use_per else 0.0

        metrics_writer.writerow([
            episode,
            total_reward,
            avg_reward_100,
            epsilon,
            episode_beta,
            episode_is_weight,
            episode_td_error,
            episode_priority,
        ])
        if episode_losses:
            writer.add_scalar("episode/loss", float(np.mean(episode_losses)), episode)
        if episode_q_means:
            writer.add_scalar("episode/q_mean", float(np.mean(episode_q_means)), episode)
        if config.use_per and episode_betas:
            writer.add_scalar("episode/beta", episode_beta, episode)
            writer.add_scalar("episode/is_weight_mean", episode_is_weight, episode)
            writer.add_scalar("episode/td_error_mean", episode_td_error, episode)
            writer.add_scalar("episode/priority_mean", episode_priority, episode)

        print(
            f"Episode {episode}, Reward: {total_reward:.1f}, "
            f"Avg100: {avg_reward_100:.1f}, Epsilon: {epsilon:.3f}"
        )

        if len(episode_rewards) >= 100 and avg_reward_100 > best_avg_reward:
            best_avg_reward = avg_reward_100
            torch.save(policy_net.state_dict(), config.model_path)
            saved_best = True

        if len(episode_rewards) >= 100 and avg_reward_100 > config.solved_threshold:
            torch.save(policy_net.state_dict(), config.model_path)
            saved_best = True
            print(
                f"Early stopping: Avg100 = {avg_reward_100:.1f} > {config.solved_threshold:.1f}. "
                f"Saved model to {config.model_path}"
            )
            break
finally:
    writer.close()
    metrics_fp.close()

if not saved_best:
    torch.save(policy_net.state_dict(), config.model_path)

env.close()

window = min(config.plot_window, len(episode_rewards))
if window >= 2:
    plt.figure(figsize=(10, 5))
    smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")

    plt.plot(episode_rewards, alpha=0.3, label="Reward")
    plt.plot(smoothed, linewidth=2, label=f"Moving Avg ({window})")
    plt.title("DQN Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.plot_path, dpi=150)
    plt.show()
