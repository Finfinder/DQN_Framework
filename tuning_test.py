"""Tuning validation: run environment with multiple seeds, report success rate."""
import sys
import os
import random
import numpy as np
import torch

os.environ["MPLBACKEND"] = "Agg"

from config.config import Config
from utils.wrappers import make_env, wrap_env
from utils.evaluate import evaluate_policy
from models.dqn_network import create_network
from memory.replay_buffer import create_buffer
from agents.dqn_agent import DQNAgent

SEEDS = [42, 123, 456, 789, 234, 567, 999, 1337, 2025, 777, 314, 628]
ENV = sys.argv[1] if len(sys.argv) > 1 else "CartPole-v1"


def run_seed(seed):
    config = Config(ENV)
    config.seed = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(config.env_name, frame_skip=config.frame_skip)
    env.action_space.seed(seed)
    env, state_shape = wrap_env(env, config)
    action_size = env.action_space.n

    policy_net = create_network(config, state_shape, action_size).to(config.device)
    target_net = create_network(config, state_shape, action_size).to(config.device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    memory = create_buffer(config)
    agent = DQNAgent(policy_net, target_net, memory, config)

    epsilon = config.epsilon
    step_count = 0
    episode_rewards = []

    for episode in range(1, config.num_episodes + 1):
        state, _ = env.reset(seed=seed) if episode == 1 else env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state, epsilon, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            train_reward = reward
            if config.env_name == "CartPole-v1" and terminated:
                train_reward = -10.0
            elif config.env_name == "MountainCar-v0":
                train_reward = reward + 10 * abs(next_state[1])

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
                if train_stats is not None and config.use_per and "indices" in train_stats:
                    memory.update_priorities(train_stats["indices"], train_stats["td_errors"])

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        episode_rewards.append(total_reward)
        avg100 = float(np.mean(episode_rewards[-100:]))

        if episode % 100 == 0:
            print(f"  Ep {episode}, Avg100: {avg100:.1f}, Eps: {epsilon:.4f}", flush=True)

        if len(episode_rewards) >= 100 and avg100 > config.solved_threshold:
            env.close()
            return True, episode, avg100

        if episode % config.eval_every == 0:
            policy_net.eval()
            eval_stats = evaluate_policy(
                policy_net, config, config.eval_episodes,
                config.device, seed=config.seed,
            )
            policy_net.train()
            if eval_stats["mean_reward"] > config.solved_threshold:
                env.close()
                return True, episode, eval_stats["mean_reward"]

    env.close()
    final_avg = float(np.mean(episode_rewards[-100:]))
    return False, config.num_episodes, final_avg


if __name__ == "__main__":
    results = []
    for seed in SEEDS:
        print(f"=== Seed {seed} ===", flush=True)
        solved, ep, avg = run_seed(seed)
        status = f"SOLVED at ep {ep}" if solved else f"FAILED (avg100={avg:.1f})"
        print(f"  -> {status}", flush=True)
        results.append((seed, solved, ep, avg))

    print()
    print("=== RESULTS ===")
    solved_count = sum(1 for _, s, _, _ in results if s)
    for seed, s, ep, avg in results:
        status = f"SOLVED at ep {ep}" if s else f"FAILED (avg100={avg:.1f})"
        print(f"Seed {seed}: {status}")
    print(f"Success rate: {solved_count}/{len(SEEDS)} = {solved_count/len(SEEDS)*100:.0f}%")
    target = len(SEEDS) * 2 // 3
    print(f"Target: {target}/{len(SEEDS)} ({target/len(SEEDS)*100:.0f}%)")
    print("PASS" if solved_count >= target else "FAIL")
