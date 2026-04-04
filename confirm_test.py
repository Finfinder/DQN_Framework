"""Quick confirmation run with additional seeds."""
import os
import random
import numpy as np
import torch

os.environ["MPLBACKEND"] = "Agg"

from config.config import Config
from utils.wrappers import make_env, wrap_env
from models.dqn_network import create_network
from memory.replay_buffer import create_buffer
from agents.dqn_agent import DQNAgent

SEEDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
results = []

for seed in SEEDS:
    print(f"=== Seed {seed} ===", flush=True)
    config = Config("CartPole-v1")
    config.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(config.env_name, frame_skip=config.frame_skip)
    env.action_space.seed(seed)
    env, state_shape = wrap_env(env, config)
    policy_net = create_network(config, state_shape, env.action_space.n).to(config.device)
    target_net = create_network(config, state_shape, env.action_space.n).to(config.device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    memory = create_buffer(config)
    agent = DQNAgent(policy_net, target_net, memory, config)

    epsilon = config.epsilon
    step_count = 0
    episode_rewards = []
    solved = False

    for episode in range(1, config.num_episodes + 1):
        state, _ = env.reset(seed=seed) if episode == 1 else env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state, epsilon, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            train_reward = -10.0 if terminated else reward
            memory.push(state, action, train_reward, next_state, done)
            state = next_state
            total_reward += float(reward)
            step_count += 1

            if len(memory) >= config.min_replay_size and step_count % config.train_every_steps == 0:
                beta = 1.0
                if config.use_per:
                    progress = min(1.0, step_count / max(1, config.per_beta_frames))
                    beta = config.per_beta_start + progress * (1.0 - config.per_beta_start)
                ts = agent.train_step(beta=beta)
                if ts and config.use_per and "indices" in ts:
                    memory.update_priorities(ts["indices"], ts["td_errors"])

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        episode_rewards.append(total_reward)
        avg100 = float(np.mean(episode_rewards[-100:]))

        if episode % 100 == 0:
            print(f"  Ep {episode}, Avg100: {avg100:.1f}", flush=True)

        if len(episode_rewards) >= 100 and avg100 > config.solved_threshold:
            print(f"  SOLVED at ep {episode}", flush=True)
            results.append((seed, True, episode, avg100))
            solved = True
            break

    if not solved:
        print(f"  FAILED ({avg100:.1f})", flush=True)
        results.append((seed, False, 800, avg100))
    env.close()

print()
for s, ok, ep, avg in results:
    tag = f"SOLVED@{ep}" if ok else f"FAIL({avg:.1f})"
    print(f"Seed {s}: {tag}")
print(f"Result: {sum(1 for _, ok, _, _ in results if ok)}/{len(results)}")
