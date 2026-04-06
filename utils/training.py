"""Shared training utilities used by train.py and tuning_test.py."""

import numpy as np


def compute_beta(config, step_count):
    """Compute importance-sampling beta for PER.

    Returns 1.0 when not using PER, otherwise linearly anneals from
    per_beta_start to 1.0 over per_beta_frames steps.
    """
    if not config.use_per:
        return 1.0
    progress = min(1.0, step_count / max(1, config.per_beta_frames))
    return config.per_beta_start + progress * (1.0 - config.per_beta_start)


def shape_reward(env_name, reward, next_state, terminated):
    """Apply environment-specific reward shaping.

    CartPole-v1: penalize failure transitions (-10) to improve value separation.
    MountainCar-v0: add velocity bonus to encourage momentum building.
    All others: return reward unchanged.
    """
    if env_name == "CartPole-v1" and terminated:
        return -10.0
    if env_name == "MountainCar-v0":
        return reward + 10 * abs(next_state[1])
    return reward


def run_episode(env, agent, memory, config, epsilon, step_count, initial_state=None):
    """Run a single training episode and return episode stats.

    Args:
        env: Gymnasium environment
        agent: DQNAgent instance
        memory: replay buffer
        config: Config instance
        epsilon: current epsilon value
        step_count: global step counter (int)
        initial_state: optional pre-reset state (skips env.reset() when provided,
            useful for seeded first episodes)

    Returns:
        dict with keys:
            total_reward (float): undiscounted episode return
            step_count (int): updated global step counter
            train_stats_list (list[dict]): list of train_step stats dicts
    """
    if initial_state is not None:
        state = initial_state
    else:
        state, _ = env.reset()
    done = False
    total_reward = 0.0
    train_stats_list = []

    while not done:
        action = agent.select_action(state, epsilon, env)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        train_reward = shape_reward(config.env_name, reward, next_state, terminated)
        memory.push(state, action, train_reward, next_state, done)

        state = next_state
        total_reward += float(reward)
        step_count += 1

        if len(memory) >= config.min_replay_size and step_count % config.train_every_steps == 0:
            beta = compute_beta(config, step_count)
            train_stats = agent.train_step(beta=beta)
            if train_stats is not None:
                train_stats["beta"] = beta
                train_stats["step"] = step_count
                train_stats_list.append(train_stats)
                if config.use_per and "indices" in train_stats:
                    memory.update_priorities(train_stats["indices"], train_stats["td_errors"])

    return {
        "total_reward": total_reward,
        "step_count": step_count,
        "train_stats_list": train_stats_list,
    }


def compute_avg100(episode_rewards):
    """Return the mean of the last 100 episode rewards."""
    return float(np.mean(episode_rewards[-100:]))
