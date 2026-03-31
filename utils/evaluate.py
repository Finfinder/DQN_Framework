import numpy as np
import torch

from utils.wrappers import make_env, wrap_env


def evaluate_policy(model, config, num_episodes, device, seed=None):
    env = make_env(config.env_name, frame_skip=config.frame_skip)
    env, _ = wrap_env(env, config)

    model.eval()
    rewards = []

    try:
        for i in range(num_episodes):
            if i == 0 and seed is not None:
                state, _ = env.reset(seed=seed)
            else:
                state, _ = env.reset()

            done = False
            total_reward = 0.0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = q_values.argmax().item()

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)

            rewards.append(total_reward)
    finally:
        env.close()

    rewards_arr = np.array(rewards)
    return {
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std()),
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "rewards": rewards,
    }
