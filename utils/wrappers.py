import gymnasium as gym
import numpy as np


def make_env(env_name, render_mode=None, frame_skip=None):
    if env_name.startswith("ALE/"):
        try:
            import ale_py
        except ImportError as exc:
            raise ImportError(
                "Environment uses ALE namespace. Install dependencies with: pip install \"gymnasium[atari]\" ale-py"
            ) from exc
        gym.register_envs(ale_py)

    kwargs = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    # Disable built-in frame skip for Atari so AtariPreprocessing handles it
    if env_name.startswith("ALE/") and frame_skip is not None and frame_skip > 1:
        kwargs["frameskip"] = 1
    return gym.make(env_name, **kwargs)


def wrap_env(env, config):
    if config.network_type != "cnn":
        state_shape = env.observation_space.shape
        return env, state_shape

    if config.is_atari:
        env = gym.wrappers.AtariPreprocessing(
            env,
            screen_size=config.frame_size[0],
            grayscale_obs=True,
            frame_skip=config.frame_skip,
        )
    else:
        env = gym.wrappers.ResizeObservation(env, config.frame_size)
        env = gym.wrappers.GrayscaleObservation(env)

    env = gym.wrappers.FrameStackObservation(env, stack_size=config.frame_stack)
    env = gym.wrappers.TransformObservation(
        env,
        func=lambda obs: np.array(obs, dtype=np.float32) / 255.0,
        observation_space=gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(config.frame_stack, config.frame_size[0], config.frame_size[1]),
            dtype=np.float32,
        ),
    )

    state_shape = env.observation_space.shape
    return env, state_shape
