import torch

class Config:

    DEFAULTS = {
        "seed": None,
        "gamma": 0.99,
        "lr": 0.001,
        "batch_size": 64,
        "memory_size": 10000,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "tau": 0.01,
        "hidden_layers": [128, 128],
        "num_episodes": 1000,
        "min_replay_size": 1000,
        "train_every_steps": 4,
        "solved_threshold": 400.0,
        "plot_window": 20,
        "play_episodes": 5,
        "model_path": "model.pth",
        "plot_path": "training_curve.png",
    }

    ENV_CONFIG = {
        "CartPole-v1": {
            "hidden_layers": [64, 64],
            "num_episodes": 800,
            "epsilon_decay": 0.985,
            "solved_threshold": 400.0,
            "model_path": "dqn_cartpole.pth",
            "plot_path": "training_curve_cartpole.png",
        },
        "MountainCar-v0": {
            "hidden_layers": [128, 128],
            "num_episodes": 2000,
            "solved_threshold": -110.0,
            "model_path": "dqn_mountaincar.pth",
            "plot_path": "training_curve_mountaincar.png",
        },
        "Acrobot-v1": {
            "hidden_layers": [128, 128],
            "num_episodes": 1500,
            "solved_threshold": -85.0,
            "model_path": "dqn_acrobot.pth",
            "plot_path": "training_curve_acrobot.png",
        },
    }

    def __init__(self, env_name="CartPole-v1"):
        if env_name not in self.ENV_CONFIG:
            available = ", ".join(sorted(self.ENV_CONFIG.keys()))
            raise ValueError(f"Unknown environment '{env_name}'. Available: {available}")

        merged = dict(self.DEFAULTS)
        merged.update(self.ENV_CONFIG[env_name])

        self.env_name = env_name
        self.seed = merged["seed"]
        self.gamma = merged["gamma"]
        self.lr = merged["lr"]
        self.batch_size = merged["batch_size"]
        self.memory_size = merged["memory_size"]
        self.epsilon = merged["epsilon"]
        self.epsilon_decay = merged["epsilon_decay"]
        self.epsilon_min = merged["epsilon_min"]
        self.tau = merged["tau"]
        self.hidden_layers = list(merged["hidden_layers"])
        self.num_episodes = merged["num_episodes"]
        self.min_replay_size = merged["min_replay_size"]
        self.train_every_steps = merged["train_every_steps"]
        self.solved_threshold = merged["solved_threshold"]
        self.plot_window = merged["plot_window"]
        self.play_episodes = merged["play_episodes"]
        self.model_path = merged["model_path"]
        self.plot_path = merged["plot_path"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")