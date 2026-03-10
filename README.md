# DQN Framework (PyTorch + Gymnasium)

Lekki framework do trenowania agenta **Deep Q-Network (DQN)** dla środowisk z dyskretną przestrzenią akcji w Gymnasium.

Aktualnie projekt wspiera konfiguracje:

- `CartPole-v1`
- `MountainCar-v0`
- `Acrobot-v1`

## Co zawiera projekt

- `train.py`: trening DQN + zapis najlepszego modelu + wykres postepu.
- `play.py`: uruchamianie wytrenowanego modelu w trybie `render_mode="human"`.
- `agents/dqn_agent.py`: logika agenta (epsilon-greedy, krok treningowy, soft update target network).
- `models/dqn_network.py`: MLP budowany dynamicznie z listy warstw ukrytych.
- `memory/replay_buffer.py`: replay buffer oparty o `deque`.
- `config/config.py`: centralna konfiguracja hiperparametrow i presetow per srodowisko.

## Wymagania

- Python 3.10+
- Pakiety Python:
  - `torch`
  - `gymnasium`
  - `numpy`
  - `matplotlib`

Przykladowa instalacja:

```bash
pip install torch gymnasium numpy matplotlib
```

## Szybki start

1. Trening:

```bash
python train.py
```

1. Trening dla konkretnego srodowiska:

```bash
python train.py CartPole-v1
python train.py MountainCar-v0
python train.py Acrobot-v1
```

1. Ogladanie wytrenowanego agenta:

```bash
python play.py
python play.py CartPole-v1
python play.py MountainCar-v0 --play-episodes 10
```

## Jak to dziala (skrot)

- Agent wybiera akcje przez **epsilon-greedy**.
- Przejscia trafiaja do **Replay Buffer**.
- Update sieci wykorzystuje wariant **Double DQN**:
  - wybor akcji `argmax` przez `policy_net`,
  - ewaluacja tej akcji przez `target_net`.
- `target_net` jest aktualizowana metoda **soft update** z parametrem `tau`.
- Dla `CartPole-v1` przejscia koncowe (`terminated`) dostaja kara treningowa `-10.0`.

## Konfiguracja

Konfiguracja znajduje sie w `config/config.py` w klasie `Config`.

Najwazniejsze pola:

- `gamma`, `lr`, `batch_size`, `memory_size`
- `epsilon`, `epsilon_decay`, `epsilon_min`
- `tau`
- `hidden_layers`
- `num_episodes`, `min_replay_size`, `train_every_steps`
- `solved_threshold`
- `model_path`, `plot_path`, `play_episodes`

Domyslnie `python train.py` uruchamia preset dla `CartPole-v1`.

Aby uruchomic trening dla innego wspieranego srodowiska, podaj je jako argument:

```bash
python train.py MountainCar-v0
python train.py Acrobot-v1
```

Uwaga: wspierane sa tylko srodowiska z `Config.ENV_CONFIG`.

## Wyniki i artefakty

Podczas treningu zapisywane sa:

- plik wag modelu (`*.pth`) do `config.model_path`,
- wykres uczenia (`*.png`) do `config.plot_path`.

Przykladowe artefakty widoczne w repo:

- `dqn_cartpole.pth`
- `training_curve_cartpole.png`

## Uwagi

- `play.py` rzuci `FileNotFoundError`, jesli model nie istnieje.
- Jesli chcesz powtarzalnych wynikow, ustaw `seed` w `Config.DEFAULTS`.
