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
- `memory/replay_buffer.py`: replay buffer w trybie `uniform` lub `Prioritized Experience Replay (PER)`.
- `config/config.py`: centralna konfiguracja hiperparametrow i presetow per srodowisko.

## Wymagania

- Python 3.10+
- Pakiety Python:
  - `torch`
  - `gymnasium`
  - `numpy`
  - `matplotlib`
  - `tensorboard`

Przykladowa instalacja:

```bash
pip install torch gymnasium numpy matplotlib tensorboard
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

Parametry PER:

- `use_per`: wlacza/wylacza PER (domyslnie `True`).
- `per_alpha`: sila priorytetyzacji (0.0 = uniform sampling).
- `per_beta_start`: poczatkowa wartosc beta dla wag IS.
- `per_beta_frames`: liczba krokow do annealingu beta do 1.0.
- `per_eps`: mala stala dodawana do priorytetu dla stabilnosci numerycznej.

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
- logi TensorBoard do katalogu `logs/<env_name>_<YYYYMMDD-HHMMSS>/`.
- metryki epizodow do CSV: `metrics/<env_name>_<model_name>_<YYYYMMDD-HHMMSS>.csv`.

Aby podejrzec metryki podczas/po treningu:

```bash
tensorboard --logdir logs
```

Logowane metryki obejmuja m.in.:

- `episode/reward`
- `episode/avg100`
- `episode/epsilon`
- `episode/loss`
- `episode/q_mean`
- `train/loss`
- `train/q_mean`
- `train/q_max_mean`
- `train/target_q_mean`
- `train/td_error_mean`
- `train/beta` (gdy `use_per=True`)
- `train/is_weight_mean` (gdy `use_per=True`)
- `train/priority_mean` (gdy `use_per=True`)

CSV zawiera kolumny:

- `episode`
- `reward`
- `avg100`
- `epsilon`
- `beta`
- `is_weight_mean`
- `td_error_mean`
- `priority_mean`

Przykladowe artefakty widoczne w repo:

- `dqn_cartpole.pth`
- `training_curve_cartpole.png`

## Uwagi

- `play.py` rzuci `FileNotFoundError`, jesli model nie istnieje.
- Jesli chcesz powtarzalnych wynikow, ustaw `seed` w `Config.DEFAULTS`.
