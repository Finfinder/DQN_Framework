# DQN Framework (PyTorch + Gymnasium)

Lekki framework do trenowania agenta **Deep Q-Network (DQN)** dla środowisk z dyskretną przestrzenią akcji w Gymnasium.

Aktualnie projekt wspiera konfiguracje:

- `CartPole-v1`
- `MountainCar-v0`
- `Acrobot-v1`
- `ALE/Pong-v5` (CNN DQN)

## Co zawiera projekt

- `train.py`: trening DQN + zapis najlepszego modelu + wykres postepu.
- `evaluate.py`: ewaluacja wytrenowanego modelu (greedy policy, epsilon=0) z podsumowaniem statystyk.
- `play.py`: uruchamianie wytrenowanego modelu w trybie `render_mode="human"`.
- `agents/dqn_agent.py`: logika agenta (epsilon-greedy, krok treningowy, soft update target network).
- `models/dqn_network.py`: MLP budowany dynamicznie z listy warstw ukrytych + factory `create_network(config, state_shape, action_dim)`.
- `models/cnn_dqn_network.py`: CNN DQN z konfigurowalnymi warstwami Conv2d i obsluga Dueling.
- `memory/replay_buffer.py`: trzy warianty replay bufora — `ReplayBuffer` (uniform), `PrioritizedReplayBuffer` (PER), `NstepReplayBuffer` (N-step returns) — z factory `create_buffer(config)`.
- `utils/evaluate.py`: wspolna funkcja `evaluate_policy()` uzywana przez `evaluate.py` i `train.py`.
- `utils/wrappers.py`: `make_env()` z `frame_skip`, `wrap_env()` z preprocessingiem obrazu (Atari + generyczne).
- `config/config.py`: centralna konfiguracja hiperparametrow i presetow per srodowisko.

## Wymagania

- Python 3.10+
- Pakiety Python:
  - `torch`
  - `gymnasium`
  - `gymnasium[atari]`, `ale-py` (opcjonalne, wymagane dla srodowisk Atari)
  - `opencv-python` (wymagane przez `AtariPreprocessing`)
  - `numpy`
  - `matplotlib`
  - `tensorboard`

Przykladowa instalacja:

```bash
pip install torch gymnasium numpy matplotlib tensorboard

# Opcjonalnie — dla srodowisk Atari (np. ALE/Pong-v5):
pip install "gymnasium[atari]" ale-py opencv-python
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
python train.py ALE/Pong-v5
```

1. Trening z okreslonym seedem (nadpisuje wartosc z configu):

```bash
python train.py MountainCar-v0 --seed 123
```

1. Ogladanie wytrenowanego agenta:

```bash
python play.py
python play.py CartPole-v1
python play.py MountainCar-v0 --play-episodes 10
```

1. Ewaluacja wytrenowanego modelu (greedy policy, epsilon=0):

```bash
python evaluate.py CartPole-v1
python evaluate.py MountainCar-v0 --episodes 50
python evaluate.py Acrobot-v1 --episodes 100 --render
python evaluate.py CartPole-v1 --render --render-episodes 5
```

## Jak to dziala (skrot)

- Agent wybiera akcje przez **epsilon-greedy**.
- Przejscia trafiaja do **Replay Buffer** (uniform, PER lub N-step — konfigurowalny przez `buffer_type`).
- Update sieci wykorzystuje wariant **Double DQN**:
  - wybor akcji `argmax` przez `policy_net`,
  - ewaluacja tej akcji przez `target_net`.
- `target_net` jest aktualizowana metoda **soft update** z parametrem `tau` (lub **hard update** co `target_update_freq` krokow dla CNN).
- Dla `CartPole-v1` przejscia koncowe (`terminated`) dostaja kara treningowa `-10.0`.
- Funkcja straty: Smooth L1 (Huber loss).
- Dla srodowisk obrazowych (np. `ALE/Pong-v5`) stosowana jest siec **CNN DQN** z preprocessingiem klatek (grayscale, resize, frame stacking).

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

Replay Buffer:

- `buffer_type`: typ bufora — `"replay"` (uniform), `"prioritized"` (PER), `"nstep"` (N-step returns). Domyslnie `"prioritized"`. Automatycznie ustawia `use_per`.
- `nstep_n`: liczba krokow N-step return (tylko gdy `buffer_type: "nstep"`). Domyslnie `3`.

| `buffer_type` | Klasa | Opis |
|---------------|-------|------|
| `"replay"` | `ReplayBuffer` | Uniform sampling z deque. Prosty, szybki. |
| `"prioritized"` | `PrioritizedReplayBuffer` | PER z IS weights. Lepszy dla sparse rewards. |
| `"nstep"` | `NstepReplayBuffer` | N-step returns + uniform. Przyspiesza propagacje wartosci. |

Parametry PER (aktywne gdy `buffer_type: "prioritized"`):

- `per_alpha`: sila priorytetyzacji (0.0 = uniform sampling).
- `per_beta_start`: poczatkowa wartosc beta dla wag IS.
- `per_beta_frames`: liczba krokow do annealingu beta do 1.0.
- `per_eps`: mala stala dodawana do priorytetu dla stabilnosci numerycznej.

Parametry architektury:

- `use_dueling`: wlacza/wylacza Dueling DQN (domyslnie `False`). Gdy `True`, siec rozdziela estymacje wartosci stanu i przewagi akcji dla lepszej generalizacji. Wszystkie artefakty (model, logi, metryki, wykresy) sa przechowywane oddzielnie dla standardowego DQN i Dueling DQN przy uzyciu sufixu (`_standard` lub `_dueling`).
- `network_type`: typ sieci — `"mlp"` (domyslnie) lub `"cnn"` (dla srodowisk obrazowych).
- `conv_layers`: lista warstw Conv2d jako krotki `(out_channels, kernel_size, stride)`. Domyslnie `[(32, 8, 4), (64, 4, 2), (64, 3, 1)]`.
- `cnn_hidden_dim`: rozmiar warstwy ukrytej po CNN trunk.
- `frame_stack`: liczba klatek stackowanych jako obserwacja (domyslnie `4`).
- `frame_size`: docelowy rozmiar klatki `[H, W]` (domyslnie `[84, 84]`).
- `frame_skip`: liczba klatek pomijanych (action repeat, domyslnie `1`).
- `is_atari`: flaga sterujaca wyborem wrapperow Atari vs generycznych.
- `target_update_freq`: hard update target network co N krokow (0 = soft update z `tau`).

Parametry ewaluacji:

- `eval_every`: co ile epizodow treningowych uruchamiana jest ewaluacja greedy policy (domyslnie `100`).
- `eval_episodes`: liczba epizodow ewaluacyjnych (domyslnie `10`).

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
- logi TensorBoard do katalogu `logs/<env_name><suffix>_<YYYYMMDD-HHMMSS>/`.
- metryki epizodow do CSV: `metrics/<env_name>_<model_name>_<YYYYMMDD-HHMMSS>.csv`.

Przykladowy sufiks: `_standard` dla standardowego DQN, `_dueling` dla Dueling DQN. Sufiks jest automatycznie dodawany do nazw modelu i wykresu, a przez to posrednio rowniez widoczny w nazwach metryk.

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
- `train/beta` (gdy `buffer_type: "prioritized"`)
- `train/is_weight_mean` (gdy `buffer_type: "prioritized"`)
- `train/priority_mean` (gdy `buffer_type: "prioritized"`)
- `eval/mean_reward` (greedy policy)
- `eval/std_reward` (greedy policy)
- `eval/min_reward` (greedy policy)
- `eval/max_reward` (greedy policy)

CSV zawiera kolumny:

- `episode`
- `reward`
- `avg100`
- `epsilon`
- `beta`
- `is_weight_mean`
- `td_error_mean`
- `priority_mean`

Oddzielny plik CSV ewaluacyjny (`*_eval.csv`) zawiera kolumny:

- `episode`
- `mean_reward`
- `std_reward`
- `min_reward`
- `max_reward`

Przykladowe artefakty widoczne w repo:

- `dqn_cartpole.pth`
- `training_curve_cartpole.png`

## Uwagi

- `play.py` rzuci `FileNotFoundError`, jesli model nie istnieje.
- Jesli chcesz powtarzalnych wynikow, ustaw `seed` w `Config.DEFAULTS` lub uzyj flagi `--seed` przy wywolaniu `train.py` (np. `python train.py MountainCar-v0 --seed 42`). Flaga `--seed` nadpisuje wartosc z configu.
