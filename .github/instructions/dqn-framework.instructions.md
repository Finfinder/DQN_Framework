---
description: "Konwencje Python/PyTorch dla DQN Framework — architektura agentów DQN, wzorce sieci neuronowych, bufory doświadczeń, konfiguracja hiperparametrów, metryki treningowe."
applyTo: "**"
---

# DQN Framework

Lekki framework do trenowania agentów Deep Q-Network (DQN) w środowiskach Gymnasium z dyskretną przestrzenią akcji. Implementuje Double DQN, Dueling DQN, Prioritized Experience Replay (PER) i N-step returns.

## Stos technologiczny

- Python 3.10+
- `torch` — sieci neuronowe, optymalizacja, CUDA
- `gymnasium` + `gymnasium[atari]`, `ale-py` — środowiska RL
- `opencv-python` — preprocessing obrazu (Atari)
- `numpy` — operacje numeryczne
- `matplotlib` — wykresy treningowe
- `tensorboard` — logowanie metryk w czasie rzeczywistym
- `pandas` — analiza metryk CSV
- `ruff` — linting w CI
- Konfiguracja w czystym Pythonie (`config/config.py`) — NIE używaj YAML/JSON dla hiperparametrów

## Architektura

```
train.py / evaluate.py / play.py  — entry pointy CLI (--version, --seed)
config/config.py                  — klasa Config: DEFAULTS + ENV_CONFIG per środowisko
models/dqn_network.py             — DQN (MLP), factory create_network()
models/cnn_dqn_network.py         — CNNDQN (CNN), z dueling variant
agents/dqn_agent.py               — DQNAgent: epsilon-greedy, Double DQN, soft update
memory/replay_buffer.py           — ReplayBuffer, PrioritizedReplayBuffer, NstepReplayBuffer, factory create_buffer()
utils/evaluate.py                 — evaluate_policy() współdzielona przez train.py i evaluate.py
utils/wrappers.py                 — wrap_env() do preprocessingu obrazu dla CNN
utils/analyze.py                  — narzędzia do analizy metryk CSV
version.py                        — jedyne źródło prawdy o wersji (__version__)
```

- Środowisko wirtualne `.venv` WYMAGANE — wszystkie komendy po aktywacji
- Artefakty `*.pth` i `*.png` wersjonowane w git; `logs/` i `metrics/` w `.gitignore`

## Wzorzec sieci neuronowej

Dwa typy sieci wybierane przez `network_type` w config (`"mlp"` lub `"cnn"`):

- **MLP DQN** — dynamiczna architektura z `hidden_layers`, dueling przełączany przez `dueling=True/False`
- **CNN DQN** — konfigurowalna lista Conv2d przez `conv_layers`, `flatten_size` obliczany dummy forward pass
- Factory `create_network(config, state_shape, action_dim)` — NIE twórz instancji sieci bezpośrednio
- Dueling: `Q(s,a) = V(s) + (A(s,a) - mean(A(s,.)))`

## Wzorzec bufora doświadczeń

Trzy warianty: `"replay"` (uniform), `"prioritized"` (PER), `"nstep"` (N-step returns):

- Factory `create_buffer(config)` — NIE twórz instancji buforów bezpośrednio
- Każdy bufor musi implementować: `push()`, `sample()`, `update_priorities()`, `mean_priority()`, `__len__()`
- PER zwraca 7-elementową krotkę (z `indices` i `is_weights`), pozostałe 5-elementową

## Wzorzec agenta

- Epsilon-greedy w `select_action()`
- `train_step()` implementuje Double DQN: policy_net wybiera akcję, target_net ewaluuje
- Soft update target network po każdym kroku (parametr `tau`)
- Gradient clipping: `clip_grad_norm_(parameters, 1.0)`
- `train_step()` zwraca dict ze statystykami lub `None`

## Konfiguracja

- Klasa `Config` w `config/config.py` — dwupoziomowa: `DEFAULTS` → `ENV_CONFIG`
- Nowe hiperparametry: dodaj do `DEFAULTS`, nadpisania do `ENV_CONFIG`
- Device wykrywany automatycznie: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Sufiks artefaktów: `_dueling`, `_standard`, `_cnn_dueling`, `_cnn_standard` — generowany przez `Config.suffix`

## Konwencje kodowania

- **Brak type annotations i docstringów** — nie dodawaj, chyba że użytkownik wyraźnie poprosi
- Komentarze tylko tam, gdzie logika nie jest oczywista (reward shaping, wzory matematyczne)
- Import: standardowe moduły → zewnętrzne pakiety → lokalne moduły
- Klasy: `PascalCase` (`DQNAgent`, `ReplayBuffer`, `Config`)
- Funkcje i zmienne: `snake_case` (`train_step`, `select_action`)
- Stałe: `UPPER_SNAKE_CASE` (`DEFAULTS`, `ENV_CONFIG`)
- Pliki: `snake_case.py`
- Ścieżki: `pathlib.Path` — NIE `os.path`
- Env name w artefaktach: `config.env_name.replace("/", "_")`
- Inferencja: `torch.no_grad()` context manager

## Logowanie i metryki

- TensorBoard: `logs/<env><suffix>_<YYYYMMDD-HHMMSS>/`
- CSV trening: `metrics/<env>_<model>_<timestamp>.csv`
- CSV eval: `metrics/<env>_<model>_<timestamp>_eval.csv`
- Nowe metryki: dodaj do `train_stats` w `DQNAgent.train_step()`, zaloguj w `train.py`

## Czego NIE robić

- NIE dodawaj type annotations ani docstringów bez wyraźnej prośby
- NIE twórz plików YAML/JSON dla konfiguracji — config w czystym Pythonie
- NIE twórz instancji sieci/buforów bezpośrednio — używaj factory `create_network()` / `create_buffer()`
- NIE używaj `os.path` — wyłącznie `pathlib.Path`
- NIE dodawaj długich treningów ani testów GPU do CI

## CI (GitHub Actions)

- Workflow: `.github/workflows/ci.yml` — triggery: `push`, `pull_request`, `workflow_dispatch`
- Job `lint-and-smoke`: instalacja CPU-only deps, `ruff check . --select E9,F63,F7,F82`, `compileall`, smoke testy CLI (`--version`), smoke test konfiguracji + kroku środowiska
- CI musi być szybkie i deterministyczne — NIE dodawaj długich treningów ani testów GPU
- Przy zmianach w CLI, `Config.ENV_CONFIG` lub wrapperach — utrzymuj kompatybilność ze smoke testem

## Komendy

```bash
# Aktywacja środowiska (wymagane)
.\.venv\Scripts\Activate.ps1

# Trening
python train.py                          # CartPole-v1 (domyślnie)
python train.py MountainCar-v0           # Konkretne środowisko
python train.py MountainCar-v0 --seed 42 # Z seedem

# Ewaluacja
python evaluate.py CartPole-v1
python evaluate.py MountainCar-v0 --episodes 50
python evaluate.py Acrobot-v1 --render --render-episodes 5

# Wizualizacja
python play.py CartPole-v1
python play.py MountainCar-v0 --play-episodes 10

# TensorBoard
tensorboard --logdir logs
```

## Dodawanie nowego środowiska

1. Dodaj wpis do `Config.ENV_CONFIG` z dostrojonymi hiperparametrami
2. Ustaw `model_path`, `plot_path` — sufiks dodawany automatycznie przez `Config.suffix`
3. Ustaw `solved_threshold` dla early stopping
4. Dla środowisk obrazowych: `network_type: "cnn"`, `is_atari`, `conv_layers`, `cnn_hidden_dim`, `frame_stack`, `frame_size`

## Commit Convention

- Opis commita ZAWSZE w języku angielskim
- Format: krótki, imperatywny (np. `Add Pong environment support`, `Fix epsilon decay schedule`)
- Nie używaj prefiksów `feat:`, `fix:` — prostota ponad konwencje

## Przed commitem

Sprawdź, czy nie trzeba zaktualizować:
- `README.md` — jeśli zmiana wpływa na dokumentację użytkownika
- `CHANGELOG.md` — dodaj wpis w sekcji `[Unreleased]`
- `version.py` — jeśli zmiana wymaga nowej wersji
