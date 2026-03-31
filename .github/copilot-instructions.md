# Instrukcje dla Copilota — DQN Framework

## Opis projektu

DQN Framework to lekki framework do trenowania agentów **Deep Q-Network (DQN)** w środowiskach Gymnasium z dyskretną przestrzenią akcji. Projekt implementuje warianty Double DQN, Dueling DQN, Prioritized Experience Replay (PER) i N-step returns.

Aktualnie wspierane środowiska: `CartPole-v1`, `MountainCar-v0`, `Acrobot-v1`, `ALE/Pong-v5`.

## Struktura projektu

```
train.py              — główny skrypt treningowy (pętla epizodów, logowanie, early stopping)
evaluate.py           — standalone ewaluacja wytrenowanego modelu (greedy policy, epsilon=0)
play.py               — wizualizacja agenta w trybie render_mode="human"
version.py            — jedyne źródło prawdy o wersji projektu (__version__)
CHANGELOG.md          — historia zmian (format Keep a Changelog + SemVer)
analysis.ipynb        — notebook Jupyter do eksploracji metryk

config/config.py      — klasa Config z domyślnymi hiperparametrami i presetami per środowisko
models/dqn_network.py — sieć MLP DQN (standard + dueling), budowana dynamicznie z hidden_layers + factory create_network()
models/cnn_dqn_network.py — sieć CNN DQN (standard + dueling), konfigurowalne warstwy Conv2d
agents/dqn_agent.py   — DQNAgent: epsilon-greedy, Double DQN train_step, soft update
memory/replay_buffer.py — ReplayBuffer (uniform), PrioritizedReplayBuffer (PER), NstepReplayBuffer + factory create_buffer()
utils/evaluate.py     — funkcja evaluate_policy() współdzielona przez train.py i evaluate.py
utils/analyze.py      — narzędzia do analizy metryk CSV (list_runs, compare_runs, diagnose, itp.)
utils/wrappers.py     — wrap_env() do preprocessingu obrazu dla CNN (Atari + generyczne envs)
```

## Konwencje kodu

### Styl i formatowanie

- Python 3.10+.
- Brak type annotations i docstringów w istniejącym kodzie — nie dodawaj ich samodzielnie, chyba że użytkownik wyraźnie poprosi.
- Komentarze tylko tam, gdzie logika nie jest oczywista (np. reward shaping, wzory matematyczne).
- Import: standardowe moduły → zewnętrzne pakiety → lokalne moduły (config, models, agents, memory, utils, version).

### Konwencja nazewnictwa

- Klasy: `PascalCase` (`DQNAgent`, `ReplayBuffer`, `Config`).
- Funkcje i zmienne: `snake_case` (`train_step`, `select_action`, `episode_rewards`).
- Stałe konfiguracyjne: `UPPER_SNAKE_CASE` (`DEFAULTS`, `ENV_CONFIG`).
- Nazwy plików: `snake_case.py`.

### Zarządzanie ścieżkami

- Używaj `pathlib.Path` do obsługi ścieżek (nie `os.path`).
- Env name w nazwach artefaktów: `config.env_name.replace("/", "_")`.
- Format timestampa: `"%Y%m%d-%H%M%S"`.
- Sufiksy artefaktów: `_dueling`, `_standard`, `_cnn_dueling`, `_cnn_standard` — generowane automatycznie przez `Config.suffix`.

### Obsługa urządzeń (device)

- Device wykrywany automatycznie w `Config`: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- Tensory i modele przenoszone na device przez `.to(config.device)`.
- Przy inferencji: `torch.no_grad()` context manager.

### Konfiguracja

- Cała konfiguracja w `config/config.py`, klasa `Config`.
- Dwupoziomowa: `DEFAULTS` (globalne) → `ENV_CONFIG` (nadpisania per środowisko).
- Nowe hiperparametry: dodaj do `DEFAULTS`, a nadpisania do odpowiednich wpisów w `ENV_CONFIG`.
- Nie twórz osobnych plików YAML/JSON — config jest w czystym Pythonie.

### Replay buffer

- Trzy warianty: `"replay"` (uniform), `"prioritized"` (PER), `"nstep"` (N-step returns).
- Factory: `create_buffer(config)` — nie twórz instancji buforów bezpośrednio.
- Każdy bufor musi implementować: `push()`, `sample()`, `update_priorities()`, `mean_priority()`, `__len__()`.
- PER zwraca 7-elementową krotkę (z indices i is_weights), pozostałe 5-elementową.

### Sieć neuronowa

Dwa typy sieci wybierane przez `network_type` w config (`"mlp"` lub `"cnn"`):

**MLP DQN** (`models/dqn_network.py`, klasa `DQN`):
- Architektura budowana dynamicznie z listy `hidden_layers` (np. `[128, 128]`).
- Parametr `dueling=True/False` przełącza między Standard DQN a Dueling DQN.
- Aktywacje: ReLU.
- Agregacja Dueling: `Q(s,a) = V(s) + (A(s,a) - mean(A(s,.)))`.

**CNN DQN** (`models/cnn_dqn_network.py`, klasa `CNNDQN`):
- Konfigurowalna lista warstw Conv2d przez `conv_layers` (np. `[(32, 8, 4), (64, 4, 2), (64, 3, 1)]`).
- Każda krotka to `(out_channels, kernel_size, stride)`.
- Po conv trunk: Flatten → Linear(`flatten_size`, `cnn_hidden_dim`) → ReLU → head.
- `flatten_size` obliczany automatycznie dummy forward pass.
- Dueling: identyczna logika jak w MLP DQN (value_head + advantage_head).
- Wymaga preprocessingu obrazu — patrz sekcja "Wrappery środowiska".

**Factory**: `create_network(config, state_shape, action_dim)` w `models/dqn_network.py` — nie twórz instancji sieci bezpośrednio.

### Wrappery środowiska

- Funkcja `wrap_env(env, config)` w `utils/wrappers.py` opakowuje env gdy `network_type == "cnn"`.
- Dla `is_atari == True`: `AtariPreprocessing` (grayscale, resize) → `FrameStackObservation` → `TransformObservation` (normalize /255.0).
- Dla `is_atari == False` i `network_type == "cnn"`: `ResizeObservation` → `GrayscaleObservation` → `FrameStackObservation` → `TransformObservation`.
- Dla `network_type == "mlp"`: zwraca env bez zmian.
- Zwraca `(wrapped_env, state_shape)` — state_shape to `env.observation_space.shape` po wrappowaniu.
- Parametry: `frame_stack` (liczba klatek), `frame_size` (docelowy rozmiar [H, W]), `is_atari` (flaga sterownicza).

### Agent (DQNAgent)

- Epsilon-greedy w `select_action()`.
- `train_step()` implementuje Double DQN: policy_net wybiera akcję, target_net ją ewaluuje.
- Soft update target network po każdym kroku treningowym (parametr `tau`).
- Gradient clipping: `clip_grad_norm_(parameters, 1.0)`.
- `train_step()` zwraca dict ze statystykami (`loss`, `q_mean`, `td_error_mean`, itp.) lub `None`.

## Logowanie i metryki

### TensorBoard

- Katalog logów: `logs/<env><suffix>_<YYYYMMDD-HHMMSS>/`.
- Metryki krokowe (`step_count`): `train/loss`, `train/q_mean`, `train/q_max_mean`, `train/target_q_mean`, `train/td_error_mean`, `train/beta`, `train/is_weight_mean`, `train/priority_mean`.
- Metryki epizodowe (`episode`): `episode/reward`, `episode/avg100`, `episode/epsilon`, `episode/loss`, `episode/q_mean`, `episode/beta`, `episode/is_weight_mean`, `episode/td_error_mean`, `episode/priority_mean`.
- Metryki ewaluacyjne: `eval/mean_reward`, `eval/std_reward`, `eval/min_reward`, `eval/max_reward`.
- Meta: `meta/version`.

### CSV

- Trening: `metrics/<env>_<model>_<timestamp>.csv` — kolumny: `episode`, `reward`, `avg100`, `epsilon`, `beta`, `is_weight_mean`, `td_error_mean`, `priority_mean`.
- Eval (w trakcie treningu): `metrics/<env>_<model>_<timestamp>_eval.csv` — kolumny: `episode`, `mean_reward`, `std_reward`, `min_reward`, `max_reward`.
- Eval (standalone): `metrics/<env>_<model>_standalone_eval_<timestamp>.csv`.

## Artefakty

- Wagi modelu: `*.pth` w katalogu głównym (np. `dqn_cartpole_dueling.pth`).
- Wykresy: `training_curve_*.png` w katalogu głównym.
- Logi TensorBoard: `logs/`.
- Metryki CSV: `metrics/`.
- Pliki `*.pth` i `*.png` są wersjonowane w git; `logs/` i `metrics/` są w `.gitignore`.

## Wersjonowanie

- Wersja projektu w `version.py` (`__version__`).
- Historia zmian w `CHANGELOG.md` (format Keep a Changelog + Semantic Versioning).
- Przy zmianie wersji: zaktualizuj `__version__` w `version.py` i dodaj wpis w `CHANGELOG.md`.
- Flaga `--version` dostępna we wszystkich skryptach CLI (`train.py`, `evaluate.py`, `play.py`).

### Branche i wypychanie zmian

- Każda wersja ma własny branch (np. `1.0.0`, `1.0.1`, `1.1.0`).
- Aktywny rozwój odbywa się na branchu odpowiadającym aktualnej wersji.
- Po commicie na danym branchu (np. `1.0.0`) zmiany propaguj w górę do wszystkich wyższych wersji, a najwyższą merguj do `master`:
  1. Commituj i pushuj zmiany na aktualnym branchu (np. `1.0.0`).
  2. Merguj do każdego wyższego brancha wersyjnego w kolejności rosnącej (np. `1.0.0` → `1.0.1` → `1.1.0`).
  3. Najwyższy branch wersyjny merguj dodatkowo do `master`.
  4. Pushuj każdy zmergowany branch.
  5. Wróć na branch, na którym pracowałeś.

Przykład — commit na `1.0.0`, istniejące branche: `1.0.0`, `1.0.1`, `1.1.0`:

```bash
# 1. Commit i push na aktualnym branchu
git add .
git commit -m "opis zmiany"
git push origin 1.0.0

# 2. Propagacja w górę
git checkout 1.0.1
git merge 1.0.0
git push origin 1.0.1

git checkout 1.1.0
git merge 1.0.1
git push origin 1.1.0

# 3. Najwyższa wersja → master
git checkout master
git merge 1.1.0
git push origin master

# 4. Powrót do brancha roboczego
git checkout 1.0.0
```

## Dodawanie nowego środowiska

1. Dodaj wpis do `Config.ENV_CONFIG` w `config/config.py` z dostrojonymi hiperparametrami.
2. Ustaw `model_path` i `plot_path` — sufiks `_dueling`/`_standard` (lub `_cnn_dueling`/`_cnn_standard`) zostanie dodany automatycznie.
3. Ustaw `solved_threshold` dla early stopping.
4. Dla środowisk obrazowych: ustaw `network_type: "cnn"`, `is_atari: True/False`, `conv_layers`, `cnn_hidden_dim`, `frame_stack`, `frame_size`.
5. Przetestuj trening: `python train.py <NazwaSrodowiska>`.

## Dodawanie nowej funkcjonalności do treningu

- Nowe metryki: dodaj do `train_stats` w `DQNAgent.train_step()`, zaloguj w `train.py` przez `writer.add_scalar()` i `metrics_writer.writerow()`.
- Nowe hiperparametry: dodaj do `Config.DEFAULTS`, opcjonalnie nadpisz w `Config.ENV_CONFIG`, i odczytaj w `__init__`.
- Nowy typ bufora: zaimplementuj klasę z interfejsem `push/sample/update_priorities/mean_priority/__len__`, dodaj do `create_buffer()`.
- Nowy typ sieci: zaimplementuj klasę `nn.Module` z interfejsem `forward(x) → q_values`, dodaj do `create_network()` w `models/dqn_network.py`.

## GitHub Actions (CI workflow)

- Workflow CI jest zdefiniowany w `.github/workflows/ci.yml`.
- Uruchamia się dla `push`, `pull_request` oraz ręcznie przez `workflow_dispatch`.
- Job `lint-and-smoke` wykonuje:
  - instalację lekkiego zestawu zależności CPU do CI,
  - lint krytycznych błędów (`ruff check . --select E9,F63,F7,F82`),
  - kompilację wszystkich plików Python (`python -m compileall -q .`),
  - smoke testy CLI (`train.py --version`, `evaluate.py --version`, `play.py --version`),
  - smoke test konfiguracji i podstawowego kroku środowiska Gymnasium.
- Przy zmianach w CLI, konfiguracji envów (`Config.ENV_CONFIG`) lub wrapperach środowiska, utrzymuj kompatybilność z tym smoke testem i aktualizuj CI jeśli to konieczne.
- CI ma być szybkie i deterministyczne: nie dodawaj długich treningów ani testów wymagających GPU do tego workflow.

## Środowisko wirtualne (wymagane)

- Wszystkie komendy wykonywane w terminalu muszą być uruchamiane po aktywacji lokalnego środowiska `.venv`.
- Dotyczy to wszystkich poleceń związanych z Pythonem i jego środowiskiem: `python`, `pip`, `tensorboard`, `jupyter`, testy, trening, ewaluacja i play.
- Nie uruchamiaj poleceń Pythonowych na interpreterze systemowym.

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

cmd:

```bat
.venv\Scripts\activate.bat
```

## Komendy

```bash
# Trening
python train.py                          # CartPole-v1 (domyślnie)
python train.py MountainCar-v0           # Konkretne środowisko
python train.py MountainCar-v0 --seed 42 # Z określonym seedem

# Ewaluacja
python evaluate.py CartPole-v1
python evaluate.py MountainCar-v0 --episodes 50
python evaluate.py Acrobot-v1 --render --render-episodes 5

# Wizualizacja
python play.py CartPole-v1
python play.py MountainCar-v0 --play-episodes 10

# TensorBoard
tensorboard --logdir logs

# Wersja
python train.py --version
```

## Zależności

- `torch` — sieci neuronowe, optymalizacja, CUDA.
- `gymnasium` — środowiska RL.
- `gymnasium[atari]`, `ale-py` — środowiska Atari (opcjonalne, wymagane dla `ALE/Pong-v5` i innych Atari envs).
- `opencv-python` — wymagane przez `AtariPreprocessing` (wrappery CNN dla Atari).
- `numpy` — operacje numeryczne.
- `matplotlib` — wykresy treningowe.
- `tensorboard` — logowanie metryk w czasie rzeczywistym.
- `pandas` — analiza metryk CSV (utils/analyze.py, analysis.ipynb).
