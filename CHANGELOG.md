# Changelog

Wszystkie istotne zmiany w projekcie DQN Framework są dokumentowane w tym pliku.

Format oparty na [Keep a Changelog](https://keepachangelog.com/pl/1.1.0/),
projekt stosuje [Semantic Versioning](https://semver.org/lang/pl/).

## [1.0.1] - Unreleased

### Dodane
- Testy jednostkowe (`tests/`) — 101 testów pokrywających Config, ReplayBuffer (wszystkie warianty), DQNAgent, utils/training, utils/evaluate, utils/wrappers; integracja z pytest + pytest-cov.
- Moduł `utils/training.py` — współdzielona logika treningowa: `run_episode()`, `compute_beta()`, `shape_reward()`, `compute_avg100()` wydzielona z `train.py` i `tuning_test.py`.
- Parametr `weight_decay` w `Config.DEFAULTS` (domyślnie `0`) i przekazywany do optymizatora Adam w `DQNAgent`.
- Konfiguracja coverage w CI (`ci.yml`) i SonarCloud (`sonar.yml`) — generowanie `coverage.xml` przed skanem.
- Integracja SonarCloud — workflow `.github/workflows/sonar.yml` z analizą na push/PR, pokryciem kodu pytest i konfiguracją `sonar-project.properties`.
- Shared binding SonarQube for IDE — `.sonarlint/connectedMode.json` dla Connected Mode z SonarCloud.

### Naprawione
- Konfiguracja SonarCloud: dodano `sonar.tests=tests`, `sonar.test.inclusions`, `sonar.coverage.exclusions` — pliki testowe i CLI entry-pointy poprawnie rozgraniczone od source'ów. Rozszerzono `sonar.coverage.exclusions` o `utils/analyze.py` i `models/cnn_dqn_network.py` (moduły nietestowalne w CI — standalone pandas tool i CNN wymagający GPU/Atari ROM).
- 11× S1244 (BUG): porównania float z `==` zamienione na `pytest.approx()` w `test_config.py` i `test_training.py`.
- S6709 (CODE SMELL): `PrioritizedReplayBuffer` — dodano parametr `seed=None` do `np.random.default_rng()`.
- 5× S1481 (CODE SMELL): nieużywane zmienne w `test_analyze.py` zamienione na `_prefix`.
- Redukcja Cognitive Complexity: `list_runs()`, `diagnose()`, `main()` w `utils/analyze.py` (przez ekstrakcję 7 prywatnych helperów); `run_seed()` w `tuning_test.py` (użycie `run_episode()` z `utils/training.py`).
- Usunięte nieużywane zmienne w `utils/analyze.py`: `meta_train`, `meta_eval` → `_`; `final_eps` usunięte.
- Puste metody `update_priorities()` w `ReplayBuffer` i `NstepReplayBuffer` opatrzone komentarzem `# No-op`.
- Nieużywane parametry interfejsu w `ReplayBuffer` i `NstepReplayBuffer`: `td_error` → `_td_error`, `beta` → `_beta`.
- Migracja `PrioritizedReplayBuffer.sample()` z `np.random.choice()` na `numpy.random.Generator` (`self.rng.choice()`).
- Duplikacja kodu między `train.py` i `tuning_test.py` — pętla treningowa i reward shaping wydzielone do `utils/training.py`.
- Błąd SonarCloud scan — usunięto `sonar.tests=tests` i `sonar.test.inclusions` z `sonar-project.properties` (brak katalogu `tests/` w repozytorium).
- Zaktualizowano `sonarqube-scan-action` z v5 na v6 (v5 zawiera lukę bezpieczeństwa i jest wycofana).

### Usunięte
- Plik `.github/copilot-instructions.md` — zastąpiony przez scoped instructions w `.github/instructions/dqn-framework.instructions.md`.

### Zmienione
- Sekcja „Środowisko wirtualne" w `.github/instructions/dqn-framework.instructions.md` rozszerzona o wymuszenie aktywacji venv (KRYTYCZNE dla CUDA/GPU).

- README po angielsku – pełne tłumaczenie `README.md` na język angielski z paskiem badge'ów (Python, PyTorch, Gymnasium, Version, License, CI) w stylu SeqMcpServer.
- Licencja MIT – dodano plik `LICENSE` (MIT, 2025, Finfinder).
- Plik `.github/instructions/dqn-framework.instructions.md` ze scoped konwencjami projektu (applyTo: `**`).
- Architektura CNN DQN (`models/cnn_dqn_network.py`) z konfigurowalnymi warstwami Conv2d i obsługą Dueling.
- Factory `create_network(config, state_shape, action_dim)` do automatycznego wyboru MLP lub CNN.
- Wrappery środowiska (`utils/wrappers.py`): `make_env()` z `frame_skip`, `wrap_env()` z preprocessingiem obrazu (Atari + generyczne).
- Preset `ALE/Pong-v5` z dedykowanymi hiperparametrami CNN.
- Nowe parametry konfiguracji: `network_type`, `conv_layers`, `cnn_hidden_dim`, `frame_stack`, `frame_size`, `frame_skip`, `is_atari`, `target_update_freq`, `adam_eps`.
- Wytrenowany model `dqn_pong_cnn_dueling.pth`.
- Konfigurowalny gradient clipping (`gradient_clip`) — domyślnie 1.0, umożliwia dostrojenie per środowisko.

### Zmienione
- Funkcja straty zmieniona z MSE na Smooth L1 (Huber loss) w `DQNAgent.train_step()`.
- Opcjonalny hard target update co `target_update_freq` kroków (gdy > 0) zamiast ciągłego soft update.
- Parametr `adam_eps` dodany do optymalizatora Adam.
- Pliki `train.py`, `evaluate.py`, `play.py` zaktualizowane do obsługi CNN i wrapperów środowiska.
- Metryki CSV flushowane po każdym zapisie dla szybszego podglądu.
- CI smoke test zaktualizowany dla nowego API `make_env`/`wrap_env`.
- Dostrojone hiperparametry CartPole-v1: `hidden_layers=[128,128]`, `epsilon_decay=0.993`, `lr=0.0005`, `tau=0.003`, `batch_size=128`, `memory_size=30000`, `train_every_steps=2`, `per_beta_frames=30000`, `adam_eps=1e-4`, `gradient_clip=0.3`. Osiągnięto 83% success rate (10/12 seedów) przy 800 epizodach.
- Zwiększono `num_episodes` CartPole-v1 z 800 do 900 — seed 42 (domyślny) wymagał 827 epizodów do rozwiązania, co powodowało deterministyczne niepowodzenie przy limicie 800.
- Dostrojone hiperparametry MountainCar-v0: `lr=0.001`, `tau=0.001`, `epsilon_decay=0.998`, `buffer_type="replay"`, `eval_every=50`, `eval_episodes=20`. Zmiana z PER na uniform replay buffer eliminuje niestabilność TD error. Osiągnięto 67% success rate (8/12 seedów) przy 3500 epizodach.
- Reward shaping dla MountainCar-v0: `reward + 10 * abs(velocity)` — zachęca agenta do budowania momentum.
- Eval-based early stopping w `train.py` — zapisuje model na najlepszym wyniku ewaluacji i kończy trening gdy eval mean > solved_threshold.
- Naprawiono blokowanie `plt.show()` — zmiana na `plt.show(block=False)` + `plt.close()` zapobiega zawieszeniu procesu po zakończeniu treningu.
- `tuning_test.py` — parametryzacja środowiska przez `sys.argv[1]`, eval-based early stopping, reward shaping per środowisko, rozszerzenie do 12 seedów.

## [1.0.0] - 2026-03-14

### Dodane
- Trening DQN z Double DQN update rule (`train.py`).
- Architektura Dueling DQN (`use_dueling` w konfiguracji).
- Trzy warianty replay bufora: uniform (`ReplayBuffer`), Prioritized Experience Replay (`PrioritizedReplayBuffer`), N-step returns (`NstepReplayBuffer`).
- Factory `create_buffer(config)` do automatycznego tworzenia bufora na podstawie konfiguracji.
- Soft update target network z parametrem `tau`.
- Centralna konfiguracja hiperparametrów w `config/config.py` z presetami per środowisko (`CartPole-v1`, `MountainCar-v0`, `Acrobot-v1`).
- Kara treningowa `-10.0` dla przejść terminalnych w `CartPole-v1`.
- Logowanie metryk do TensorBoard (`logs/`) i CSV (`metrics/`).
- Oddzielne pliki CSV dla metryk treningowych i ewaluacyjnych.
- Ewaluacja greedy policy co `eval_every` epizodów podczas treningu.
- Standalone ewaluacja modelu (`evaluate.py`) z opcją renderowania.
- Wizualizacja wytrenowanego agenta w trybie `render_mode="human"` (`play.py`).
- Automatyczne sufiksy `_dueling` / `_standard` dla artefaktów.
- Flaga `--seed` w `train.py` do nadpisywania seeda z konfiguracji.
- Early stopping po przekroczeniu `solved_threshold`.
- Wykres postępu treningu z wygładzaniem (moving average).
- Mechanizm wersjonowania (`version.py`) i plik CHANGELOG.
