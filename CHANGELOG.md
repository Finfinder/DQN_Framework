# Changelog

Wszystkie istotne zmiany w projekcie DQN Framework są dokumentowane w tym pliku.

Format oparty na [Keep a Changelog](https://keepachangelog.com/pl/1.1.0/),
projekt stosuje [Semantic Versioning](https://semver.org/lang/pl/).

## [1.0.1] - Unreleased

### Zmienione

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
