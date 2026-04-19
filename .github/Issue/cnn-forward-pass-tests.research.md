# Testy forward pass CNN (bez GPU) — Wynik analizy

## Szczegóły zadania

| Pole | Wartość |
|---|---|
| Jira ID | Brak |
| Tytuł | Dodaj testy forward pass CNN (bez GPU) dla `models/cnn_dqn_network.py` |
| Opis | Utworzenie zestawu testów jednostkowych weryfikujących forward pass sieci `CNNDQN` — tworzenie instancji, kształt wyjścia, warianty standard/dueling, integracja z factory `create_network()`. Testy muszą działać na CPU (kompatybilność z CI). |
| Priorytet | Średni |
| Zgłaszający | — |
| Data utworzenia | 2026-04-19 |
| Termin realizacji | — |
| Etykiety | testy, CNN, pokrycie kodu |
| Szacowany nakład pracy | S (mały) |
| Złożoność analizy rozwiązań | Nie dotyczy |

## Wpływ biznesowy

Pokrycie testami modułu `cnn_dqn_network.py` wynosi obecnie **11.32%** (coverage.xml). Dodanie testów forward pass:
- Zabezpieczy przed regresją przy zmianach architektury CNN (np. nowe warstwy konwolucyjne, zmiana dueling logic)
- Zwiększy pewność przy refaktoringu i dodawaniu nowych środowisk Atari
- Poprawi metrykę jakości kodu w CI (coverage gate)
- Testy CPU-only gwarantują wykonanie w GitHub Actions bez GPU

## Zebrane informacje

### Baza wiedzy i narzędzia do zarządzania zadaniami

Brak powiązanych zadań w Jira/Confluence. Zadanie pochodzi z bezpośredniego polecenia użytkownika.

### Baza kodu

#### Testowany moduł: `models/cnn_dqn_network.py`

Klasa `CNNDQN(nn.Module)` — sieć konwolucyjna DQN z opcjonalnym wariantem dueling:

- **`__init__(self, input_shape, action_dim, conv_layers=None, hidden_dim=512, dueling=False)`**
  - `input_shape`: krotka `(channels, height, width)` — np. `(4, 84, 84)` dla 4-frame stack
  - `conv_layers`: lista krotek `(out_channels, kernel_size, stride)` — domyślnie `[(32, 8, 4), (64, 4, 2), (64, 3, 1)]`
  - `hidden_dim`: rozmiar warstwy fully-connected — domyślnie `512`
  - `dueling`: `True` → osobne głowy value + advantage; `False` → jedna głowa Q-values
  - Flatten size obliczany dynamicznie przez dummy forward pass przez conv trunk
  - Inicjalizacja wag: `orthogonal_` z `relu_gain` dla conv/fc, `gain=1.0` dla głów output

- **`forward(self, x)`**
  - Standard: `conv_trunk → fc → q_head` → `(batch, action_dim)`
  - Dueling: `conv_trunk → fc → value_head + (advantage_head - mean(advantage))` → `(batch, action_dim)`

#### Factory: `models/dqn_network.py` → `create_network(config, state_shape, action_dim)`

- Gdy `config.network_type == "cnn"` → tworzy `CNNDQN` z parametrami z configa
- Parametry mapowane: `conv_layers`, `cnn_hidden_dim` → `hidden_dim`, `use_dueling` → `dueling`

#### Konfiguracja CNN: `config/config.py` → `Config`

- `ALE/Pong-v5` to jedyne środowisko z `network_type: "cnn"`:
  - `conv_layers: [(32, 8, 4), (64, 4, 2), (64, 3, 1)]`
  - `cnn_hidden_dim: 1024`, `frame_stack: 4`, `frame_size: [84, 84]`
  - `is_atari: True`, `use_dueling: True`

#### Istniejące testy i pokrycie

| Plik | Pokrycie | Uwagi |
|---|---|---|
| `models/cnn_dqn_network.py` | **11.32%** | Tylko import + deklaracja klasy. Żaden test nie tworzy instancji `CNNDQN` ani nie wywołuje `forward()` |
| `models/dqn_network.py` | Wyższe | `create_network()` testowany pośrednio przez `test_dqn_agent.py`, ale tylko wariant MLP |
| Brak pliku `tests/test_cnn_dqn_network.py` | — | Plik testowy nie istnieje |

Linie z `hits=0` w `coverage.xml` dla `cnn_dqn_network.py` obejmują: cały `__init__` (L7–L43), cały `_init_weights` (L45–L63), cały `forward` (L65–L76) — praktycznie cały kod poza deklaracjami.

#### Konwencje testowe (z istniejących testów)

- Plik testowy: `tests/test_<moduł>.py`
- Grupowanie: `class TestXxx:`
- Fixture'y w `conftest.py`: `config`, `small_config`, `per_config`
- Helper: `tests/helpers.py` — `make_transitions()`, `fill_buffer()`
- Brak type annotations i docstringów w testach
- `pytest` z `pythonpath = ["."]` w `pyproject.toml`
- Marker `requires_cuda` dla testów wymagających GPU — testy CNN forward pass NIE powinny go używać

#### CI: `.github/workflows/ci.yml`

- Instalacja CPU-only (`torch==2.5.1` bez CUDA)
- `pytest tests/ --cov=... -q` — pokrywa folder `models`
- Brak GPU w CI — testy MUSZĄ działać na CPU
- Brak `gymnasium[atari]` w CI — nie można tworzyć środowisk Atari, ale `CNNDQN` nie wymaga środowiska

### Powiązane linki

- `models/cnn_dqn_network.py` — testowany moduł
- `models/dqn_network.py` — factory `create_network()`
- `config/config.py` — konfiguracja `ALE/Pong-v5` z parametrami CNN
- `tests/conftest.py` — istniejące fixture'y
- `tests/test_dqn_agent.py` — wzorcowy plik testowy (dla MLP)
- `.github/workflows/ci.yml` — pipeline CI
- `coverage.xml` — raport pokrycia (line-rate 0.1132 dla CNN)

### Analiza rozwiązań

Nie przeprowadzono — wymagania jednoznaczne, technologia wybrana (pytest + PyTorch CPU).

### Powiązane wykresy i diagramy

Brak.

## Aktualny stan implementacji

### Istniejące komponenty

- `models/cnn_dqn_network.py` — `CNNDQN` — wymaga testów (brak pokrycia forward pass)
- `models/dqn_network.py` — `create_network()` — można ponownie użyć (factory testowana pośrednio dla MLP, wymaga testu dla CNN)
- `tests/conftest.py` — fixture'y `config`, `small_config`, `per_config` — wymagają rozszerzenia o fixture CNN (np. `cnn_config`)
- `tests/helpers.py` — helpery `make_transitions()`, `fill_buffer()` — można ponownie użyć (ale nie są niezbędne dla forward pass)
- `config/config.py` — `Config("ALE/Pong-v5")` — można ponownie użyć do testów integracyjnych z factory

### Kluczowe pliki i katalogi

- `models/cnn_dqn_network.py` — moduł do przetestowania
- `models/dqn_network.py` — factory `create_network()`, import `CNNDQN`
- `tests/` — katalog na nowy plik testowy `test_cnn_dqn_network.py`
- `tests/conftest.py` — miejsce na nową fixture `cnn_config`
- `config/config.py` — `Config` z `ENV_CONFIG["ALE/Pong-v5"]` dla parametrów CNN
- `.github/workflows/ci.yml` — CI automatycznie wykryje nowy plik testowy (glob `tests/`)

## Analiza luk

### Pytanie 1
#### Czy testy powinny obejmować również `_init_weights()` (weryfikacja wartości wag po inicjalizacji)?
Nie — zakres zadania to **forward pass**. Testowanie `_init_weights()` to osobne zadanie. Forward pass pośrednio weryfikuje, że inicjalizacja nie powoduje NaN/Inf w outputach.

### Pytanie 2
#### Czy potrzebna jest nowa fixture `cnn_config` w `conftest.py` czy wystarczy tworzenie `Config("ALE/Pong-v5")` inline?
Rekomendacja: Dodać fixture `cnn_config` w `conftest.py` z minimalnymi parametrami (mały `cnn_hidden_dim`, np. 32–64) dla szybkości testów. `Config("ALE/Pong-v5")` ma `cnn_hidden_dim=1024`, co spowalnia testy. Fixture powinna nadpisywać:
- `cnn_hidden_dim` → mała wartość (np. 64)
- Opcjonalnie mniejsze `conv_layers` dla jeszcze szybszych testów

### Pytanie 3
#### Jakie scenariusze forward pass powinny być pokryte?
Na podstawie analizy kodu `CNNDQN.forward()` — wymagane scenariusze:

| Scenariusz | Ścieżka kodu | Priorytet |
|---|---|---|
| Standard forward (dueling=False) | `q_head(fc_out)` | Krytyczny |
| Dueling forward (dueling=True) | `value + (advantage - mean)` | Krytyczny |
| Output shape: `(batch, action_dim)` | oba warianty | Krytyczny |
| Batch size = 1 | `mean(dim=1, keepdim=True)` z single sample | Wysoki |
| Batch size > 1 | normalna ścieżka | Wysoki |
| Custom conv_layers | inna architektura conv trunk | Średni |
| Custom hidden_dim | inna szerokość FC | Średni |
| Output dtype: float | `torch.float32` | Średni |
| Deterministyczność (eval mode) | ten sam input → ten sam output | Średni |
| Factory `create_network()` z CNN config | integracja z `Config` | Wysoki |
| Output requires_grad (train mode) | backward compatibility | Niski |

### Pytanie 4
#### Czy mniejszy `frame_size` (np. 32x32 zamiast 84x84) jest akceptowalny w testach?
Tak — testy forward pass weryfikują kształty i logikę, nie jakość predykcji. Mniejszy input = szybszy test. Należy jednak zachować przynajmniej jeden test z domyślnymi parametrami (84x84, domyślne conv_layers), aby upewnić się, że konfiguracja produkcyjna nie powoduje błędów wymiarów.

### Pytanie 5
#### Czy testy powinny weryfikować właściwość dueling (advantage mean ≈ 0)?
Tak — to kluczowa cecha architektury dueling DQN. Test powinien sprawdzić, że po forward pass z dueling=True, wewnętrzna normalizacja advantage (odejmowanie mean) działa poprawnie. Można to zweryfikować pośrednio: output powinien mieć poprawny kształt i nie zawierać NaN/Inf.
