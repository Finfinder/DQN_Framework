# conftest.py z auto-walidacją venv i CUDA - Wynik analizy

## Szczegóły zadania

| Pole | Wartość |
|---|---|
| Jira ID | — |
| Tytuł | conftest.py z auto-aktywacją venv w CI: Rozważyć fixture autouse do walidacji CUDA |
| Opis | Rozbudowa `tests/conftest.py` o autouse fixture walidujący środowisko testowe (aktywność venv, dostępność CUDA) z bannerem diagnostycznym oraz marker `@pytest.mark.requires_cuda` z auto-skipem |
| Priorytet | Normalny |
| Zgłaszający | — |
| Data utworzenia | 2026-04-07 |
| Termin realizacji | — |
| Etykiety | testing, DX, infrastructure |
| Szacowany nakład pracy | S (kilka godzin) |
| Złożoność analizy rozwiązań | Nie dotyczy |

## Wpływ biznesowy

Zadanie poprawia Developer Experience (DX) i niezawodność pipeline'u testowego:
- **Eliminuje ciche fallbacki** — deweloper dowie się natychmiast, że testy biegną na CPU zamiast GPU, zamiast odkrywać to po godzinach wolnego treningu.
- **Redukuje czas debugowania** — banner diagnostyczny na starcie sesji testowej podaje kluczowe info o środowisku (Python, PyTorch, CUDA, venv).
- **Przygotowuje infrastrukturę** pod przyszłe testy GPU-only (marker `requires_cuda`), bez blokowania obecnego CI.
- **Zabezpiecza CI** — walidacja wie, że CI jest CPU-only i nie blokuje testów.

## Zebrane informacje

### Baza wiedzy i narzędzia do zarządzania zadaniami

Brak podłączonych narzędzi Jira/Confluence. Kontekst dostarczony bezpośrednio w opisie zadania.

### Baza kodu

#### Aktualny `tests/conftest.py`

Plik zawiera 3 fixture'y konfiguracyjne — żaden nie jest `autouse`:

```python
@pytest.fixture
def config():             # Config("CartPole-v1") — domyślny
@pytest.fixture
def small_config():       # Config z minimalnym memory_size, batch_size, hidden_layers
@pytest.fixture
def per_config():         # Config z PER enabled
```

Brak jakiejkolwiek walidacji środowiska, informacji o CUDA, venv, ani markerów testowych.

#### CI workflow (`.github/workflows/ci.yml`)

- Runner: `ubuntu-latest`, Python 3.11
- **CPU-only PyTorch** — `pip install torch==2.5.1` (bez `+cu121`)
- **Brak venv** — pakiety instalowane globalnie w runnerze GitHub Actions
- Testy: `pytest tests/ --cov=... -q`
- Smoke testy CLI: `train.py --version`, `evaluate.py --version`, `play.py --version`
- Smoke test Configu + środowiska (tworzenie env, step, close)

#### Konfiguracja pytest (`pyproject.toml`)

Minimalna — tylko `pythonpath = ["."]`. Brak definicji markerów, pluginów, filtrów ostrzeżeń.

#### Detekcja device w `Config.__init__()`

```python
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Cichy fallback na CPU — brak ostrzeżeń, logów, ani sygnalizacji.

#### `requirements.txt` — CUDA toolchain

```
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
torchvision==0.20.1+cu121
```

Wymaga `--extra-index-url https://download.pytorch.org/whl/cu121` oraz aktywnego `.venv`.

#### Testy używające `torch` i `config.device`

| Plik testowy | Używa torch? | Transfery .to(device) | Fixture |
|---|---|---|---|
| `test_dqn_agent.py` | ✅ Tak | ✅ `create_network(...).to(config.device)` | `small_config`, `per_config` |
| `test_evaluate.py` | ✅ Tak | ✅ `create_network(...).to(config.device)` | `small_config` |
| `test_config.py` | ✅ Tak (import) | ❌ Testuje `Config`, nie tensory | brak |
| `test_replay_buffer.py` | ❌ Nie | ❌ Tylko numpy | `config` (pośrednio) |
| `test_training.py` | ❌ Nie | ❌ Mocki | `small_config`, `per_config` |
| `test_wrappers.py` | ❌ Nie | ❌ Środowisko | brak |
| `test_analyze.py` | ❌ Nie | ❌ Pandas/CSV | brak |

**Wniosek**: Żaden test nie jest stricte GPU-only — wszystkie działają na CPU dzięki fallbackowi `config.device`. Marker `requires_cuda` jest potrzebny jako infrastruktura pod przyszłe testy.

#### Helpers (`tests/helpers.py`)

Zawiera `make_transitions()` i `fill_buffer()` — operacje CPU-only (numpy). Nie wymaga zmian.

### Powiązane linki

- Instrukcje projektu: `.github/instructions/dqn-framework.instructions.md` — sekcja "Środowisko wirtualne (KRYTYCZNE)" dokumentuje wymóg `.venv`
- pytest fixtures docs: https://docs.pytest.org/en/stable/how-to/fixtures.html#autouse-fixtures
- pytest markers docs: https://docs.pytest.org/en/stable/how-to/mark.html

### Analiza rozwiązań

Nie przeprowadzono — wymagania jednoznaczne, technologia wybrana.

### Powiązane wykresy i diagramy

```
Przepływ walidacji środowiska (session start):

pytest session start
        │
        ▼
┌──────────────────────┐
│ pytest_sessionstart   │  ← Hook: banner diagnostyczny
│ (print env info)      │    Python, PyTorch, CUDA, venv, CI
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ pytest_configure      │  ← Hook: rejestracja markera requires_cuda
│ (register markers)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ pytest_collection_    │  ← Hook: auto-skip testów @requires_cuda
│ modifyitems           │    gdy CUDA niedostępne
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────┐
│ validate_environment()    │  ← Fixture autouse (session)
│ - CI? → skip validation   │    Ostrzeżenia: brak venv, brak CUDA
│ - venv? → warn if missing │
│ - CUDA? → warn if missing │
└──────────┬───────────────┘
           │
           ▼
     Testy uruchamiają się
```

## Aktualny stan implementacji

### Istniejące komponenty

- `tests/conftest.py` — `e:\AI_WORKSPACE\Moje projekty\DQN_Framework\tests\conftest.py` — wymaga rozszerzenia (dodać fixture autouse + hooki pytest)
- `pyproject.toml` — `e:\AI_WORKSPACE\Moje projekty\DQN_Framework\pyproject.toml` — wymaga rozszerzenia (dodać definicję markera `requires_cuda` w `[tool.pytest.ini_options]`)
- `config/config.py` — `e:\AI_WORKSPACE\Moje projekty\DQN_Framework\config\config.py` — można ponownie użyć (detekcja CUDA)
- `.github/workflows/ci.yml` — `e:\AI_WORKSPACE\Moje projekty\DQN_Framework\.github\workflows\ci.yml` — bez zmian (CPU-only, fixture musi to obsłużyć)
- `tests/helpers.py` — `e:\AI_WORKSPACE\Moje projekty\DQN_Framework\tests\helpers.py` — bez zmian
- `tests/test_dqn_agent.py` — `e:\AI_WORKSPACE\Moje projekty\DQN_Framework\tests\test_dqn_agent.py` — bez zmian (korzysta z config.device)
- `tests/test_evaluate.py` — `e:\AI_WORKSPACE\Moje projekty\DQN_Framework\tests\test_evaluate.py` — bez zmian

### Kluczowe pliki i katalogi

- `tests/conftest.py` — główny plik do modyfikacji; dodanie autouse fixture + hooków pytest
- `pyproject.toml` — dodanie pytest markers (`requires_cuda`) i opcjonalnie `filterwarnings`
- `tests/` — cały katalog testów; żaden plik testowy nie wymaga zmian (autouse jest automatyczny)
- `.github/workflows/ci.yml` — punkt odniesienia dla zachowania CI (musi nadal przechodzić bez zmian)

## Analiza luk

### Pytanie 1
#### Gdy CUDA jest niedostępne lokalnie (brak .venv lub maszyna bez GPU), fixture powinien:
Tylko ostrzegać (`warnings.warn`). Testy przejdą na CPU, ale deweloper zobaczy ostrzeżenie. Nie blokujemy sesji testowej — Config i tak robi fallback na CPU.

### Pytanie 2
#### Czy fixture powinien drukować info o środowisku (banner diagnostyczny) na starcie sesji testowej?
Tak — pełny banner diagnostyczny. Zawierający: wersję Pythona, wersję PyTorch, dostępność CUDA (+ nazwa karty jeśli jest), status venv, wykrycie CI. Ułatwi debugowanie problemów ze środowiskiem, szczególnie w CI logs.

### Pytanie 3
#### Czy dodać teraz marker @pytest.mark.requires_cuda z auto-skipem, czy odłożyć?
Dodać teraz. Przygotuje infrastrukturę pod przyszłe testy GPU-only. Wymaga:
- `pytest_configure` hook do rejestracji markera
- `pytest_collection_modifyitems` hook do auto-skipu oznaczonych testów gdy CUDA niedostępne
- Wpis `markers` w `pyproject.toml` sekcji `[tool.pytest.ini_options]`

## Wymagania szczegółowe

### WYM-1: Autouse session fixture `validate_environment`

- **Scope**: `session` (jedno sprawdzenie per sesja testowa)
- **Autouse**: `True` (uruchamia się automatycznie, bez deklaracji w testach)
- **Zachowanie w CI** (`os.environ.get("CI")` == `"true"`):
  - Nie emituje ostrzeżeń o brakującym CUDA ani venv
  - Pozwala testom przejść normalnie
- **Zachowanie lokalne**:
  - Jeśli `sys.prefix == sys.base_prefix` → `warnings.warn("Running outside .venv!...")`
  - Jeśli `not torch.cuda.is_available()` → `warnings.warn("CUDA is not available...")`
- **Nie modyfikuje** `Config.device` ani żadnego stanu globalnego

### WYM-2: Banner diagnostyczny (`pytest_sessionstart` hook)

Wypisuje na stdout na początku sesji testowej:
```
============================================================
DQN Framework Test Session
Python: 3.x.x
PyTorch: 2.5.1+cu121
CUDA available: True
CUDA device: NVIDIA GeForce RTX ...
venv active: True
CI: false
============================================================
```

Pola:
- `Python` — `sys.version` (skrócone)
- `PyTorch` — `torch.__version__`
- `CUDA available` — `torch.cuda.is_available()`
- `CUDA device` — `torch.cuda.get_device_name(0)` (tylko gdy CUDA dostępne)
- `venv active` — `sys.prefix != sys.base_prefix`
- `CI` — `os.environ.get("CI", "false")`

### WYM-3: Marker `@pytest.mark.requires_cuda`

- Rejestracja: `pytest_configure` hook w `conftest.py`
- Auto-skip: `pytest_collection_modifyitems` hook — gdy `not torch.cuda.is_available()`, dodaje `pytest.mark.skip(reason="CUDA not available")` do testów z markerem `requires_cuda`
- Definicja w `pyproject.toml`:
  ```toml
  [tool.pytest.ini_options]
  markers = [
      "requires_cuda: mark test as requiring CUDA GPU",
  ]
  ```
- Żadne istniejące testy nie dostają tego markera — jest infrastrukturą pod przyszłe testy

### WYM-4: Zachowanie istniejących fixture'ów

Trzy istniejące fixture'y (`config`, `small_config`, `per_config`) muszą pozostać bez zmian. Nowy kod dodaje się POWYŻEJ nich (hooki) i PONIŻEJ nich (autouse fixture), lub odwrotnie — ważne, aby nie modyfikować istniejących definicji.

### WYM-5: Kompatybilność z CI

CI workflow (`.github/workflows/ci.yml`) **NIE wymaga żadnych zmian**. Fixture musi:
- Nie failować gdy CUDA niedostępne
- Nie failować gdy venv nieaktywne
- Nie dodawać nowych zależności (wszystko z `torch`, `pytest`, `os`, `sys`, `warnings`)
- Nie skipować istniejących testów (żaden nie ma markera `requires_cuda`)

### Ograniczenia i decyzje

| Decyzja | Uzasadnienie |
|---|---|
| `warnings.warn` zamiast `pytest.fail` | CI jest CPU-only — fail złamałby pipeline |
| `scope="session"` | Walidacja raz per sesja, nie per test — wydajność |
| Hook `pytest_sessionstart` zamiast `print` w fixture | Semantycznie poprawne — lifecycle session, nie dostarczanie wartości |
| Detekcja CI przez `os.environ.get("CI")` | GitHub Actions ustawia `CI=true` automatycznie |
| Detekcja venv przez `sys.prefix != sys.base_prefix` | Działa z venv, virtualenv, conda |
| Marker bez przypisania do testów | Infrastruktura pod przyszłość — obecne testy działają na CPU |
