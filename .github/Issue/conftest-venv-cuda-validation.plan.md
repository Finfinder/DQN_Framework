# conftest.py z auto-walidacją venv i CUDA - Plan Implementacji

## Szczegóły Zadania

| Pole | Wartość |
|---|---|
| Tytuł | conftest.py z auto-aktywacją venv w CI: fixture autouse do walidacji CUDA |
| Opis | Rozbudowa `tests/conftest.py` o autouse fixture walidujący środowisko testowe (aktywność venv, dostępność CUDA), banner diagnostyczny oraz marker `@pytest.mark.requires_cuda` z auto-skipem |
| Priorytet | Normalny |
| Powiązany Research | [conftest-venv-cuda-validation.research.md](conftest-venv-cuda-validation.research.md) |

## Proponowane Rozwiązanie

Rozszerzenie `tests/conftest.py` o trzy mechanizmy pytest:

1. **Hooki pytest** (`pytest_configure`, `pytest_sessionstart`, `pytest_collection_modifyitems`) — rejestracja markera, banner diagnostyczny, auto-skip testów GPU-only.
2. **Autouse session fixture** `validate_environment` — emituje `warnings.warn` gdy venv lub CUDA nie są dostępne lokalnie; w CI milczy.
3. **Marker `requires_cuda`** — rejestrowany w pytest i `pyproject.toml`; testy oznaczone tym markerem są automatycznie skipowane gdy CUDA niedostępne.

Rozwiązanie nie wprowadza nowych zależności — korzysta wyłącznie z `torch`, `pytest`, `os`, `sys`, `warnings` (już dostępne w projekcie). Nie modyfikuje CI workflow, istniejących fixture'ów, `Config` ani żadnych plików testowych.

```
pytest session start
        │
        ▼
  pytest_configure()        →  rejestracja markera requires_cuda
        │
        ▼
  pytest_sessionstart()     →  banner diagnostyczny (Python, PyTorch, CUDA, venv, CI)
        │
        ▼
  pytest_collection_modifyitems()  →  auto-skip @requires_cuda gdy brak CUDA
        │
        ▼
  validate_environment()    →  warnings.warn jeśli brak venv / CUDA (pominięte w CI)
        │
        ▼
  istniejące fixture'y      →  config, small_config, per_config (bez zmian)
        │
        ▼
  testy uruchamiają się
```

## Uzasadnienie Rozwiązania

### Wybrane podejście

Podejście hybrydowe: **hooki pytest** dla elementów lifecycle'u sesji (rejestracja markera, banner, modyfikacja kolekcji) + **autouse fixture** dla walidacji emitującej ostrzeżenia. Hooki są semantycznie poprawne dla operacji session-level, fixture zapewnia integrację z pytestowym systemem ostrzeżeń (`-W` flags, `filterwarnings`).

### Porównanie z alternatywami

| Kryterium | Hooki + autouse fixture (wybrane) | Tylko hooki (bez fixture) | Conftest plugin |
|---|---|---|---|
| Semantyczna poprawność | ✅ Hooki do lifecycle, fixture do walidacji | ⚠️ Ostrzeżenia w hookach są mniej standardowe | ✅ Pełna kontrola |
| Integracja z `-W` flags | ✅ `warnings.warn` filtrowalne przez pytest | ❌ `print` nie filtrowalny | ✅ |
| Złożoność | ✅ Niska — jeden plik | ✅ Niska | ⚠️ Wymaga osobnego pakietu |
| Kompatybilność z CI | ✅ CI detection przez `os.environ` | ✅ | ✅ |

### Dlaczego odrzucono alternatywy

- **Tylko hooki**: `warnings.warn` w `pytest_sessionstart` nie integruje się naturalnie z pytestowym systemem filtrowania ostrzeżeń na poziomie fixture'ów. Fixture `autouse` lepiej współgra z `-W error::UserWarning` i `filterwarnings` w `pyproject.toml`.
- **Conftest plugin**: Zbyt duża złożoność dla prostej walidacji środowiska. Wymaga rejestracji pluginu, osobnego pakietu — over-engineering.

## Model C4

### Diagram kontekstowy (Context)

Nie dotyczy — zadanie obejmuje zmianę w infrastrukturze testowej jednego projektu, bez interakcji z systemami zewnętrznymi.

### Diagram kontenerów (Container)

Nie dotyczy — zmiana dotyczy jednego pliku konfiguracyjnego testów.

### Diagram komponentów (Component)

Nie dotyczy — zadanie obejmuje pojedynczy komponent (`tests/conftest.py`).

## Rejestry Decyzji Architektonicznych (ADR)

### ADR-001: Ostrzeżenia zamiast failures przy braku CUDA

| Pole | Wartość |
|---|---|
| Status | Zaakceptowany |
| Data | 2026-04-07 |
| Kontekst | CI jest CPU-only (brak GPU w GitHub Actions). Fixture musi informować o braku CUDA bez blokowania testów. |

**Rozważane opcje**:
1. `warnings.warn` — ostrzeżenie, testy przechodzą dalej
2. `pytest.fail` — przerwanie sesji testowej
3. `pytest.skip` na poziomie sesji — skipnięcie wszystkich testów

**Decyzja**: Opcja 1 — `warnings.warn`

**Uzasadnienie**: Wszystkie istniejące testy działają na CPU (fallback `config.device`). Przerwanie lub skipnięcie złamałoby CI i uniemożliwiłoby weryfikację regresji na CPU.

**Konsekwencje**:
- ✅ CI pozostaje stabilne — brak CUDA nie blokuje testów
- ✅ Deweloper lokalnie widzi ostrzeżenie i może zareagować
- ⚠️ Ostrzeżenie może być zignorowane — ale to akceptowalne, bo testy i tak przechodzą

### ADR-002: Detekcja venv przez `sys.prefix`

| Pole | Wartość |
|---|---|
| Status | Zaakceptowany |
| Data | 2026-04-07 |
| Kontekst | Instrukcje projektu wymagają `.venv` dla CUDA PyTorch. Potrzebny mechanizm detekcji aktywnego venv. |

**Rozważane opcje**:
1. `sys.prefix != sys.base_prefix` — standardowa detekcja Pythona
2. Sprawdzanie `VIRTUAL_ENV` env var — ustawiane przez `activate`
3. Sprawdzanie ścieżki do interpretera

**Decyzja**: Opcja 1 — `sys.prefix != sys.base_prefix`

**Uzasadnienie**: Działa z `venv`, `virtualenv` i `conda`. Nie zależy od skryptu aktywacji (`VIRTUAL_ENV` nie jest ustawiane gdy Python uruchamiany bezpośrednio z `.venv/bin/python`).

**Konsekwencje**:
- ✅ Uniwersalne — działa z różnymi narzędziami virtualenv
- ✅ Nie wymaga dodatkowych zmiennych środowiskowych
- ⚠️ W CI (bez venv) zwróci False — obsłużone przez CI detection (`os.environ.get("CI")`)

## Analiza Aktualnej Implementacji

### Już Zaimplementowane
- Detekcja CUDA/CPU w `Config.__init__()` — `config/config.py` — `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Fixture `config` — `tests/conftest.py` — domyślna konfiguracja CartPole-v1
- Fixture `small_config` — `tests/conftest.py` — minimalna konfiguracja do szybkich testów
- Fixture `per_config` — `tests/conftest.py` — konfiguracja z PER enabled
- Helpery testowe — `tests/helpers.py` — `make_transitions()`, `fill_buffer()`
- CI workflow — `.github/workflows/ci.yml` — pipeline z CPU-only PyTorch, pytest + coverage
- Konfiguracja pytest — `pyproject.toml` — `pythonpath = ["."]`

### Do Modyfikacji
- `tests/conftest.py` — dodanie hooków pytest (3 funkcje), autouse fixture `validate_environment`, helperów `_is_ci()` i `_is_venv()`; istniejące fixture'y pozostają bez zmian
- `pyproject.toml` — dodanie sekcji `markers` w `[tool.pytest.ini_options]`

### Do Utworzenia
- Brak nowych plików — wszystkie zmiany w istniejących plikach

## Otwarte Pytania

| # | Pytanie | Odpowiedź | Status |
|---|----------|--------|--------|
| 1 | Zachowanie przy braku CUDA lokalnie | Ostrzegać (`warnings.warn`), nie failować | ✅ Rozwiązane |
| 2 | Banner diagnostyczny na starcie sesji | Tak — pełny banner z Python, PyTorch, CUDA, venv, CI | ✅ Rozwiązane |
| 3 | Marker `requires_cuda` teraz czy później | Teraz — infrastruktura pod przyszłe testy GPU-only | ✅ Rozwiązane |

## Plan Implementacji

### Faza 1: Konfiguracja pytest i marker `requires_cuda`

#### Zadanie 1.1 - [MODYFIKUJ] Dodanie markera `requires_cuda` w `pyproject.toml`
**Opis**: Rozszerzenie sekcji `[tool.pytest.ini_options]` o definicję markera `requires_cuda`. Zapobiega ostrzeżeniom pytest o niezarejestrowanych markerach przy przyszłym użyciu `@pytest.mark.requires_cuda`.

**Definicja Ukończenia (Definition of Done)**:
- [x] Sekcja `[tool.pytest.ini_options]` zawiera `markers = ["requires_cuda: mark test as requiring CUDA GPU"]`
- [x] Istniejąca konfiguracja `pythonpath = ["."]` pozostaje nienaruszona
- [x] `pytest --markers` wyświetla marker `requires_cuda` z opisem

#### Zadanie 1.2 - [MODYFIKUJ] Dodanie hooka `pytest_configure` w `tests/conftest.py`
**Opis**: Dodanie funkcji `pytest_configure(config)` rejestrującej marker `requires_cuda` programowo w conftest. Zapewnia redundancję z `pyproject.toml` (double-registration jest bezpieczna w pytest).

**Definicja Ukończenia (Definition of Done)**:
- [x] Funkcja `pytest_configure` istnieje w `tests/conftest.py`
- [x] Rejestruje marker `requires_cuda` przez `config.addinivalue_line("markers", ...)`
- [x] Istniejące fixture'y (`config`, `small_config`, `per_config`) nie są zmienione

### Faza 2: Banner diagnostyczny i auto-skip

#### Zadanie 2.1 - [MODYFIKUJ] Dodanie helpera `_is_ci()` i `_is_venv()` w `tests/conftest.py`
**Opis**: Dwie prywatne funkcje pomocnicze do detekcji środowiska CI i aktywnego venv. Wykorzystywane przez hook i fixture.

**Definicja Ukończenia (Definition of Done)**:
- [x] `_is_ci()` zwraca `True` gdy `os.environ.get("CI", "").lower() == "true"`
- [x] `_is_venv()` zwraca `True` gdy `sys.prefix != sys.base_prefix`
- [x] Funkcje umieszczone na początku pliku, przed hookami i fixture'ami

#### Zadanie 2.2 - [MODYFIKUJ] Dodanie hooka `pytest_sessionstart` w `tests/conftest.py`
**Opis**: Banner diagnostyczny drukowany na starcie sesji testowej. Zawiera wersje Python/PyTorch, status CUDA (+ nazwa karty), status venv i wykrycie CI.

**Definicja Ukończenia (Definition of Done)**:
- [x] Funkcja `pytest_sessionstart(session)` istnieje w `tests/conftest.py`
- [x] Drukuje banner z: Python version, PyTorch version, CUDA available (bool), CUDA device name (jeśli dostępne), venv active (bool), CI (bool)
- [x] Banner opakowany w linie `=` (60 znaków) dla czytelności
- [x] Banner widoczny w output zarówno lokalnie jak i w CI logs

#### Zadanie 2.3 - [MODYFIKUJ] Dodanie hooka `pytest_collection_modifyitems` w `tests/conftest.py`
**Opis**: Automatyczne skipowanie testów oznaczonych `@pytest.mark.requires_cuda` gdy CUDA nie jest dostępne. Iteruje po zebranych testach i dodaje marker `skip` z powodem.

**Definicja Ukończenia (Definition of Done)**:
- [x] Funkcja `pytest_collection_modifyitems(config, items)` istnieje w `tests/conftest.py`
- [x] Gdy `torch.cuda.is_available()` jest `False`, dodaje `pytest.mark.skip(reason="CUDA not available")` do testów z markerem `requires_cuda`
- [x] Gdy CUDA jest dostępne, żadne testy nie są skipowane
- [x] Istniejące testy (bez markera `requires_cuda`) nie są w żaden sposób dotknięte

### Faza 3: Autouse fixture i weryfikacja

#### Zadanie 3.1 - [MODYFIKUJ] Dodanie autouse fixture `validate_environment` w `tests/conftest.py`
**Opis**: Session-scoped autouse fixture emitujący ostrzeżenia `warnings.warn` gdy venv lub CUDA nie są dostępne lokalnie. W CI (wykryte przez `_is_ci()`) nie emituje ostrzeżeń.

**Definicja Ukończenia (Definition of Done)**:
- [x] Fixture `validate_environment` istnieje z `@pytest.fixture(autouse=True, scope="session")`
- [x] W środowisku CI (`_is_ci()` == `True`) fixture zwraca natychmiast bez ostrzeżeń
- [x] Gdy venv nieaktywne (`_is_venv()` == `False`) i nie w CI — emituje `warnings.warn` z komunikatem o `.venv`
- [x] Gdy CUDA niedostępne (`torch.cuda.is_available()` == `False`) i nie w CI — emituje `warnings.warn` z komunikatem o CUDA
- [x] Fixture nie modyfikuje `Config.device` ani żadnego globalnego stanu
- [x] Istniejące fixture'y `config`, `small_config`, `per_config` pozostają nienaruszone

#### Zadanie 3.2 - [UŻYJ PONOWNIE] Weryfikacja kompatybilności z CI
**Opis**: Uruchomienie istniejących testów lokalnie i weryfikacja, że CI pipeline nadal przechodzi bez zmian. Żadne istniejące testy nie powinny failować ani być skipowane.

**Definicja Ukończenia (Definition of Done)**:
- [x] `pytest tests/ -v` przechodzi lokalnie bez failures ani unexpected skips
- [x] Banner diagnostyczny jest widoczny w output
- [x] Ostrzeżenia `warnings.warn` wyświetlają się poprawnie (jeśli venv lub CUDA niedostępne)
- [x] Żaden istniejący test nie jest skipowany (brak markera `requires_cuda` na istniejących testach)

#### Zadanie 3.3 - [MODYFIKUJ] Aktualizacja `CHANGELOG.md`
**Opis**: Dodanie wpisu w sekcji `[Unreleased]` o nowych mechanizmach walidacji środowiska testowego.

**Definicja Ukończenia (Definition of Done)**:
- [x] Sekcja `[Unreleased]` zawiera wpis opisujący: autouse fixture `validate_environment`, banner diagnostyczny, marker `requires_cuda`
- [x] Wpis jest w kategorii `Added`

### Faza 4: Code review

#### Zadanie 4.1 - Code review przez agenta `code-reviewer`
**Opis**: Pełny przegląd zmian przez agenta `code-reviewer`.

**Definicja Ukończenia (Definition of Done)**:
- [ ] Code review przeprowadzony i zaakceptowany

## Aspekty Bezpieczeństwa

- **Brak ryzyk bezpieczeństwa** — zmiana dotyczy infrastruktury testowej, nie kodu produkcyjnego
- **Brak nowych zależności** — `os`, `sys`, `warnings` to moduły stdlib; `torch`, `pytest` już w projekcie
- **Brak przetwarzania danych użytkownika** — fixture czyta wyłącznie zmienne systemowe (`CI`, `sys.prefix`)

## Strategia Testowania

### Piramida testów

| Typ testu | Zakres | Szacowana liczba | Pokrycie |
|---|---|---|---|
| Jednostkowe | Helpery `_is_ci()`, `_is_venv()` | 4-6 | Podstawowe ścieżki True/False |
| Integracyjne | Autouse fixture + hooki z istniejącymi testami | 1 (pełny `pytest tests/ -v` run) | Weryfikacja braku regresji |
| E2E | Nie dotyczy | 0 | — |

### Podejście do testowania

- [ ] Testy regresji — uruchomienie pełnego `pytest tests/ -v` po zmianach, weryfikacja braku failures/unexpected skips
- [ ] Ręczna weryfikacja banneru — wizualna inspekcja output z banneru diagnostycznego
- [ ] Opcjonalnie: test `_is_ci()` z podmianą env var `CI` za pomocą `monkeypatch`
- [ ] Opcjonalnie: test `_is_venv()` z podmianą `sys.prefix` i `sys.base_prefix` za pomocą `monkeypatch`
- [ ] Weryfikacja auto-skipu markera `requires_cuda` — tymczasowy test z `@pytest.mark.requires_cuda` na maszynie bez GPU

### Testy wydajnościowe

Nie dotyczy.

### Testy dostępności

Nie dotyczy.

### Testy architektoniczne

Nie dotyczy.

### Testy mutacyjne

Nie dotyczy.

## Zapewnienie Jakości

- [x] `pytest tests/ -v` przechodzi bez failures zarówno lokalnie (z venv + CUDA) jak i bez CUDA
- [x] Banner diagnostyczny wyświetla poprawne informacje o środowisku
- [x] Istniejące 3 fixture'y (`config`, `small_config`, `per_config`) nie zostały zmodyfikowane
- [x] `pytest --markers` wyświetla `requires_cuda` z opisem
- [x] Ostrzeżenia pojawiają się gdy venv nieaktywne i nie w CI
- [x] Ostrzeżenia pojawiają się gdy CUDA niedostępne i nie w CI
- [x] W środowisku CI (z `CI=true`) brak ostrzeżeń o venv/CUDA
- [x] CI workflow (`.github/workflows/ci.yml`) nie wymaga żadnych zmian i przechodzi
- [x] `CHANGELOG.md` zawiera wpis w `[Unreleased]`

## Usprawnienia (Poza Zakresem)

- **Fixture `device` (session-scoped)**: Dedykowany fixture zwracający `torch.device(...)` — pozwoliłby testom deklarować zależność od device zamiast tworzyć go przez `Config`. Odłożone — obecne testy korzystają z `config.device`.
- **`pytest-timeout` dla testów GPU**: Plugin wymuszający timeout na testach GPU, aby zapobiec zawieszeniom przy problemach z CUDA. Odłożone — brak testów GPU-only.
- **Warunkowe category do `filterwarnings`**: Konfiguracja `pyproject.toml` umożliwiająca traktowanie ostrzeżeń o braku venv jako errors (`-W error::UserWarning`). Odłożone — wymaga dokładnego filtrowania, aby nie łapać innych UserWarnings.

## Changelog

- 2026-04-07: Utworzenie planu implementacji na podstawie researchu
- 2026-04-07: Implementacja zakończona — 101/101 testów przechodzi
- 2026-04-07: Code review przeprowadzony — APPROVE. Poprawiono `stacklevel=2` → `stacklevel=1` w `warnings.warn` (Issue #2 z review — `stacklevel=2` wskazywał na pytest internals zamiast linię w conftest.py). Duplikacja rejestracji markera (Issue #1) pozostawiona zgodnie z planem jako świadoma redundancja.

## Code Review Findings

**Przeprowadzono**: 2026-04-07 | **Wynik**: APPROVE

| Issue | Severity | Rozwiązanie |
|---|---|---|
| Podwójna rejestracja markera `requires_cuda` (`pyproject.toml` + `pytest_configure`) daje duplikat w `pytest --markers` | Minor | Pozostawione — plan świadomie akceptuje redundancję jako odporność na brak `pyproject.toml` |
| `stacklevel=2` w `warnings.warn` wewnątrz session-scoped fixture wskazywał na pytest internals zamiast linię w `conftest.py` | Minor | Naprawione: zmieniono na `stacklevel=1` |

Wszystkie kryteria akceptacji spełnione. 101 testów przechodzi. Konwencje projektu zachowane.
