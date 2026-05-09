# Testy jednostkowe utils/analyze.py — Wynik analizy

## Szczegóły zadania

| Pole | Wartość |
|---|---|
| Jira ID | Nie dotyczy |
| Tytuł | Dodanie testów jednostkowych dla utils/analyze.py z mockami |
| Opis | Dodać testy dla `utils/analyze.py` z mockami (bez realnych CSV) — podniesie overall coverage |
| Priorytet | Średni |
| Zgłaszający | — |
| Data utworzenia | 2026-04-19 |
| Termin realizacji | — |
| Etykiety | testy, coverage |
| Szacowany nakład pracy | S (mały — jeden plik testowy, znane wzorce) |
| Złożoność analizy rozwiązań | Nie dotyczy |

## Wpływ biznesowy

Podniesienie pokrycia kodu testami z obecnych ~51.7% do ~72% (szacunkowo). Moduł `analyze.py` ma obecnie najniższy `line-rate` w projekcie (18.75%), co obniża ogólną jakość i utrudnia bezpieczne refaktoryzacje. Pokrycie testami zabezpieczy logikę diagnostyczną i raportującą przed regresjami.

## Zebrane informacje

### Baza wiedzy i narzędzia do zarządzania zadaniami

Nie dotyczy — zadanie podane bezpośrednio w opisie, bez powiązania z Jira/Confluence.

### Baza kodu

#### Analizowany moduł: `utils/analyze.py`

Plik zawiera **160 linii kodu** z **18.75% pokryciem** (dane z `coverage.xml`). Jedynie importy i sygnatury funkcji są pokryte (hits=1 na definicjach). Całe ciała funkcji mają hits=0.

**Istniejące testy** (`tests/test_analyze.py`):
- Klasa `TestParseRunFilename` — 8 testów pokrywających wyłącznie `_parse_run_filename()`
- Żadna inna funkcja z `analyze.py` nie jest testowana

**Funkcje publiczne bez pokrycia** (14 funkcji):

| Funkcja | Linia | Typ | Zależności do mockowania |
|---|---|---|---|
| `list_runs()` | 46 | I/O | `METRICS_DIR.glob()`, `_parse_run_filename` |
| `load_run()` | 81 | I/O | `pd.read_csv`, `Path` |
| `load_latest()` | 89 | I/O | `list_runs`, `load_run` |
| `compare_runs()` | 109 | I/O | `list_runs`, `load_run` |
| `run_summary()` | 156 | Raport | `compare_runs` |
| `_diagnose_trend()` | 176 | Logika | Brak (czysta logika, DataFrame in-memory) |
| `_diagnose_epsilon()` | 196 | Logika | Brak (czysta logika, DataFrame in-memory) |
| `_diagnose_td_error()` | 204 | Logika | Brak (czysta logika, DataFrame in-memory) |
| `_diagnose_eval_vs_train()` | 217 | Logika + I/O | `load_latest` |
| `diagnose()` | 249 | Orkiestracja | `load_latest` |
| `build_summary_report()` | 267 | Raport | `compare_runs` |
| `export_summary_report()` | 291 | Raport + I/O | `build_summary_report`, `df.to_csv` |
| `parse_args()` | 306 | CLI | `sys.argv` |
| `main()` | 374 | CLI | `parse_args`, `list_runs`, inne |
| `_print_env_list()` | 340 | CLI | Brak (drukuje) |
| `_print_train_eval_results()` | 351 | CLI | `compare_runs`, `diagnose` |

#### Strategia mockowania — podział na grupy

**Grupa 1 — Funkcje I/O** (`list_runs`, `load_run`, `load_latest`, `compare_runs`):
- Mockować `METRICS_DIR.glob()` zwracając fake `Path` obiekty z kontrolowanymi `.stem` i `.name`
- Mockować `pd.read_csv` zwracając `pd.DataFrame` tworzony in-memory
- NIE tworzyć realnych plików CSV na dysku

**Grupa 2 — Funkcje diagnostyczne** (`_diagnose_trend`, `_diagnose_epsilon`, `_diagnose_td_error`):
- Czysta logika — tworzyć `pd.DataFrame` in-memory z odpowiednimi kolumnami
- BEZ mocków — to testy czystej logiki (parsowanie DataFrame → lista obserwacji)
- Scenariusze: każda ścieżka warunkowa (np. BRAK UCZENIA, WCZESNE PLATEAU, DOBRY TREND)

**Grupa 3 — `_diagnose_eval_vs_train`**:
- Tworzyć `df_train` in-memory
- Mockować `load_latest()` — zwraca kontrolowany DataFrame eval lub `(None, None)`
- Scenariusze: eval << train, eval > train, wysoki std, brak danych eval, fallback na standalone_eval

**Grupa 4 — Funkcje raportujące** (`diagnose`, `run_summary`, `build_summary_report`, `export_summary_report`):
- Mockować zależności niższego poziomu (`load_latest`, `compare_runs`)
- Dla `export_summary_report` użyć `tmp_path` fixture do weryfikacji zapisu pliku

**Grupa 5 — CLI** (`parse_args`, `_print_env_list`, `_print_train_eval_results`, `main`):
- `parse_args`: mockować `sys.argv`
- `_print_env_list` / `_print_train_eval_results`: użyć `capsys` fixture do weryfikacji stdout
- `main`: mockować `parse_args` + `list_runs` + inne zależności; minimalne testy

#### Coverage — wpływ ilościowy

| Metryka | Przed | Po (szacunkowo) |
|---|---|---|
| `analyze.py` line-rate | 18.75% (30/160) | ~85-90% (~140/160) |
| Overall line-rate | 51.66% (326/631) | ~72% (~456/631) |
| Przyrost linii pokrytych | — | +~130 linii |

#### Wzorce testowania w projekcie

Na podstawie istniejących testów (`test_config.py`, `test_evaluate.py`, `test_replay_buffer.py`):

- **Grupowanie klasowe**: `class TestXxx:` — każda klasa grupuje powiązane testy
- **Mockowanie**: `unittest.mock.MagicMock`, `patch` z `unittest.mock`
- **Fixtures**: `conftest.py` definiuje współdzielone fixtures (np. `small_config`)
- **Helpers**: `tests/helpers.py` — współdzielone funkcje pomocnicze
- **Brak type annotations i docstringów** — zgodnie z konwencją projektu
- **Importy**: bezpośrednie z modułu (`from utils.analyze import ...`)
- **Nazewnictwo testów**: `test_<co_testujemy>` w snake_case

#### Kluczowe pola DataFrame używane w analyze.py

**Kolumny treningowe** (train CSV): `reward`, `avg100`, `epsilon`, `td_error_mean` (opcjonalna)

**Kolumny ewaluacyjne** (eval CSV): `mean_reward`, `std_reward` (opcjonalna)

**Kolumny metadata runu** (z `list_runs`): `file`, `path`, `env`, `model`, `timestamp`, `type`

### Powiązane linki

- `coverage.xml` — aktualne dane pokrycia kodu
- Konwencje projektu: `.github/instructions/dqn-framework.instructions.md`

### Analiza rozwiązań

Nie przeprowadzono — wymagania jednoznaczne, technologia wybrana. Testy oparte na `pytest` + `unittest.mock`, zgodnie z istniejącymi wzorcami projektu.

### Powiązane wykresy i diagramy

Nie dotyczy.

## Aktualny stan implementacji

### Istniejące komponenty

- `utils/analyze.py` — `utils/analyze.py` — moduł do analizy metryk CSV: **wymaga pokrycia testami**
- `tests/test_analyze.py` — `tests/test_analyze.py` — istniejący plik testowy z 8 testami `_parse_run_filename`: **wymaga rozszerzenia**
- `tests/conftest.py` — `tests/conftest.py` — fixtures współdzielone: **można ponownie użyć** (fixture `small_config` przydatny jeśli potrzebny)
- `tests/helpers.py` — `tests/helpers.py` — helpery testowe: **można ponownie użyć** (wzorzec)

### Kluczowe pliki i katalogi

- `utils/analyze.py` — moduł docelowy do pokrycia testami
- `tests/test_analyze.py` — plik testowy do rozszerzenia
- `coverage.xml` — aktualne metryki pokrycia (line-rate 0.1875 dla analyze.py)
- `pyproject.toml` — konfiguracja pytest (pythonpath, markers)
- `tests/conftest.py` — konfiguracja sesji testowej, fixtures

## Analiza luk

Brak luk w wymaganiach — zadanie jest jednoznaczne i kompletne. Poniżej pytania analityczne z ustalonymi odpowiedziami:

### Pytanie 1
#### Czy rozszerzać istniejący `test_analyze.py` czy tworzyć nowy plik?
Rozszerzać istniejący plik `tests/test_analyze.py`. Istniejąca klasa `TestParseRunFilename` zostaje — nowe klasy testowe dodawane poniżej. Zgodne z konwencją "jeden plik testowy per moduł".

### Pytanie 2
#### Jakie scenariusze diagnostyczne wymagają pokrycia?
Każda funkcja `_diagnose_*` ma 2-3 ścieżki warunkowe:
- `_diagnose_trend`: BRAK UCZENIA (`improve_total < 0.1`), WCZESNE PLATEAU (`improve_first > 0.2 and improve_second < 0.05`), DOBRY TREND (`improve_second > 0.1`), brak obserwacji (żaden warunek nie spełniony)
- `_diagnose_epsilon`: SZYBKI SPADEK (`mid_eps < 0.1`), brak obserwacji
- `_diagnose_td_error`: ROSNĄCY TD ERROR (`td_late > td_early * 1.5`), SPADAJĄCY TD ERROR (`td_late < td_early * 0.5`), brak kolumny `td_error_mean`, brak obserwacji
- `_diagnose_eval_vs_train`: EVAL << TRAIN (gap > 30%), EVAL > TRAIN (gap > 10%), WYSOKI STD (std > 20% mean), brak danych eval, fallback na `standalone_eval`

### Pytanie 3
#### Jak mockować `METRICS_DIR.glob()` w `list_runs()`?
Użyć `unittest.mock.patch` na `utils.analyze.METRICS_DIR` — podmienić na mock `Path` z kontrolowanym `.glob()` zwracającym listę fake `Path` obiektów. Każdy fake Path musi mieć atrybuty `.stem`, `.name`.

### Pytanie 4
#### Czy `export_summary_report` wymaga zapisu do realnego pliku?
Tak, ale używamy `tmp_path` fixture (wbudowany w pytest) — tworzy tymczasowy katalog, który jest automatycznie usuwany. NIE mockujemy `to_csv`, weryfikujemy realny zapis do tymczasowej lokalizacji. Alternatywnie można mockować `DataFrame.to_csv` i weryfikować wywołanie.
