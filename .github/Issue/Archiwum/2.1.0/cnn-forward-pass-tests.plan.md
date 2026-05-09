# Testy forward pass CNN (bez GPU) — Plan Implementacji

## Szczegóły Zadania

| Pole | Wartość |
|---|---|
| Tytuł | Dodaj testy forward pass CNN (bez GPU) dla `models/cnn_dqn_network.py` |
| Opis | Utworzenie zestawu testów jednostkowych weryfikujących forward pass sieci `CNNDQN` — tworzenie instancji, kształt wyjścia, warianty standard/dueling, integracja z factory `create_network()`. Testy CPU-only, kompatybilne z CI. |
| Priorytet | Średni |
| Powiązany Research | [cnn-forward-pass-tests.research.md](cnn-forward-pass-tests.research.md) |

## Proponowane Rozwiązanie

Utworzenie nowego pliku testowego `tests/test_cnn_dqn_network.py` z czterema klasami testowymi pokrywającymi:
1. **Tworzenie instancji** `CNNDQN` z różnymi parametrami (domyślne, custom, dueling)
2. **Forward pass standard** (dueling=False) — kształt wyjścia, dtype, brak NaN/Inf
3. **Forward pass dueling** (dueling=True) — kształt wyjścia, dtype, normalizacja advantage
4. **Integracja z factory** `create_network()` z konfiguracją CNN

Dodanie fixture `cnn_config` w `tests/conftest.py` z minimalnymi parametrami dla szybkich testów (mały `cnn_hidden_dim`, mniejsze `conv_layers`).

Wszystkie testy działają wyłącznie na CPU — brak markera `requires_cuda`, kompatybilność z GitHub Actions CI.

## Uzasadnienie Rozwiązania

### Wybrane podejście

Bezpośrednie testy klasy `CNNDQN` i factory `create_network()` z użyciem pytest + PyTorch na CPU. Fixture z minimalnymi parametrami sieci zapewnia szybkość testów przy pełnym pokryciu ścieżek kodu.

### Porównanie z alternatywami

| Kryterium | Bezpośrednie testy CNNDQN (wybrane) | Testy przez DQNAgent z CNN | Testy E2E z środowiskiem Atari |
|---|---|---|---|
| Izolacja | ✅ Czysta — testuje tylko sieć | ❌ Zależność od agenta, bufora | ❌ Zależność od Atari, wrapperów |
| Szybkość | ✅ Milisekundy | ⚠️ Wolniejsze (bufor, optimizer) | ❌ Bardzo wolne + wymaga ale-py |
| Kompatybilność CI | ✅ CPU-only, brak zależności | ✅ CPU-only | ❌ Wymaga gymnasium[atari] |
| Pokrycie forward pass | ✅ Pełne (obie ścieżki) | ⚠️ Pośrednie | ⚠️ Pośrednie |
| Diagnostyka błędów | ✅ Precyzyjna lokalizacja | ❌ Trudna do zdiagnozowania | ❌ Trudna do zdiagnozowania |

### Dlaczego odrzucono alternatywy
- **Testy przez DQNAgent**: Dodają niepotrzebne zależności (bufor, optimizer, epsilon-greedy). Forward pass jest testowany pośrednio — błąd w sieci CNN maskowany przez inne komponenty.
- **Testy E2E z Atari**: Wymagają `gymnasium[atari]` i `ale-py`, które nie są zainstalowane w CI. Zbyt wolne na testy jednostkowe.

## Model C4

### Diagram kontekstowy (Context)

Nie dotyczy — zadanie obejmuje pojedynczy komponent (plik testowy) w istniejącym frameworku.

### Diagram kontenerów (Container)

Nie dotyczy — brak nowych kontenerów.

### Diagram komponentów (Component)

Nie dotyczy — zadanie obejmuje pojedynczy komponent.

## Rejestry Decyzji Architektonicznych (ADR)

Nie dotyczy.

## Analiza Aktualnej Implementacji

### Już Zaimplementowane
- `CNNDQN` (klasa sieci CNN) - `models/cnn_dqn_network.py` - testowany moduł, bez zmian
- `create_network()` (factory) - `models/dqn_network.py` - factory tworząca CNNDQN dla network_type="cnn", bez zmian
- `Config("ALE/Pong-v5")` - `config/config.py` - konfiguracja CNN z parametrami produkcyjnymi, bez zmian
- `conftest.py` (fixture'y) - `tests/conftest.py` - istniejące fixture'y config, small_config, per_config
- `helpers.py` - `tests/helpers.py` - helpery make_transitions(), fill_buffer() — dostępne ale nie wymagane

### Do Modyfikacji
- `conftest.py` - `tests/conftest.py` - dodanie nowej fixture `cnn_config` z minimalnymi parametrami CNN

### Do Utworzenia
- `test_cnn_dqn_network.py` - nowy plik testowy z 4 klasami testowymi pokrywającymi forward pass CNNDQN

## Otwarte Pytania

| # | Pytanie | Odpowiedź | Status |
|---|----------|--------|--------|
| 1 | Czy testy powinny obejmować `_init_weights()`? | Nie — zakres to forward pass. Pośrednia weryfikacja przez brak NaN/Inf. | ✅ Rozwiązane |
| 2 | Czy potrzebna fixture `cnn_config` w conftest.py? | Tak — Config("ALE/Pong-v5") ma cnn_hidden_dim=1024, co spowalnia testy. Fixture z cnn_hidden_dim=64. | ✅ Rozwiązane |
| 3 | Czy mniejszy frame_size jest akceptowalny? | Tak — testy weryfikują kształty, nie jakość predykcji. Zachować 1 test z 84x84. | ✅ Rozwiązane |
| 4 | Czy testować advantage mean ≈ 0 w dueling? | Pośrednio — weryfikacja kształtu i brak NaN/Inf. Bezpośredni test normalizacji advantage. | ✅ Rozwiązane |

## Plan Implementacji

### Faza 1: Fixture CNN

#### Zadanie 1.1 - [MODYFIKUJ] Dodanie fixture `cnn_config` w `tests/conftest.py`
**Opis**: Dodanie nowej fixture `cnn_config` na końcu pliku `tests/conftest.py`. Fixture tworzy `Config("ALE/Pong-v5")` z nadpisanymi parametrami dla szybkości testów: mały `cnn_hidden_dim` (64), mniejsze `conv_layers` `[(8, 4, 2), (16, 3, 1)]`. Wymuszenie `device = torch.device("cpu")`.

**Definicja Ukończenia (Definition of Done)**:
- [x] Fixture `cnn_config` dodana na końcu `tests/conftest.py`
- [ ] Fixture bazuje na `Config("ALE/Pong-v5")` z nadpisanymi: `cnn_hidden_dim=64`, `conv_layers=[(8, 4, 2), (16, 3, 1)]`, `device=torch.device("cpu")`
- [x] `pytest --collect-only` zbiera fixture bez błędów
- [x] Fixture jest dostępna w nowym pliku testowym

### Faza 2: Testy forward pass CNNDQN

#### Zadanie 2.1 - [UTWÓRZ] Klasa `TestCNNDQNCreation` w `tests/test_cnn_dqn_network.py`
**Opis**: Testy tworzenia instancji `CNNDQN` z różnymi parametrami — weryfikacja, że konstruktor nie rzuca wyjątków i poprawnie ustawia atrybuty.

Scenariusze:
- Tworzenie z domyślnymi `conv_layers` (None → fallback na [(32,8,4), (64,4,2), (64,3,1)]), `input_shape=(4, 84, 84)`, `action_dim=6`
- Tworzenie z `dueling=True` — weryfikacja obecności `value_head` i `advantage_head`
- Tworzenie z `dueling=False` — weryfikacja obecności `q_head`
- Tworzenie z custom `conv_layers=[(8, 4, 2), (16, 3, 1)]` i małym `hidden_dim=32`
- Weryfikacja atrybutu `action_dim` i `dueling`

**Definicja Ukończenia (Definition of Done)**:
- [x] Klasa `TestCNNDQNCreation` z minimum 4 metodami testowymi
- [x] Testy tworzenia z domyślnymi parametrami, dueling=True, dueling=False, custom conv_layers
- [x] Weryfikacja atrybutów `action_dim`, `dueling`, obecności odpowiednich głów (`q_head` / `value_head` + `advantage_head`)
- [x] Wszystkie testy przechodzą na CPU

#### Zadanie 2.2 - [UTWÓRZ] Klasa `TestCNNDQNForwardStandard` w `tests/test_cnn_dqn_network.py`
**Opis**: Testy forward pass dla wariantu standard (dueling=False). Weryfikacja kształtu wyjścia, typu danych, braku NaN/Inf, deterministyczności w eval mode.

Scenariusze:
- Forward pass z batch=1: input `(1, C, H, W)` → output `(1, action_dim)`
- Forward pass z batch>1 (np. batch=4): input `(4, C, H, W)` → output `(4, action_dim)`
- Output dtype: `torch.float32`
- Brak NaN i Inf w output
- Deterministyczność: ten sam input w `eval()` mode → identyczny output
- Forward pass z domyślnymi parametrami (84x84) — weryfikacja konfiguracji produkcyjnej

Użyć fixture `cnn_config` do pobrania parametrów sieci i `torch.no_grad()` w testach inferencyjnych.

**Definicja Ukończenia (Definition of Done)**:
- [x] Klasa `TestCNNDQNForwardStandard` z minimum 5 metod testowych
- [x] Testy pokrywają: batch=1, batch>1, output shape, dtype float32, brak NaN/Inf, deterministyczność eval
- [x] Przynajmniej 1 test z domyślnymi parametrami (input 84x84, domyślne conv_layers)
- [x] Wszystkie testy używają `torch.no_grad()` i działają na CPU

#### Zadanie 2.3 - [UTWÓRZ] Klasa `TestCNNDQNForwardDueling` w `tests/test_cnn_dqn_network.py`
**Opis**: Testy forward pass dla wariantu dueling (dueling=True). Weryfikacja kształtu wyjścia, typu danych, braku NaN/Inf, normalizacji advantage.

Scenariusze:
- Forward pass z batch=1: input `(1, C, H, W)` → output `(1, action_dim)`
- Forward pass z batch>1 (np. batch=4): input `(4, C, H, W)` → output `(4, action_dim)`
- Output dtype: `torch.float32`
- Brak NaN i Inf w output
- Normalizacja advantage: bezpośredni test, że `advantage - advantage.mean(dim=1)` daje mean ≈ 0 per sample (dostęp przez `advantage_head`)
- Output shape identyczny z wariantem standard (ta sama sieć, inna ścieżka forward)

**Definicja Ukończenia (Definition of Done)**:
- [x] Klasa `TestCNNDQNForwardDueling` z minimum 5 metod testowych
- [x] Testy pokrywają: batch=1, batch>1, output shape, dtype float32, brak NaN/Inf
- [x] Test normalizacji advantage — weryfikacja, że znormalizowane advantage mają mean ≈ 0 per sample
- [x] Wszystkie testy używają `torch.no_grad()` i działają na CPU

#### Zadanie 2.4 - [UTWÓRZ] Klasa `TestCNNDQNFactory` w `tests/test_cnn_dqn_network.py`
**Opis**: Testy integracji factory `create_network()` z konfiguracją CNN. Weryfikacja, że `create_network()` tworzy instancję `CNNDQN` z poprawnymi parametrami i że forward pass działa.

Scenariusze:
- `create_network(cnn_config, state_shape, action_dim)` zwraca instancję `CNNDQN`
- Forward pass przez sieć utworzoną z factory — poprawny kształt wyjścia
- Dueling variant z factory (config.use_dueling) — weryfikacja atrybutu dueling

Użyć fixture `cnn_config` (z conftest.py) z nadpisanymi parametrami.

**Definicja Ukończenia (Definition of Done)**:
- [x] Klasa `TestCNNDQNFactory` z minimum 3 metody testowe
- [x] Test weryfikujący typ zwracanej instancji (`isinstance(net, CNNDQN)`)
- [x] Test forward pass przez sieć z factory — poprawny output shape
- [x] Test dueling variant z factory
- [x] Wszystkie testy działają na CPU

### Faza 3: Weryfikacja i Code Review

#### Zadanie 3.1 - [WERYFIKACJA] Uruchomienie testów i sprawdzenie pokrycia
**Opis**: Uruchomienie pełnego zestawu testów (`pytest tests/ -v`) i weryfikacja, że:
- Wszystkie nowe testy przechodzą
- Żaden istniejący test nie jest złamany
- Pokrycie `cnn_dqn_network.py` znacząco wzrosło (cel: ≥70% line coverage)

**Definicja Ukończenia (Definition of Done)**:
- [x] `pytest tests/test_cnn_dqn_network.py -v` — wszystkie testy przechodzą
- [x] `pytest tests/ -v` — żaden istniejący test nie został złamany
- [x] Pokrycie `models/cnn_dqn_network.py` ≥70% (wzrost z 11.32% → 100%)

#### Zadanie 3.2 - [CODE REVIEW] Przegląd kodu przez agenta `code-reviewer`
**Opis**: Pełny przegląd kodu nowego pliku testowego i zmian w conftest.py przez agenta `code-reviewer`. Weryfikacja zgodności z konwencjami projektu DQN_Framework.

**Definicja Ukończenia (Definition of Done)**:
- [x] Code review przeprowadzony przez agenta `code-reviewer`
- [x] Wszystkie uwagi krytyczne rozwiązane
- [x] Kod zgodny z konwencjami projektu (brak type annotations, brak docstringów, grupowanie klasowe, snake_case)

## Aspekty Bezpieczeństwa

- Nie dotyczy — zadanie obejmuje wyłącznie testy jednostkowe. Brak danych wejściowych użytkownika, brak operacji sieciowych, brak I/O plikowego.

## Strategia Testowania

### Piramida testów

| Typ testu | Zakres | Szacowana liczba | Pokrycie |
|---|---|---|---|
| Jednostkowe | Forward pass CNNDQN: tworzenie instancji, standard forward, dueling forward, factory | ~17-20 testów | ≥70% line coverage dla `cnn_dqn_network.py` (wzrost z 11.32%) |
| Integracyjne | Nie dotyczy | 0 | — |
| E2E | Nie dotyczy | 0 | — |

### Podejście do testowania
- [x] Testy CPU-only — brak markera `requires_cuda`, kompatybilność z CI
- [x] `torch.no_grad()` w testach inferencyjnych (forward pass)
- [x] Fixture `cnn_config` z minimalnymi parametrami dla szybkości
- [x] Grupowanie w klasy `class TestXxx:` zgodnie z konwencją projektu
- [x] Brak type annotations i docstringów (konwencja DQN_Framework)
- [x] Przynajmniej 1 test z parametrami produkcyjnymi (84x84, domyślne conv_layers)

### Testy wydajnościowe

Nie dotyczy.

### Testy dostępności

Nie dotyczy.

### Testy architektoniczne

Nie dotyczy.

### Testy mutacyjne

Nie dotyczy.

## Zapewnienie Jakości

- [x] Wszystkie nowe testy przechodzą (`pytest tests/test_cnn_dqn_network.py -v`)
- [x] Żaden istniejący test nie jest złamany (`pytest tests/ -v`)
- [x] Pokrycie `models/cnn_dqn_network.py` wzrosło z 11.32% do ≥70%
- [x] Testy działają na CPU (bez GPU) — kompatybilność z CI
- [x] Czas wykonania nowych testów < 5 sekund (małe parametry sieci)
- [x] Kod zgodny z konwencjami DQN_Framework (brak type annotations, brak docstringów, grupowanie klasowe, snake_case)
- [x] Fixture `cnn_config` nie koliduje z istniejącymi fixture'ami

## Usprawnienia (Poza Zakresem)

- Testy `_init_weights()` — weryfikacja wartości wag po inicjalizacji ortogonalnej
- Testy backward pass / gradient flow — weryfikacja, że `loss.backward()` propaguje gradienty
- Parametryzacja testów z `@pytest.mark.parametrize` dla różnych `frame_size` (32x32, 64x64, 128x128)
- Testy wydajności forward pass — benchmarki czasu inferencji CPU vs GPU
- Pokrycie `cnn_dqn_network.py` > 90% (wymaga testów `_init_weights` i edge case'ów)
- Testy z losowym seedem (`torch.manual_seed`) dla reprodukowalności

## Code Review Findings

### Przegląd #1 (2026-04-19)

**Data przeglądu**: 2026-04-19

```
Postęp analizy:
- [x] Krok 1: Zrozum opis zadania
- [x] Krok 2: Zrozum plan implementacji zadania
- [x] Krok 3: Przeanalizuj zaimplementowane rozwiązanie i porównaj je z opisem zadania oraz planem implementacji
- [x] Krok 4: Zweryfikuj, czy rozwiązanie zawiera wszystkie niezbędne testy
- [x] Krok 5: Uruchom dostępne testy
- [x] Krok 6: Zweryfikuj, czy rozwiązanie przestrzega najlepszych praktyk
- [x] Krok 7: Uruchom narzędzia do statycznej analizy kodu i formatowania
- [x] Krok 8: Zweryfikuj, czy rozwiązanie jest bezpieczne
- [x] Krok 9: Zweryfikuj, czy rozwiązanie jest skalowalne
```

**Decyzja**: ✅ Zatwierdzony z uwagą MUST FIX — usunąć `import pytest` przed commitem.

**Uwagi MUST FIX**: 1 (F401 unused import). **Uwagi SHOULD FIX**: 1 (DoD 1.1 deviation — zaakceptowane).

---

### Przegląd #2 — Re-review po naprawie (2026-04-19)

**Data przeglądu**: 2026-04-19

```
Postęp analizy:
- [x] Krok 1: Zrozum opis zadania
- [x] Krok 2: Zrozum plan implementacji zadania
- [x] Krok 3: Przeanalizuj zaimplementowane rozwiązanie i porównaj je z opisem zadania oraz planem implementacji
- [x] Krok 4: Zweryfikuj, czy rozwiązanie zawiera wszystkie niezbędne testy
- [x] Krok 5: Uruchom dostępne testy
- [x] Krok 6: Zweryfikuj, czy rozwiązanie przestrzega najlepszych praktyk
- [x] Krok 7: Uruchom narzędzia do statycznej analizy kodu i formatowania
- [x] Krok 8: Zweryfikuj, czy rozwiązanie jest bezpieczne
- [x] Krok 9: Zweryfikuj, czy rozwiązanie jest skalowalne
```

#### Weryfikacja naprawy z Przeglądu #1

| Finding | Status | Szczegóły |
|---|---|---|
| F1 — `import pytest` unused (F401) | ✅ Naprawione | Linia usunięta. `ruff check` przechodzi bez findings. |
| M1 — DoD 1.1 deviation | ✅ Zaakceptowane | Odchylenie udokumentowane, checkbox odznaczony. |

#### Wyniki testów

| Metryka | Wartość |
|---|---|
| Nowe testy | 25 (4 klasy) |
| Istniejące testy | 154 |
| Łącznie po zmianach | 179 |
| Status | ✅ 179/179 passed |
| Czas nowych testów | 0.20s |
| Pokrycie `cnn_dqn_network.py` przed | 11.32% |
| Pokrycie `cnn_dqn_network.py` po | **100%** |

#### Analiza statyczna

| Narzędzie | Status | Uwagi |
|---|---|---|
| `ruff check --select E9,F63,F7,F82` (CI rules) | ✅ Przechodzi | Brak naruszeń |
| `ruff check` (pełny) | ✅ Przechodzi | 0 findings w `test_cnn_dqn_network.py` i `conftest.py` |
| `ruff format --check` | ℹ️ Niezgodności | Istniejące pliki testowe również nie przechodzą — nie jest regresja |
| SonarQube for IDE | ✅ Brak issues | Deweloper potwierdził brak issues w panelu Problems |

#### Uwagi krytyczne

Brak.

#### Uwagi do naprawienia (MUST FIX)

Brak.

#### Uwagi mniejsze (SHOULD FIX)

Brak nowych. M1 z przeglądu #1 zaakceptowane — bez dalszych działań.

#### Pozytywne aspekty

- **100% pokrycia** `cnn_dqn_network.py` — przekracza cel ≥70%
- **0.20s czas wykonania** — znacznie poniżej limitu 5s
- **Czysta izolacja** — testy bezpośrednie na `CNNDQN`, brak zależności od agenta/bufora/środowiska
- **Dobre pokrycie scenariuszy**: tworzenie (7), standard forward (7), dueling forward (7), factory (4)
- **Test advantage normalization** (`test_advantage_normalization_mean_near_zero`) weryfikuje, że `forward()` daje ten sam wynik co ręczne obliczenie `value + (advantage - mean(advantage))` — wartościowy test regresyjny
- **Test produkcyjny 84x84** — weryfikuje realistyczne wymiary Atari
- **Stałe modułowe** (`_SMALL_SHAPE`, `_ACTION_DIM`, etc.) eliminują powtórzenia
- **Konwencje projektu** w pełni przestrzegane: brak type annotations, brak docstringów w testach, grupowanie klasowe, snake_case
- **Import czysty** — tylko `torch` i dwa lokalne importy, zero zbędnych zależności
- **Brak problemów bezpieczeństwa** — testy jednostkowe bez I/O, bez danych użytkownika
- **Brak problemów skalowalności** — kod testowy, nie produkcyjny

#### Aspekty bezpieczeństwa

Nie dotyczy — zadanie obejmuje wyłącznie testy jednostkowe. Brak wektorów ataku.

#### Decyzja

**✅ Zatwierdzony — brak uwag blokujących.** Gotowe do commita.

## Changelog

- 2026-04-19: Code review #1 przeprowadzony — 0 uwag krytycznych, 1 MUST FIX (unused import), 1 SHOULD FIX (DoD deviation)
- 2026-04-19: Usunięto `import pytest` z `tests/test_cnn_dqn_network.py` (F401 — unused import zgłoszony przez ruff)
- 2026-04-19: Code review #2 (re-review) — wszystkie findings z #1 rozwiązane, 0 nowych uwag, ✅ zatwierdzony do commita
