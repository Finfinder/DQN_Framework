# Duplikacja w replay_buffer.py — BaseReplayBuffer - Plan Implementacji

## Szczegóły Zadania

| Pole | Wartość |
|---|---|
| Tytuł | Duplikacja w replay_buffer.py (~20%): wprowadzenie klasy bazowej BaseReplayBuffer |
| Opis | Klasy ReplayBuffer i NstepReplayBuffer mają zbliżoną strukturę — 4 z 6 metod to identyczny kod. Introdukcja ABC `BaseReplayBuffer` eliminuje duplikację i formalizuje kontrakt interfejsu buforów. |
| Priorytet | Medium |
| Powiązany Research | [replay-buffer-base-class.research.md](replay-buffer-base-class.research.md) |

## Proponowane Rozwiązanie

Wprowadzenie abstrakcyjnej klasy bazowej `BaseReplayBuffer(ABC)` w pliku `memory/replay_buffer.py`, która:

1. Definiuje kontrakt interfejsu: `push()` jako `@abstractmethod`, domyślne implementacje `sample()`, `update_priorities()`, `mean_priority()`, `__len__()`.
2. Eliminuje 16 zduplikowanych linii kodu między `ReplayBuffer` a `NstepReplayBuffer`.
3. Wymusza implementację `push()` w każdej podklasie via ABC mechanism.
4. Wszystkie 3 istniejące klasy buforów dziedziczą po `BaseReplayBuffer`.

Refaktoryzacja jest **w pełni backward-compatible** — publiczny interfejs klas, factory i importy nie ulegają zmianie.

```
                    ┌───────────────────────────┐
                    │    BaseReplayBuffer (ABC)  │
                    ├───────────────────────────┤
                    │ capacity                  │
                    ├───────────────────────────┤
                    │ @abstractmethod push()    │
                    │ sample()     [default]    │
                    │ update_pr()  [no-op]      │
                    │ mean_pri()   [return 0.0] │
                    │ __len__()    [default]    │
                    └─────────┬─────────────────┘
             ┌────────────────┼──────────────────┐
             ▼                ▼                  ▼
    ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐
    │ ReplayBuffer │  │ Prioritized     │  │ NstepReplay   │
    │              │  │ ReplayBuffer    │  │ Buffer        │
    ├──────────────┤  ├─────────────────┤  ├───────────────┤
    │ memory(deque)│  │ memory(list)    │  │ memory(deque) │
    ├──────────────┤  │ priorities      │  │ _buffer(deque)│
    │ push() ✓     │  ├─────────────────┤  ├───────────────┤
    │              │  │ push() ✓        │  │ push() ✓      │
    │              │  │ sample() ✓      │  │ _flush_one()  │
    │              │  │ update_pr() ✓   │  │               │
    │              │  │ mean_pri() ✓    │  │               │
    │              │  │ __len__() ✓     │  │               │
    └──────────────┘  └─────────────────┘  └───────────────┘

    ✓ = nadpisuje metodę bazową
```

## Uzasadnienie Rozwiązania

### Wybrane podejście

ABC z `@abstractmethod` na `push()` i domyślnymi implementacjami wspólnych metod. Wszystkie 3 klasy dziedziczą po `BaseReplayBuffer`, zachowując pełną kompatybilność wsteczną.

### Porównanie z alternatywami

| Kryterium | ABC BaseReplayBuffer (wybrane) | Protocol (typing.Protocol) | Status quo (brak zmian) |
|---|---|---|---|
| Eliminacja duplikacji | ✅ Tak — 16 linii usunięte | ❌ Nie — Protocol to tylko interfejs, bez shared code | ❌ Nie |
| Enforced interface | ✅ TypeError przy brakującym push() | ✅ Structural subtyping | ❌ Brak — duck typing |
| isinstance checks | ✅ Tak | ⚠️ Wymaga runtime_checkable | ❌ Nie |
| Backward compatibility | ✅ Pełna | ✅ Pełna | ✅ Brak zmian |
| Złożoność | Niska — 1 klasa bazowa | Niska — 1 Protocol | Brak |
| Zgodność z Pythonem | ✅ abc jest standardem biblioteki | ⚠️ Wymaga Python 3.8+ | ✅ — |

### Dlaczego odrzucono alternatywy

- **Protocol**: Nie eliminuje duplikacji kodu — Protocol definiuje jedynie interfejs (structural subtyping), ale nie dostarcza domyślnych implementacji współdzielonych metod.
- **Status quo**: Utrzymuje 20% duplikacji, zwiększa ryzyko niespójności przy zmianach interfejsu, brak enforced contract.

## Model C4

### Diagram kontekstowy (Context)

Nie dotyczy — refaktoryzacja wewnętrzna modułu `memory`, bez wpływu na granice systemu.

### Diagram kontenerów (Container)

Nie dotyczy — zmiana w obrębie jednego pliku Pythona.

### Diagram komponentów (Component)

Nie dotyczy — zadanie obejmuje pojedynczy moduł (`memory/replay_buffer.py`). Diagram hierarchii klas w sekcji "Proponowane Rozwiązanie" w pełni opisuje architekturę.

## Rejestry Decyzji Architektonicznych (ADR)

### ADR-001: ABC vs Protocol vs plain base class dla BaseReplayBuffer

| Pole | Wartość |
|---|---|
| Status | Zaakceptowany |
| Data | 2026-04-07 |
| Kontekst | Trzy klasy buforów implementują ten sam 6-metodowy interfejs. ReplayBuffer i NstepReplayBuffer dzielą identyczny kod 4 metod (~16 linii). Potrzebny mechanizm eliminujący duplikację i formalizujący kontrakt. |

**Rozważane opcje**:
1. `abc.ABC` z `@abstractmethod` — klasa bazowa ze współdzieloną logiką i wymuszoną implementacją `push()`
2. `typing.Protocol` — structural subtyping, sam interfejs bez współdzielonej logiki
3. Zwykła klasa bazowa — współdzielona logika bez wymuszenia implementacji

**Decyzja**: Opcja 1 — `abc.ABC` z `@abstractmethod`

**Uzasadnienie**: Jedyna opcja eliminująca duplikację kodu **i** wymuszająca kontrakt. Protocol rozwiązuje tylko enforcement, plain class rozwiązuje tylko duplikację. ABC rozwiązuje oba problemy jednocześnie.

**Konsekwencje**:
- ✅ Eliminacja 16 linii zduplikowanego kodu
- ✅ TypeError przy próbie instancjonowania podklasy bez `push()`
- ✅ Spójna hierarchia typów — `isinstance(buf, BaseReplayBuffer)` działa dla wszystkich buforów
- ✅ Nowy import `from abc import ABC, abstractmethod` — standardowa biblioteka, zero zależności
- ⚠️ PrioritizedReplayBuffer nadpisuje prawie wszystkie metody — korzyść z dzielenia kodu jest minimalna, ale kontrakt jest wymuszony
- ⚠️ Domyślna `sample()` zakłada istnienie `self.memory` — nowe podklasy muszą pamiętać o stworzeniu tego atrybutu

### ADR-002: Lokalizacja BaseReplayBuffer — ten sam plik vs osobny plik

| Pole | Wartość |
|---|---|
| Status | Zaakceptowany |
| Data | 2026-04-07 |
| Kontekst | Moduł `memory/` zawiera jeden plik `replay_buffer.py`. BaseReplayBuffer może zostać dodany w tym samym pliku lub jako `base_replay_buffer.py`. |

**Rozważane opcje**:
1. Ten sam plik `replay_buffer.py` — mniej plików, prostsze importy
2. Osobny plik `base_replay_buffer.py` — lepsze SRP

**Decyzja**: Opcja 1 — ten sam plik

**Uzasadnienie**: Projekt stosuje konwencję "jeden moduł = jeden plik" w `memory/`. BaseReplayBuffer jest ściśle powiązany z konkretnymi buforami i nie ma sensu bez nich. Osobny plik niepotrzebnie zwiększa liczbę plików w module.

**Konsekwencje**:
- ✅ Zero zmian w ścieżkach importu konsumentów
- ✅ Spójne z istniejącą konwencją projektu
- ⚠️ Plik `replay_buffer.py` rośnie o ~15 linii (BaseReplayBuffer + import abc)

## Analiza Aktualnej Implementacji

### Już Zaimplementowane

- `create_buffer()` — `memory/replay_buffer.py` — factory pattern bez zmian, zwraca konkretne instancje
- `DQNAgent.train_step()` — `agents/dqn_agent.py` — consumer via duck typing, bez zmian
- `train.py` / `confirm_test.py` — entry pointy używające factory, bez zmian
- `tests/conftest.py` — fixture `small_config`, `per_config` — bez zmian
- `tests/helpers.py` — `make_transitions()`, `fill_buffer()` — bez zmian
- `config/config.py` — konfiguracja `buffer_type` i parametrów buforów — bez zmian

### Do Modyfikacji

- `ReplayBuffer` — `memory/replay_buffer.py` — dodać dziedziczenie po `BaseReplayBuffer`, `super().__init__(capacity)`, usunąć 4 zduplikowane metody (`sample`, `update_priorities`, `mean_priority`, `__len__`)
- `NstepReplayBuffer` — `memory/replay_buffer.py` — dodać dziedziczenie po `BaseReplayBuffer`, `super().__init__(capacity)`, usunąć 4 zduplikowane metody
- `PrioritizedReplayBuffer` — `memory/replay_buffer.py` — dodać dziedziczenie po `BaseReplayBuffer`, `super().__init__(capacity)`, zachować wszystkie nadpisane metody
- `tests/test_replay_buffer.py` — rozszerzyć o testy klasy bazowej ABC + testy isinstance w factory

### Do Utworzenia

- `BaseReplayBuffer` — nowa klasa ABC w `memory/replay_buffer.py` z `@abstractmethod push()` i domyślnymi implementacjami 4 metod
- `TestBaseReplayBuffer` — nowa klasa testowa w `tests/test_replay_buffer.py` weryfikująca kontrakt ABC

## Otwarte Pytania

| # | Pytanie | Odpowiedź | Status |
|---|----------|--------|--------|
| 1 | Czy PrioritizedReplayBuffer powinien dziedziczyć po BaseReplayBuffer? | Tak — wspólna hierarchia dla enforced interface | ✅ Rozwiązane |
| 2 | ABC z @abstractmethod czy zwykła klasa bazowa? | ABC z @abstractmethod na push() | ✅ Rozwiązane |
| 3 | Ten sam plik czy osobny? | Ten sam plik replay_buffer.py | ✅ Rozwiązane |

## Plan Implementacji

### Faza 1: Refaktoryzacja modułu buforów

#### Zadanie 1.1 - [CREATE] Dodanie klasy BaseReplayBuffer
**Opis**: Dodać klasę `BaseReplayBuffer(ABC)` na początku pliku `memory/replay_buffer.py` (po importach, przed `ReplayBuffer`). Klasa zawiera:
- `__init__(self, capacity)` — przechowuje `self.capacity`
- `@abstractmethod push(self, state, action, reward, next_state, done, _td_error=None)`
- Domyślna `sample(self, batch_size, _beta=0.4)` — obecna implementacja z `ReplayBuffer`
- Domyślna `update_priorities(self, indices, td_errors)` — no-op `pass`
- Domyślna `mean_priority(self)` — `return 0.0`
- Domyślna `__len__(self)` — `return len(self.memory)`

Dodać import `from abc import ABC, abstractmethod` na początku pliku.

**Definicja Ukończenia (Definition of Done)**:
- [x] Klasa `BaseReplayBuffer` istnieje w `memory/replay_buffer.py` i dziedziczy po `ABC`
- [x] Metoda `push()` jest oznaczona `@abstractmethod`
- [x] Domyślne implementacje `sample()`, `update_priorities()`, `mean_priority()`, `__len__()` przeniesione z `ReplayBuffer`
- [x] Import `from abc import ABC, abstractmethod` dodany na początku pliku

#### Zadanie 1.2 - [MODIFY] Refaktoryzacja ReplayBuffer
**Opis**: Zmienić `ReplayBuffer` aby dziedziczył po `BaseReplayBuffer`. W `__init__` wywołać `super().__init__(capacity)` i zachować `self.memory = deque(maxlen=capacity)`. Usunąć metody `sample()`, `update_priorities()`, `mean_priority()`, `__len__()` — dziedziczone z bazy. Zachować `push()` (unikalna logika) i docstring klasy.

**Definicja Ukończenia (Definition of Done)**:
- [x] `ReplayBuffer` dziedziczy po `BaseReplayBuffer`
- [x] `__init__` wywołuje `super().__init__(capacity)`
- [x] Metody `sample()`, `update_priorities()`, `mean_priority()`, `__len__()` usunięte z klasy
- [x] Metoda `push()` zachowana bez zmian
- [x] Docstring klasy zachowany

#### Zadanie 1.3 - [MODIFY] Refaktoryzacja PrioritizedReplayBuffer
**Opis**: Zmienić `PrioritizedReplayBuffer` aby dziedziczył po `BaseReplayBuffer`. W `__init__` wywołać `super().__init__(capacity)` (zastępuje `self.capacity = capacity`). Wszystkie metody zachowane — PER nadpisuje każdą domyślną implementację z bazy.

**Definicja Ukończenia (Definition of Done)**:
- [x] `PrioritizedReplayBuffer` dziedziczy po `BaseReplayBuffer`
- [x] `__init__` wywołuje `super().__init__(capacity)` zamiast `self.capacity = capacity`
- [x] Wszystkie istniejące metody zachowane bez zmian (nadpisują bazowe)
- [x] Docstring klasy zachowany

#### Zadanie 1.4 - [MODIFY] Refaktoryzacja NstepReplayBuffer
**Opis**: Zmienić `NstepReplayBuffer` aby dziedziczył po `BaseReplayBuffer`. W `__init__` wywołać `super().__init__(capacity)` i zachować `self.memory`, `self._buffer`, `self.n_step`, `self.gamma`. Usunąć metody `sample()`, `update_priorities()`, `mean_priority()`, `__len__()` — dziedziczone z bazy. Zachować `push()` i `_flush_one()`.

**Definicja Ukończenia (Definition of Done)**:
- [x] `NstepReplayBuffer` dziedziczy po `BaseReplayBuffer`
- [x] `__init__` wywołuje `super().__init__(capacity)`
- [x] Metody `sample()`, `update_priorities()`, `mean_priority()`, `__len__()` usunięte z klasy
- [x] Metody `push()` i `_flush_one()` zachowane bez zmian
- [x] Docstring klasy zachowany

### Faza 2: Testy

#### Zadanie 2.1 - [CREATE] Testy klasy bazowej BaseReplayBuffer
**Opis**: Dodać klasę `TestBaseReplayBuffer` w `tests/test_replay_buffer.py` zawierającą:
- Test, że bezpośrednia instancja `BaseReplayBuffer` rzuca `TypeError` (ABC enforcement)
- Test, że konkretna podklasa z implementacją `push()` może być instancjonowana
- Testy isinstance dla każdej z 3 konkretnych klas buforów

Dodać import `BaseReplayBuffer` w sekcji importów testu.

**Definicja Ukończenia (Definition of Done)**:
- [x] Klasa `TestBaseReplayBuffer` istnieje w `tests/test_replay_buffer.py`
- [x] Test `test_cannot_instantiate_directly` — weryfikuje TypeError przy bezpośredniej instancji ABC
- [x] Test `test_concrete_subclass_instantiable` — weryfikuje, że podklasa z push() jest instancjonowalna
- [x] Test `test_isinstance_replay_buffer` — `isinstance(ReplayBuffer(10), BaseReplayBuffer)` jest True
- [x] Test `test_isinstance_prioritized` — `isinstance(PrioritizedReplayBuffer(10), BaseReplayBuffer)` jest True
- [x] Test `test_isinstance_nstep` — `isinstance(NstepReplayBuffer(10), BaseReplayBuffer)` jest True
- [x] Import `BaseReplayBuffer` dodany w sekcji importów

#### Zadanie 2.2 - [MODIFY] Rozszerzenie testów factory o isinstance
**Opis**: Dodać asercję `isinstance(buf, BaseReplayBuffer)` w wybranych istniejących testach factory (`TestCreateBufferFactory`), aby potwierdzić, że factory zwraca instancje zgodne z hierarchią typów.

**Definicja Ukończenia (Definition of Done)**:
- [x] Co najmniej jeden test factory zawiera asercję `isinstance(buf, BaseReplayBuffer)`
- [x] Wszystkie istniejące 27 testów przechodzą bez zmian

#### Zadanie 2.3 - [REUSE] Uruchomienie pełnego zestawu testów
**Opis**: Uruchomić kompletny zestaw testów (`pytest tests/ -v`) w środowisku `.venv` i potwierdzić brak regresji. Zweryfikować, że CI lint (`ruff check . --select E9,F63,F7,F82`) przechodzi pomyślnie.

**Definicja Ukończenia (Definition of Done)**:
- [x] Wszystkie istniejące testy przechodzą (0 failures)
- [x] Nowe testy BaseReplayBuffer przechodzą
- [x] `ruff check . --select E9,F63,F7,F82` — zero błędów
- [x] `python -m compileall -q .` — zero błędów kompilacji

### Faza 3: Dokumentacja

#### Zadanie 3.1 - [MODIFY] Aktualizacja CHANGELOG.md
**Opis**: Dodać wpis w sekcji `[Unreleased]` w CHANGELOG.md opisujący refaktoryzację. Kategoria: `Changed`.

**Definicja Ukończenia (Definition of Done)**:
- [x] Wpis w `[Unreleased]` opisuje wprowadzenie `BaseReplayBuffer` ABC
- [x] Format zgodny z Keep a Changelog

### Faza 4: Code Review

#### Zadanie 4.1 - Code Review przez agenta `code-reviewer`
**Opis**: Przegląd kodu wykonany przez agenta `code-reviewer`. Weryfikacja poprawności refaktoryzacji, zachowania backward compatibility, zgodności z konwencjami projektu i kompletności testów.

**Definicja Ukończenia (Definition of Done)**:
- [x] Agent `code-reviewer` przeprowadził przegląd zmian
- [x] Wszystkie uwagi krytyczne i ważne zostały zaadresowane
- [x] Potwierdzono backward compatibility (brak zmian w consumerach)

## Aspekty Bezpieczeństwa

Brak aspektów bezpieczeństwa — refaktoryzacja wewnętrzna modułu buforów doświadczeń, bez wpływu na I/O, sieć, dane użytkownika ani konfigurację.

## Strategia Testowania

### Piramida testów

| Typ testu | Zakres | Szacowana liczba | Pokrycie |
|---|---|---|---|
| Jednostkowe | ABC enforcement, isinstance, regresja istniejących buforów | 5 nowych + 27 istniejących | ≥80% branch coverage dla replay_buffer.py |

### Podejście do testowania

- [x] Testy regresji — wszystkie 27 istniejących testów muszą przejść bez modyfikacji
- [x] Testy kontraktu ABC — weryfikacja TypeError przy bezpośredniej instancji
- [x] Testy hierarchii typów — isinstance checks dla wszystkich 3 klas

### Testy wydajnościowe

Nie dotyczy — refaktoryzacja nie zmienia logiki wykonania, dziedziczenie w Pythonie ma pomijalny narzut.

### Testy dostępności

Nie dotyczy — brak komponentów UI.

### Testy architektoniczne

Nie dotyczy — projekt nie używa narzędzi do automatycznego egzekwowania reguł architektonicznych. Testy isinstance w Fazie 2 pełnią rolę ręcznych testów architektonicznych potwierdzających hierarchię typów.

### Testy mutacyjne

Nie dotyczy — refaktoryzacja eliminuje duplikację bez zmiany logiki biznesowej. Istniejące testy pokrywają zachowanie metod.

## Zapewnienie Jakości

- [x] Wszystkie 27 istniejących testów przechodzą bez modyfikacji (regresja)
- [x] 5 nowych testów BaseReplayBuffer przechodzi
- [x] `BaseReplayBuffer` nie może być instancjonowana bezpośrednio (TypeError)
- [x] Wszystkie 3 konkretne klasy buforów są instancjami `BaseReplayBuffer`
- [x] `create_buffer()` factory działa bez zmian
- [x] `DQNAgent` działa bez zmian z każdym typem bufora
- [x] CI pipeline przechodzi: ruff lint + compileall + pytest + CLI smoke tests
- [x] Plik `memory/replay_buffer.py` nie zawiera zduplikowanych metod `sample()`, `update_priorities()`, `mean_priority()`, `__len__()` między klasami

## Usprawnienia (Poza Zakresem)

- **SumTree dla PER**: Obecna implementacja PER używa liniowego skanowania priorytetów — SumTree (`O(log n)` sampling) poprawiłby wydajność przy dużych buforach.
- **Generyczna BaseReplayBuffer z typowaniem**: Dodanie `Generic[T]` do BaseReplayBuffer z type hints (np. `Transition = tuple[np.ndarray, int, float, np.ndarray, bool]`) — odłożone, bo projekt nie stosuje type annotations.
- **Analogiczna refaktoryzacja sieci neuronowych**: `DQN` i `CNNDQN` dzielą wzorzec dueling head, ale różnice strukturalne (MLP vs CNN) czynią wspólną bazę mniej opłacalną.

## Changelog

| Data | Zmiana |
|---|---|
| 2026-04-07 | Utworzenie planu implementacji |
| 2026-04-07 | Code review przeprowadzony — werdykt: APPROVED. Szczegóły w sekcji Code Review Findings |

## Code Review Findings

**Werdykt**: APPROVED

**Przegląd przeprowadzony**: 2026-04-07

```
Postęp analizy:
- [x] Krok 1: Zrozum opis zadania (research.md)
- [x] Krok 2: Zrozum plan implementacji (plan.md)
- [x] Krok 3: Przeanalizuj implementację vs plan
- [x] Krok 4: Zweryfikuj testy (34/34 passed)
- [x] Krok 5: Uruchom testy (107/107 passed)
- [x] Krok 6: Najlepsze praktyki (SOLID, DRY, KISS)
- [x] Krok 7: Statyczna analiza kodu (ruff clean, SonarQube S1186 fixed)
- [x] Krok 8: Bezpieczeństwo (S2245 — safe, brak OWASP concerns)
- [x] Krok 9: Skalowalność (brak circular deps, granice modułów OK)
```

### Znaleziska

| # | Severity | Opis | Status |
|---|---|---|---|
| F-1 | MINOR | S1186: `update_priorities()` w `BaseReplayBuffer` — pusta metoda bez komentarza (SonarQube for IDE) | ✅ Naprawione — dodano `# No-op: uniform buffers do not use priorities` |
| F-2 | MINOR | S1186: `MinimalBuffer.push()` w teście — pusta metoda bez komentarza (SonarQube for IDE) | ✅ Naprawione — dodano komentarz wyjaśniający |
| F-3 | INFO | S2245: `random.sample()` w `BaseReplayBuffer.sample()` — security hotspot PRNG (SonarQube for IDE) | ✅ Safe — bufor RL używa PRNG do symulacji, nie do bezpieczeństwa |
| F-4 | IMPORTANT | Implicit `self.memory` dependency w `sample()` i `__len__()` bazowej klasy — nowe podklasy muszą pamiętać o ⁠inicjalizacji `self.memory` | ✅ Akceptowalne — udokumentowane w ADR-001 jako znany kompromis projektowy |

### Podsumowanie

Refaktoryzacja jest poprawna, backward-compatible i dobrze przetestowana. Hierarchia ABC właściwie zastosowana: `push()` wymuszony przez `@abstractmethod`, 4 domyslne implementacje wyeliminowały ~16 linii duplikacji. Wszystkie 107 testów przechodzi, lint czysty, konsumenci (`DQNAgent`, `create_buffer()`, `train.py`) bez zmian. Dwa drobne issues SonarQube (S1186) naprawione podczas review. Security hotspot S2245 oceniony jako safe.
