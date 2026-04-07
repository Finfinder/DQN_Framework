# Duplikacja w replay_buffer.py — BaseReplayBuffer - Wynik analizy

## Szczegóły zadania

| Pole | Wartość |
|---|---|
| Jira ID | N/A (wewnętrzne zadanie refaktoryzacyjne) |
| Tytuł | Duplikacja w replay_buffer.py (~20%): wprowadzenie klasy bazowej BaseReplayBuffer |
| Opis | Klasy ReplayBuffer i NstepReplayBuffer mają zbliżoną strukturę — 4 z 6 metod to identyczny kod. Należy wprowadzić klasę bazową BaseReplayBuffer eliminującą duplikację. |
| Priorytet | Medium |
| Zgłaszający | SonarCloud / analiza jakości kodu |
| Data utworzenia | 2026-04-07 |
| Termin realizacji | — |
| Etykiety | refactoring, code-quality, duplication |
| Szacowany nakład pracy | S (Small) |
| Złożoność analizy rozwiązań | Nie dotyczy |

## Wpływ biznesowy

Redukcja duplikacji kodu w module buforów doświadczeń poprawia utrzymywalność frameworka. Każda zmiana w interfejsie buforów (np. dodanie nowej metody, zmiana formatu zwracanych danych przez `sample()`) musi być obecnie propagowana ręcznie do wielu klas. Wprowadzenie klasy bazowej eliminuje ryzyko niespójności i zmniejsza koszt przyszłych zmian.

## Zebrane informacje

### Baza wiedzy i narzędzia do zarządzania zadaniami

Brak powiązanych zadań w systemach zarządzania. Zadanie zidentyfikowane na podstawie analizy jakości kodu (metryka duplikacji ~20%).

### Baza kodu

#### Analiza pliku `memory/replay_buffer.py`

Plik zawiera 3 klasy buforów doświadczeń i 1 funkcję factory:

| Klasa | Linie kodu | Storage | Rola |
|---|---|---|---|
| `ReplayBuffer` | ~35 | `deque(maxlen=capacity)` | Uniform sampling — prosty bufor cykliczny |
| `PrioritizedReplayBuffer` | ~65 | `list` + `np.array` (priorities) | PER z importance-sampling weights |
| `NstepReplayBuffer` | ~50 | `deque(maxlen=capacity)` + `deque(maxlen=n_step)` | N-step returns z akumulacją gamma-discounted reward |
| `create_buffer()` | ~15 | — | Factory tworząca bufor na podstawie `config.buffer_type` |

#### Zduplikowane metody między ReplayBuffer i NstepReplayBuffer

| Metoda | Linie kodu | Status duplikacji |
|---|---|---|
| `sample(self, batch_size, _beta)` | 9 linii | **Identyczny** kod — `random.sample`, `zip(*batch)`, `np.array` |
| `update_priorities(self, indices, td_errors)` | 3 linie | **Identyczny** — no-op `pass` |
| `mean_priority(self)` | 2 linie | **Identyczny** — `return 0.0` |
| `__len__(self)` | 2 linie | **Identyczny** — `return len(self.memory)` |
| `__init__()` | różne | Różne — NstepReplayBuffer dodaje `n_step`, `gamma`, `_buffer` |
| `push()` | różne | Różne — NstepReplayBuffer akumuluje n-step returns |

**Podsumowanie: 4 z 6 metod to identyczny kod (~16 linii zduplikowanych).**

#### PrioritizedReplayBuffer — analiza kompatybilności

PrioritizedReplayBuffer różni się fundamentalnie:
- Storage: `list` + `np.array` priorities (nie `deque`)
- `push()` — zarządza priorytetami i `max_priority`
- `sample()` — zwraca 7 elementów (+ `indices`, `is_weights`) vs 5 elementów
- `update_priorities()` — realna implementacja (nie no-op)
- `mean_priority()` — oblicza rzeczywistą średnią priorytetów
- `__len__()` — zwraca `self.size` zamiast `len(self.memory)`

Mimo fundamentalnych różnic, PER implementuje **ten sam kontrakt interfejsu**. Dziedziczenie po BaseReplayBuffer z nadpisaniem wszystkich metod jest uzasadnione dla spójności typów i enforced interface.

#### Consumer analysis — kto korzysta z buforów

| Plik | Import | Sposób użycia |
|---|---|---|
| `agents/dqn_agent.py` | pośrednio (`self.memory`) | Duck typing — `push()`, `sample()`, `update_priorities()` |
| `train.py` | `create_buffer` | Factory pattern — `memory = create_buffer(config)` |
| `confirm_test.py` | `create_buffer` | Factory pattern — identycznie jak train.py |
| `tests/test_replay_buffer.py` | wszystkie klasy + factory | Testy bezpośrednio importują konkretne klasy |
| `tests/test_dqn_agent.py` | `ReplayBuffer`, `PrioritizedReplayBuffer`, `create_buffer` | Tworzenie agenta z buforem |

**Wniosek:** Żaden consumer nie sprawdza typu bufora (brak `isinstance`). Dodanie klasy bazowej nie wymaga zmian w konsumentach.

#### Istniejący wzorzec w projekcie — sieci neuronowe

Projekt **nie stosuje** klas bazowych dla sieci (`DQN` i `CNNDQN` to niezależne klasy `nn.Module`). Jednak sieci mają mniej duplikacji — wspólny jest tylko wzorzec dueling head, który ma różną wewnętrzną strukturę.

#### Pokrycie testowe

- **27 testów jednostkowych** w `tests/test_replay_buffer.py` (7 ReplayBuffer + 10 PER + 6 NstepReplayBuffer + 4 factory)
- Testy importują konkretne klasy — nie wymagają zmian
- Potrzebne: dodatkowe testy weryfikujące, że BaseReplayBuffer jest prawidłową ABC (np. test instantiation failure)

### Powiązane linki

- `memory/replay_buffer.py` — plik źródłowy z duplikacją
- `tests/test_replay_buffer.py` — testy jednostkowe buforów
- `agents/dqn_agent.py` — główny consumer interfejsu buforów
- `config/config.py` — konfiguracja `buffer_type` i parametrów buforów

### Analiza rozwiązań

Nie przeprowadzono — wymagania jednoznaczne, wzorzec ABC jest standardową praktyką Pythona.

### Powiązane wykresy i diagramy

```
Obecna struktura (bez dziedziczenia):

┌──────────────┐  ┌─────────────────────────┐  ┌───────────────────┐
│ ReplayBuffer │  │ PrioritizedReplayBuffer │  │ NstepReplayBuffer │
├──────────────┤  ├─────────────────────────┤  ├───────────────────┤
│ capacity     │  │ capacity                │  │ capacity          │
│ memory(deque)│  │ memory(list)            │  │ memory(deque)     │
├──────────────┤  │ priorities(ndarray)     │  │ _buffer(deque)    │
│ push()       │  │ position, size          │  │ n_step, gamma     │
│ sample()  ◄──┼──┼─ identyczna logika ─────┼──┤► sample()        │
│ update_pr ◄──┼──┼─ no-op ────────────────┼──┤► update_pr()     │
│ mean_pri  ◄──┼──┼─ return 0.0 ───────────┼──┤► mean_pri()      │
│ __len__   ◄──┼──┼─ len(self.memory) ─────┼──┤► __len__()       │
└──────────────┘  └─────────────────────────┘  └───────────────────┘

Docelowa struktura (z dziedziczeniem):

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

## Aktualny stan implementacji

### Istniejące komponenty

- `ReplayBuffer` — `memory/replay_buffer.py` — wymaga modyfikacji (dodanie dziedziczenia, usunięcie zduplikowanych metod)
- `PrioritizedReplayBuffer` — `memory/replay_buffer.py` — wymaga modyfikacji (dodanie dziedziczenia)
- `NstepReplayBuffer` — `memory/replay_buffer.py` — wymaga modyfikacji (dodanie dziedziczenia, usunięcie zduplikowanych metod)
- `create_buffer()` — `memory/replay_buffer.py` — bez zmian
- `DQNAgent` — `agents/dqn_agent.py` — bez zmian (duck typing)
- `tests/test_replay_buffer.py` — wymaga rozszerzenia (testy ABC)
- `tests/conftest.py` — bez zmian
- `tests/helpers.py` — bez zmian

### Kluczowe pliki i katalogi

- `memory/replay_buffer.py` — jedyny plik do refaktoryzacji, zawiera wszystkie 3 klasy buforów i factory
- `tests/test_replay_buffer.py` — testy jednostkowe do rozszerzenia o testy klasy bazowej
- `agents/dqn_agent.py` — consumer weryfikujący brak regresji interfejsu
- `config/config.py` — definicja `buffer_type` i parametrów per-buffer

## Analiza luk

### Pytanie 1
#### Czy PrioritizedReplayBuffer powinien dziedziczyć po BaseReplayBuffer, mimo że nadpisuje prawie wszystkie metody?
**Tak** — wszystkie 3 klasy dziedziczą po BaseReplayBuffer. Daje to spójną hierarchię typów, enforced interface via ABC i możliwość isinstance checks w przyszłości.

### Pytanie 2
#### Czy BaseReplayBuffer powinien być abc.ABC z @abstractmethod na push(), czy zwykłą klasą bazową?
**ABC z @abstractmethod** — wymusza implementację `push()` w podklasach (błąd przy próbie instancjonowania klasy bez implementacji push). Zgodne ze standardowymi praktykami Pythona dla klas kontraktowych.

### Pytanie 3
#### Czy BaseReplayBuffer powinien pozostać w tym samym pliku (replay_buffer.py) czy wydzielić do osobnego pliku?
**Ten sam plik** `replay_buffer.py` — prostsze, zgodne z obecną konwencją projektu (jeden moduł = jeden plik). Klasa bazowa będzie eksportowana z tego samego modułu.
