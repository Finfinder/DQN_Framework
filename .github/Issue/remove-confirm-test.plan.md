# Remove confirm_test.py - Plan Implementacji

## Szczegóły Zadania

| Pole | Wartość |
|---|---|
| Tytuł | Usunąć `confirm_test.py` — skrypt z 0% coverage, nieużywany w CI |
| Opis | Plik `confirm_test.py` to jednorazowy skrypt walidacyjny (multi-seed confirmation run), który powiela funkcjonalność `tuning_test.py`, nie jest używany w CI i ma 0% pokrycia testami. Należy go usunąć z repozytorium i oczyścić referencje w konfiguracji. |
| Priorytet | Niski |
| Powiązany Research | Brak — zadanie wynika z audytu pokrycia kodu |

## Proponowane Rozwiązanie

Usunąć plik `confirm_test.py` z repozytorium oraz wyczyścić referencję do niego w `sonar-project.properties`. Plik `tuning_test.py` w pełni zastępuje jego funkcjonalność (multi-seed validation) i dodatkowo korzysta z wydzielonych helperów z `utils/training.py` (`run_episode`, `compute_avg100`), podczas gdy `confirm_test.py` zawiera zduplikowaną, ręcznie pisaną pętlę treningową.

## Uzasadnienie Rozwiązania

### Wybrane podejście

Usunięcie pliku zamiast dodania do `.gitignore`. Plik nie zawiera wartości do zachowania — jest starszą, zduplikowaną wersją logiki obecnej w `tuning_test.py`.

### Porównanie z alternatywami

| Kryterium | Usunięcie pliku | Dodanie do .gitignore |
|---|---|---|
| Czystość repozytorium | Plik znika z historii przyszłych commitów | Plik dalej istnieje lokalnie, gitignore go nie usunie z tracked files |
| Skuteczność | Natychmiastowe usunięcie martwego kodu | Wymaga dodatkowego `git rm --cached` aby plik przestał być śledzony |
| Odwracalność | Odzyskanie z historii git | Brak potrzeby odzyskiwania |
| Ryzyko | Brak — plik nie jest importowany ani uruchamiany przez żaden inny moduł | Brak |

### Dlaczego odrzucono alternatywy
- **Dodanie do `.gitignore`**: `.gitignore` nie usuwa plików już śledzonych przez git. Wymagałoby to dodatkowego kroku `git rm --cached confirm_test.py`. Ponadto plik nie ma wartości lokalnej — `tuning_test.py` w pełni go zastępuje. Dodanie do `.gitignore` zaciemnia repozytorium niepotrzebną regułą.

## Model C4

Nie dotyczy — zadanie obejmuje usunięcie jednego pliku i aktualizację konfiguracji.

## Rejestry Decyzji Architektonicznych (ADR)

Nie dotyczy.

## Analiza Aktualnej Implementacji

### Już Zaimplementowane
- `tuning_test.py` — skrypt multi-seed validation korzystający z `utils/training.py` (helperów `run_episode`, `compute_avg100`). W pełni zastępuje `confirm_test.py`.
- `sonar-project.properties` — konfiguracja SonarCloud z listą wykluczeń pokrycia zawierającą `confirm_test.py`.

### Do Modyfikacji
- `sonar-project.properties` — usunąć `confirm_test.py` z `sonar.coverage.exclusions` (plik nie będzie już istniał).
- `CHANGELOG.md` — dodać wpis o usunięciu w sekcji `[Unreleased]`.

### Do Utworzenia
Brak.

## Otwarte Pytania

| # | Pytanie | Odpowiedź | Status |
|---|---------|-----------|--------|
| 1 | Czy `confirm_test.py` jest importowany lub wywoływany przez inny moduł? | Nie — brak referencji w CI, README, pyproject.toml ani w żadnym innym pliku projektu. | ✅ Rozwiązane |
| 2 | Czy `tuning_test.py` pokrywa tę samą funkcjonalność? | Tak — oba skrypty uruchamiają multi-seed training loop i raportują success rate. `tuning_test.py` jest nowszą wersją korzystającą z `utils/training.py`. | ✅ Rozwiązane |

## Plan Implementacji

### Faza 1: Usunięcie pliku i oczyszczenie konfiguracji

#### Zadanie 1.1 - [DELETE] Usunąć `confirm_test.py`
**Opis**: Usunąć plik `confirm_test.py` z repozytorium.

**Definicja Ukończenia (Definition of Done)**:
- [x] Plik `confirm_test.py` nie istnieje w drzewie roboczym
- [x] `git status` pokazuje plik jako usunięty (staged deletion)

#### Zadanie 1.2 - [MODIFY] Wyczyścić referencję w `sonar-project.properties`
**Opis**: Usunąć `confirm_test.py` z listy `sonar.coverage.exclusions` w `sonar-project.properties`, ponieważ plik nie będzie już istniał.

**Definicja Ukończenia (Definition of Done)**:
- [x] Linia `sonar.coverage.exclusions` w `sonar-project.properties` nie zawiera `confirm_test.py`
- [x] Pozostałe wykluczenia (`train.py`, `evaluate.py`, `play.py`, `tuning_test.py`, `version.py`, `utils/analyze.py`, `models/cnn_dqn_network.py`) pozostają nienaruszone

### Faza 2: Aktualizacja dokumentacji

#### Zadanie 2.1 - [MODIFY] Dodać wpis do `CHANGELOG.md`
**Opis**: Dodać wpis w sekcji `[Unreleased]` → `### Removed` opisujący usunięcie `confirm_test.py`.

**Definicja Ukończenia (Definition of Done)**:
- [x] Sekcja `[Unreleased]` w `CHANGELOG.md` zawiera podsekcję `### Removed` z wpisem o usunięciu `confirm_test.py`
- [x] Wpis zawiera uzasadnienie (zduplikowana funkcjonalność z `tuning_test.py`, 0% coverage, brak użycia w CI)

### Faza 3: Code Review

#### Zadanie 3.1 - Code Review przez agenta `code-reviewer`
**Opis**: Przegląd zmian przez agenta `code-reviewer` w celu weryfikacji, że usunięcie jest kompletne i nie pozostały osierocone referencje.

**Definicja Ukończenia (Definition of Done)**:
- [x] Code review potwierdza brak referencji do `confirm_test.py` w repozytorium
- [x] Code review potwierdza poprawność zmian w `sonar-project.properties`
- [x] Code review potwierdza poprawność wpisu w `CHANGELOG.md`

## Aspekty Bezpieczeństwa

Nie dotyczy — zadanie polega na usunięciu nieużywanego pliku.

## Strategia Testowania

### Piramida testów

Nie dotyczy — zadanie nie wprowadza nowego kodu. Istniejący test suite (`pytest tests/`) nie obejmuje `confirm_test.py` i nie wymaga zmian.

### Podejście do testowania
- [x] Weryfikacja, że `pytest tests/` przechodzi bez błędów po usunięciu (regresja)
- [x] Weryfikacja, że `ruff check . --select E9,F63,F7,F82` przechodzi (lint CI)

### Testy wydajnościowe
Nie dotyczy.

### Testy dostępności
Nie dotyczy.

### Testy architektoniczne
Nie dotyczy.

### Testy mutacyjne
Nie dotyczy.

## Zapewnienie Jakości

- [x] Plik `confirm_test.py` nie istnieje w repozytorium
- [x] `sonar-project.properties` nie zawiera referencji do `confirm_test.py`
- [x] `CHANGELOG.md` zawiera wpis o usunięciu
- [x] `pytest tests/` przechodzi bez błędów
- [x] `ruff check . --select E9,F63,F7,F82` przechodzi
- [ ] CI pipeline przechodzi (lint, unit tests, smoke tests)

## Usprawnienia (Poza Zakresem)

- Rozważyć analogiczne potraktowanie `tuning_test.py` — jest również wykluczony z coverage i nie jest częścią CI. Jeśli nie jest regularnie używany, można go dodać do `.gitignore` lub oznaczyć w README jako narzędzie deweloperskie.
- Ujednolicić politykę dot. skryptów deweloperskich — osobny katalog `scripts/` z `.gitignore` lub jawne wykluczenie w CI/Sonar.

## Changelog

| Wersja | Data | Opis |
|--------|------|------|
| 1.0 | 2026-04-07 | Plan utworzony |
| 1.1 | 2026-04-07 | Implementacja zakończona. Code review wykonany przez agenta `code-reviewer`. Wykryta drobna niezgodność kolejności sekcji w `CHANGELOG.md` (`### Removed` przed `### Changed`) — naprawione przywracając kolejność Keep a Changelog: `Changed → Added → Removed`. |

## Code Review Findings

**Werdykt**: Zmiany kompletne i poprawne. Implementacja w pełni realizuje plan.

| Obszar | Status | Szczegóły |
|--------|--------|-----------|
| Plik usunięty | ✅ | `confirm_test.py` nie istnieje w drzewie roboczym |
| Referencje w aktywnych plikach | ✅ | Brak referencji w `.py`, `.yml`, `.toml`, `.properties`, `README.md` |
| `sonar-project.properties` | ✅ | `confirm_test.py` usunięty, 7 pozostałych wpisów nienaruszone |
| `CHANGELOG.md` — zawartość | ✅ | Wpis kompletny i dobrze uzasadniony |
| `CHANGELOG.md` — kolejność sekcji | ✅ | Naprawione: `Changed → Added → Removed` (Keep a Changelog) |
| Testy / lint | ✅ | 107 testów przechodzi, ruff clean |

**Referencje w archiwalnych planach** (`.github/Issue/`): Dopasowania `confirm_test` znalezione w 4 archiwalnych plikach planu — wszystkie akceptowalne jako historyczna dokumentacja, nie wpływają na działanie projektu.
