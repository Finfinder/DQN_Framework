# SonarCloud New Code Period Previous Version - Wynik analizy

## Szczegóły zadania

| Pole | Wartość |
| --- | --- |
| Jira ID | Nie dotyczy |
| Tytuł | Rozważyć zmianę New Code Period w SonarCloud na "Previous Version" z tagami semver |
| Opis | Ocenić, czy dla `DQN_Framework` warto zmienić definicję SonarCloud New Code Period na "Previous Version" i powiązać ją z istniejącym workflow semver/release opartym o tagi `vX.Y.Z` oraz branche wersyjne `X.Y.Z`. |
| Priorytet | Średni |
| Zgłaszający | — |
| Data utworzenia | 2026-05-13 |
| Termin realizacji | — |
| Etykiety | sonarcloud, quality-gate, semver, ci-cd, release |
| Szacowany nakład pracy | S/M - niewielka zmiana konfiguracyjna, ale wymagająca decyzji procesowej |
| Złożoność analizy rozwiązań | M |

## Wpływ biznesowy

Zmiana definicji New Code Period może poprawić przewidywalność Quality Gate i lepiej powiązać ocenę jakości z cyklami release. Obecne problemy SonarCloud w historii projektu dotyczyły przede wszystkim metryk new code: coverage, duplications oraz reliability rating. Jeżeli new code będzie wyznaczane względem poprzedniej wersji semver, zespół otrzyma bardziej zrozumiały sygnał: "czy zmiany w bieżącym cyklu wersji są gotowe jakościowo", zamiast sygnału zależnego od arbitralnego okna czasowego.

Dla projektu z jawnie utrzymywanym `version.py`, README badge, `CHANGELOG.md`, tagami release `vX.Y.Z` i branchami wersyjnymi `X.Y.Z` taka zmiana wzmacnia spójność między release managementem a kontrolą jakości. Ryzykiem biznesowym jest fałszywe poczucie poprawy jakości, jeśli SonarCloud zostanie przełączony na "Previous Version", ale skany nie będą konsekwentnie przekazywać poprawnej wersji projektu.

## Zebrane informacje

### Baza wiedzy i narzędzia do zarządzania zadaniami

Nie podano identyfikatora Jira ani linków do Confluence/Figma/PDF. Źródłem wejściowym jest bezpośredni opis zadania oraz wcześniejszy lokalny plan SonarCloud Quality Gate w repozytorium.

Kontekst historyczny z `.github/Issue/Archiwum/2.1.0/sonarcloud-quality-gate-v3.plan.md`:

- Po wcześniejszych naprawach Quality Gate jako usprawnienie poza zakresem zapisano: "Rozważyć zmianę New Code Period w SonarCloud na \"Previous Version\" z tagami semver".
- Wcześniejsze problemy dotyczyły metryk new code, szczególnie `new_coverage` przy progu 80%.
- Zespół traktuje SonarCloud Quality Gate jako element kontroli jakości dla zmian wprowadzanych do projektu.

Ustalenia z dokumentacji SonarCloud:

- SonarCloud rozdziela overall code i new code; domyślny Quality Gate koncentruje się na new code.
- SonarCloud wspiera definicje new code: `Previous version`, `Number of days`, `Specific version`, `Specific date`.
- Przy `Previous version` bieżąca wersja projektu jest określana w czasie analizy.
- Dla projektów bez Maven/Gradle, czyli także dla tego projektu Python/SonarScanner CLI, wersję trzeba jawnie przekazać jako `sonar.projectVersion`.
- Tagi Git `vX.Y.Z` nie są same w sobie wersją projektu dla SonarCloud; są triggerem/procesowym znacznikiem release. Wartość semver przekazywana do `sonar.projectVersion` powinna mieć format `X.Y.Z`.

Ustalenia z konfiguracji SonarCloud przekazanej przez użytkownika 2026-05-13:

- Na poziomie organizacji `finfinder` domyślna definicja new code dla nowych projektów jest ustawiona na `Previous version`.
- Na poziomie projektu `DQN_Framework` definicja new code jest ustawiona na `Previous version`.
- Projekt `DQN_Framework` ma long-lived branches pattern obejmujący `main` oraz branche semver w formacie `X.Y.Z`.
- W SonarCloud widoczne są long-lived branche `main`, `2.1.1` oraz `2.1.0`; branch `2.1.1` ma status Quality Gate `Passed`, a `main` i `2.1.0` mają status `Failed`.
- Ostatnie skany `main` i `2.1.1` zostały wykonane około 8 godzin temu; `main` wskazuje commit `0153d0f0` i Quality Gate `Failed`.
- Dla `main` SonarCloud pokazuje `New code Since about 2 months ago`, `Coverage 55.16%`, próg `>= 80.0%` oraz `281 New Lines to cover`; `New Issues` wynosi `0`, a `Duplications` `0.0%` na `2.9k New Lines`.
- Logi GitHub Actions dla skanów `main` i `2.1.1` pokazują komendę `sonar-scanner -Dsonar.projectBaseDir=.` bez parametru `sonar.projectVersion`, co potwierdza brak raportowania wersji projektu do SonarCloud.
- Logi obu skanów zawierają ostrzeżenie o braku `sonar.python.version`; nie wpływa to bezpośrednio na New Code Period, ale warto uzupełnić konfigurację, aby analiza Pythona była precyzyjna dla wersji używanej w CI.

### Baza kodu

Repozytorium `DQN_Framework` jest projektem Python/PyTorch z istniejącą integracją SonarCloud i dojrzałym workflow semver:

- `sonar-project.properties` zawiera `sonar.projectKey=Finfinder_DQN_Framework`, `sonar.organization=finfinder`, zakres analizy Python, konfigurację testów oraz `sonar.python.coverage.reportPaths=coverage.xml`.
- `.github/workflows/sonar.yml` uruchamia skan SonarCloud na push/PR do `main` i branchy pasujących do wzorca `[0-9]+.[0-9]+.[0-9]+`.
- `.github/workflows/sonar.yml` ma `fetch-depth: 0`, co jest zgodne z zaleceniem SonarSource dla jakości analizy historii/blame.
- Workflow Sonara uruchamia testy z coverage przed skanem, ale obecnie nie przekazuje `sonar.projectVersion`.
- `version.py` zawiera aktualną wersję `2.1.1`.
- README zawiera badge wersji `2.1.1` i badge Quality Gate SonarCloud dla projektu `Finfinder_DQN_Framework`.
- `CHANGELOG.md` deklaruje Semantic Versioning i zawiera historię release `2.0.0`, `2.1.0`, `2.1.1` oraz sekcję `[Unreleased]`.
- `.github/workflows/release.yml` uruchamia release na tagach `v*` i waliduje zgodność wersji względem `github.ref_name`.
- `.github/workflows/open-next-version-branch.yml` po udanym release uruchamia workflow otwierający następny branch wersyjny.
- `.github/workflows/reusable-open-next-version-branch.yml` tworzy branch o nazwie `next_version` i aktualizuje targety wersji.
- `.github/release/next-version.json` wskazuje historycznie `release_version=2.1.0` i `next_version=2.1.1`.
- `.github/versioning/version-targets.json` wskazuje `version.py` jako target wersji z strategią `python_dunder_version`.
- `.github/versioning/readme-targets.json` wskazuje README jako target wersji z strategią `readme_badge`.
- Skrypty wersjonowania normalizują wartości `vX.Y.Z` i `X.Y.Z` do `X.Y.Z`, co pasuje do rozróżnienia między tagiem Git a semver projektu.

Istotne ograniczenie lokalnej analizy:

- Screenshot potwierdza ustawienie `Previous version` na poziomie projektu. Logi ostatnich skanów potwierdzają, że `sonar.projectVersion` nie został przekazany, więc SonarCloud nie ma jawnej wersji projektu potrzebnej do przewidywalnego działania trybu `Previous version`.

### Powiązane linki

Wszelkie przydatne linki do dokumentacji, projektów lub innych zasobów:

- `.github/Issue/sonarcloud-new-code-period-previous-version.solution-research.md` - osobna analiza porównawcza opcji SonarCloud new code definition
- <https://docs.sonarsource.com/sonarqube-cloud/standards/about-new-code> - SonarCloud: Quality standards and new code
- <https://docs.sonarsource.com/sonarqube-cloud/managing-your-projects/project-analysis/configuring-new-code-calculation> - SonarCloud: New code definition i konfiguracja projektu
- <https://docs.sonarsource.com/sonarqube-cloud/analyzing-source-code/ci-based-analysis/github-actions-for-sonarcloud> - SonarCloud: GitHub Actions i parametry skanera
- <https://docs.sonarsource.com/sonarqube-cloud/analyzing-source-code/scanners/sonarscanner-cli> - SonarScanner CLI i `sonar-project.properties`
- <https://github.com/SonarSource/sonarqube-scan-action> - oficjalna akcja SonarQube/SonarCloud Scan dla GitHub Actions
- <https://semver.org/> - specyfikacja Semantic Versioning 2.0.0

### Analiza rozwiązań

- Plik analizy: `.github/Issue/sonarcloud-new-code-period-previous-version.solution-research.md`
- Rekomendowane rozwiązanie: Przejście na `Previous version` w SonarCloud, pod warunkiem konsekwentnego przekazywania `sonar.projectVersion` zgodnego z wersją semver projektu.
- Oceniona złożoność: M

## Aktualny stan implementacji

### Istniejące komponenty

Lista istniejących komponentów, funkcji lub funkcjonalności powiązanych z tym zadaniem:

- `sonar-project.properties` - `sonar-project.properties` - wymaga rozszerzenia o kontrakt wersji albo pozostawienia wersji do przekazania przez workflow
- `.github/workflows/sonar.yml` - `.github/workflows/sonar.yml` - wymaga modyfikacji, jeśli wersja ma być przekazywana przez `args` akcji SonarCloud
- `version.py` - `version.py` - można ponownie użyć jako kanoniczne źródło wartości `X.Y.Z`
- `README.md` - `README.md` - można ponownie użyć jako dodatkowe źródło walidowane przez istniejący kontrakt wersji
- `.github/workflows/release.yml` - `.github/workflows/release.yml` - można ponownie użyć jako źródło procesu release na tagach `v*`
- `.github/workflows/open-next-version-branch.yml` - `.github/workflows/open-next-version-branch.yml` - można ponownie użyć jako element cyklu otwierania nowej wersji
- `.github/workflows/reusable-version-consistency.yml` - `.github/workflows/reusable-version-consistency.yml` - można ponownie użyć jako istniejący gate spójności wersji
- `scripts/validate-version-consistency.ps1` - `scripts/validate-version-consistency.ps1` - można ponownie użyć do potwierdzania semver i normalizacji prefiksu `v`
- `.github/versioning/version-targets.json` - `.github/versioning/version-targets.json` - można ponownie użyć do wskazania kanonicznego targetu wersji
- `.github/Issue/Archiwum/2.1.0/sonarcloud-quality-gate-v3.plan.md` - `.github/Issue/Archiwum/2.1.0/sonarcloud-quality-gate-v3.plan.md` - źródło poprzedniego kontekstu zadania

### Kluczowe pliki i katalogi

- `sonar-project.properties` - konfiguracja identyfikacji projektu, zakresu analizy i coverage dla SonarCloud
- `.github/workflows/sonar.yml` - miejsce uruchamiania testów z coverage i skanu SonarCloud
- `.github/workflows/ci.yml` - osobny gate CI z wersjonowaniem, lintem, testami i smoke testami; istotny jako porównanie coverage command
- `.github/workflows/release.yml` - release na tagach `v*` oraz walidacja `expected-version`
- `.github/workflows/open-next-version-branch.yml` - automatyczne otwarcie następnego brancha wersyjnego
- `.github/workflows/reusable-open-next-version-branch.yml` - logika tworzenia branchy `X.Y.Z` i aktualizacji targetów wersji
- `.github/release/next-version.json` - manifest release/next version
- `.github/versioning/` - deskryptory źródeł wersji dla walidatorów
- `scripts/` - walidatory i automatyzacja wersji
- `CHANGELOG.md` - publiczna historia semver i punkt odniesienia dla release

## Analiza luk

Wszelkie brakujące informacje i luki w opisie zadania wraz z udzielonymi odpowiedziami.

### Pytanie 1

#### Czy sama zmiana SonarCloud New Code Period na "Previous Version" wystarczy?

Nie. Dla projektu Python skanowanego przez SonarScanner CLI konieczne jest przekazywanie `sonar.projectVersion` w analizie. Bez tego SonarCloud nie ma wiarygodnej informacji o bieżącej wersji projektu i tryb "Previous Version" nie będzie zgodny z semver/tagami.

### Pytanie 2

#### Czy tagi `vX.Y.Z` są bezpośrednio używane przez SonarCloud jako wersje projektu?

Nie na podstawie przeanalizowanej dokumentacji i obecnej konfiguracji. Tagi `vX.Y.Z` uruchamiają release workflow i są świetnym procesowym punktem odniesienia, ale dla SonarCloud wersją projektu powinno być jawne `sonar.projectVersion=X.Y.Z`. SemVer potwierdza, że `v1.2.3` jest nazwą taga, a wersją semver jest `1.2.3`.

### Pytanie 3

#### Czy obecny workflow Sonara jest gotowy do trybu "Previous Version"?

Częściowo. Workflow ma poprawne podstawy: `fetch-depth: 0`, skan na `main` i branchach wersyjnych, coverage przed skanem oraz stabilną konfigurację projektu. Brakuje widocznego kontraktu przekazania wersji projektu do SonarCloud.

Po dodatkowej weryfikacji UI wiadomo, że projekt ma `Previous version`, a long-lived branches pattern obejmuje `main` i branche semver. Logi skanów potwierdziły brak `sonar.projectVersion`, więc problem jest po stronie parametrów workflow/skanera, nie po stronie ustawienia New Code w UI.

### Pytanie 4

#### Czy należy użyć `Number of days`, `Previous version` czy `Specific version/date`?

Rekomendacja z analizy rozwiązań: `Previous version`. `Number of days` jest prostsze, ale słabiej pasuje do semver i branchy wersyjnych. `Specific version/date` daje kontrolę, ale zwiększa koszt utrzymania przez Web API i potencjalnie wymaga dodatkowych uprawnień.

### Pytanie 5

#### Jaki jest główny warunek akceptacji dla tej zmiany?

Quality Gate i metryki new code powinny być interpretowane względem bieżącego cyklu wersji semver. W praktyce oznacza to, że po zmianie SonarCloud powinien mieć ustawione `Previous version`, a każda relewantna analiza projektu powinna raportować tę samą wersję, którą walidują istniejące kontrakty `version.py`/README/release.

### Pytanie 6

#### Jakie informacje wymagają jeszcze potwierdzenia przez właściciela SonarCloud/procesu release?

Nie trzeba już potwierdzać `sonar.projectVersion` w ostatnim skanie: logi pokazują, że nie został przekazany. Do decyzji pozostaje, czy osobny skan SonarCloud powinien uruchamiać się także na tagach release, czy wystarczy analiza `main`/branchy wersyjnych po zmianie wersji w repozytorium.
