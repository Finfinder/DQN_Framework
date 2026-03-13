---
name: dqn-tuning
description: "**WORKFLOW SKILL** — Automatyczna optymalizacja modeli RL DQN. USE FOR: analiza metryk treningowych/ewaluacyjnych z CSV i TensorBoard, porównywanie runów w pandas/Jupyter, dostrajanie hiperparametrów (lr, epsilon_decay, batch_size, tau, gamma, hidden_layers, memory_size, num_episodes, use_dueling, use_per), uruchamianie kolejnych iteracji treningu i ewaluacji aż do osiągnięcia celu. DO NOT USE FOR: trening pojedynczy bez analizy, ręczna zmiana jednego parametru, debugowanie kodu agenta."
argument-hint: "Podaj środowisko (np. CartPole-v1) i cel (np. eval mean_reward > 450)"
---

# Optymalizacja DQN — Iteracyjny Tuning Hiperparametrów

## Kiedy używać

- Chcesz automatycznie poprawić wyniki modelu DQN dla danego środowiska
- Masz wyniki z poprzednich treningów (CSV w `metrics/`) i chcesz je przeanalizować
- Potrzebujesz systematycznego podejścia do tuningu hiperparametrów
- Chcesz uruchomić pętlę: trening → ewaluacja → analiza → korekta → ponowny trening

## Wymagania wstępne

- Aktywne środowisko Python z `torch`, `gymnasium`, `numpy`, `matplotlib`, `tensorboard`, `pandas`, `jupyter`
- Projekt DQN Framework z plikami: `train.py`, `evaluate.py`, `config/config.py`, `utils/analyze.py`, `analysis.ipynb`
- Zdefiniowane środowisko w `Config.ENV_CONFIG` (np. `CartPole-v1`, `MountainCar-v0`, `Acrobot-v1`)

## Procedura

### Krok 0: Inicjalizacja — Określ cel i środowisko

Przed rozpoczęciem ustal:

1. **Środowisko** — np. `CartPole-v1`, `MountainCar-v0`, `Acrobot-v1`
2. **Metryka celu** — dwie metryki decydują o sukcesie:
   - **Główna**: `eval/mean_reward` (greedy policy, epsilon=0) z ewaluacji
   - **Wspomagająca**: `avg100` (średnia z ostatnich 100 epizodów treningowych)
3. **Wartość docelowa** — np. `eval mean_reward > 450` dla CartPole-v1, `eval mean_reward > -110` dla MountainCar-v0
4. **Maksymalna liczba iteracji** — domyślnie 5, żeby uniknąć nieskończonej pętli

Zapisz te parametry — będą potrzebne w każdej iteracji.

### Krok 1: Zbierz stan bazowy (Baseline)

**1.1. Sprawdź istniejące metryki CSV**

Przeczytaj ostatnie pliki CSV z `metrics/` dla danego środowiska:

- Pliki treningowe: `metrics/<env>_<model>_<timestamp>.csv`
  - Kolumny: `episode, reward, avg100, epsilon, beta, is_weight_mean, td_error_mean, priority_mean`
- Pliki ewaluacyjne: `metrics/<env>_<model>_<timestamp>_eval.csv`
  - Kolumny: `episode, mean_reward, std_reward, min_reward, max_reward`

Do szybkiej analizy programatycznej preferuj:

```python
from utils.analyze import list_runs, compare_runs, diagnose

compare_runs("CartPole-v1", "train")
diagnose("CartPole-v1")
```

Do interaktywnej eksploracji i wykresów użyj notebooka:

```bash
jupyter notebook analysis.ipynb
```

**1.2. Jeśli brak metryk — uruchom baseline trening**

```bash
python train.py <env_name>
```

**1.3. Uruchom ewaluację aktualnego modelu**

```bash
python evaluate.py <env_name> --episodes 100
```

Wyniki ewaluacji zostaną wydrukowane na stdout **i** zapisane do CSV: `metrics/<env>_<model>_standalone_eval_<timestamp>.csv`.

Zapisz wyniki baseline: `eval mean_reward`, `eval std_reward`, `avg100`.

**1.4. Wizualizacja w TensorBoard (opcjonalnie)**

Do porównania przebiegów między iteracjami użyj TensorBoard:

```bash
tensorboard --logdir logs
```

Kluczowe panele:
- `eval/mean_reward` — główna metryka sukcesu (greedy policy)
- `episode/avg100` — trend uczenia
- `episode/epsilon` — tempo eksploracji
- `train/loss`, `train/td_error_mean` — stabilność uczenia
- `train/is_weight_mean`, `train/priority_mean` — diagnostyka PER

TensorBoard automatycznie nakłada przebiegi z różnych runów, co ułatwia porównanie iteracji.

### Krok 2: Analiza metryk

Przeanalizuj zebrane dane CSV. Skorzystaj z [przewodnika analizy](./references/analysis-guide.md) aby zdiagnozować problemy.

**Kluczowe pytania diagnostyczne:**

| Pytanie | Gdzie szukać odpowiedzi |
|---------|------------------------|
| Czy agent się uczy? | `avg100` rośnie w czasie? |
| Jak szybko epsilon spada? | Kolumna `epsilon` — osiąga `epsilon_min` za wcześnie/za późno? |
| Czy TD error jest stabilny? | `td_error_mean` — nie rośnie nieograniczenie? |
| Czy IS weights są zrównoważone? | `is_weight_mean` bliski 1.0 pod koniec treningu? |
| Jak duża wariancja eval? | `std_reward` z eval CSV — wysoka = niestabilna policy |
| Czy eval >> avg100? | Greedy policy lepsza niż epsilon-greedy = dobry znak |
| Czy eval << avg100? | Policy overfittuje do exploration noise = zły znak |

### Krok 3: Dobierz strategię tuningu

Na podstawie diagnozy z Kroku 2, wybierz odpowiednią strategię z [katalogu strategii](./references/tuning-strategies.md).

**Szybki wybór:**

| Diagnoza | Parametry do zmiany | Kierunek |
|----------|---------------------|----------|
| Agent nie uczy się wcale | `lr` ↑, `hidden_layers` ↑ | Zwiększ pojemność |
| Uczy się, ale plateau za wcześnie | `epsilon_decay` ↑ (bliżej 1.0), `num_episodes` ↑ | Więcej eksploracji |
| Uczy się, ale niestabilnie | `lr` ↓, `tau` ↓, `batch_size` ↑ | Stabilizuj |
| Eval dużo gorszy niż avg100 | `epsilon_min` ↓, `tau` ↓ | Policy alignment |
| Wysoki TD error | `lr` ↓, `gamma` ↓ | Mniejsze cele |
| Wolna konwergencja | `train_every_steps` ↓, `memory_size` ↑ | Więcej updatów |
| Dobry wynik, ale wysoki std | `batch_size` ↑, `tau` ↓ | Stabilizuj |
| PER priorities zdominowane | `per_alpha` ↓ | Mniej priorytetyzacji |

### Krok 4: Zastosuj zmiany w konfiguracji

Zmodyfikuj odpowiedni blok w `config/config.py` → `ENV_CONFIG["<env_name>"]`.

**WAŻNE UWAGI:**

- Ścieżki modeli mają dynamiczny suffix (`_dueling` / `_standard`) dodawany automatycznie przez `Config` na podstawie `use_dueling` — nie dodawaj go ręcznie w `model_path`.
- `CartPole-v1` stosuje reward shaping: przejścia `terminated` dostają karę `-10.0` w `train.py`. Przez to `avg100` w treningu może spadać poniżej zera — to normalne i nie oznacza błędu.

**ZASADY BEZPIECZEŃSTWA:**

- Zmieniaj **maksymalnie 2-3 parametry** na iterację — inaczej nie wiesz co pomogło
- Rób **małe kroki**: lr ×0.5 lub ×2, nie ×10
- **Nie zmieniaj** `seed` — zachowaj porównywalność między iteracjami
- Zanotuj co zmieniłeś i dlaczego (w output do użytkownika)

### Krok 5: Uruchom trening

```bash
python train.py <env_name>
```

Monitoruj wynik w konsoli. Po zakończeniu treningu automatycznie powstanie:
- Plik wag: `<model_path>` (nadpisany jeśli lepszy)
- CSV treningowy: `metrics/<env>_<model>_<timestamp>.csv`
- CSV ewaluacyjny: `metrics/<env>_<model>_<timestamp>_eval.csv`
- Logi TensorBoard: `logs/<env><suffix>_<timestamp>/`

Do wizualnego porównania przebiegów użyj `tensorboard --logdir logs`.

### Krok 6: Ewaluacja po treningu

```bash
python evaluate.py <env_name> --episodes 100
```

Wyniki zostaną wydrukowane na stdout i zapisane do `metrics/<env>_<model>_standalone_eval_<timestamp>.csv`.

Porównaj wyniki z baseline i poprzednią iteracją:
- `eval mean_reward` — główna metryka
- `eval std_reward` — stabilność
- Poprawa względem baseline (%)

### Krok 7: Decyzja — kontynuować czy kończyć?

```
JEŚLI eval_mean_reward >= cel_docelowy:
    → SUKCES. Zakończ. Podsumuj wyniki.

JEŚLI iteracja >= max_iteracji:
    → STOP. Podsumuj najlepszy wynik. Zasugeruj dalsze kroki.

JEŚLI eval_mean_reward > poprzednia_iteracja:
    → POSTĘP. Wróć do Kroku 2 z nowymi metrykami.

JEŚLI eval_mean_reward <= poprzednia_iteracja:
    → REGRESJA. Cofnij ostatnie zmiany. Spróbuj innej strategii. Wróć do Kroku 3.
```

### Krok 8: Podsumowanie końcowe

Po osiągnięciu celu lub wyczerpaniu iteracji, przedstaw:

1. **Tabela iteracji**: nr iteracji → zmiany parametrów → eval mean_reward → delta
2. **Najlepsza konfiguracja**: pełny zrzut finalnych parametrów z `ENV_CONFIG`
3. **Wnioski**: co zadziałało, co nie, jakie wzorce zaobserwowano
4. **Rekomendacje**: dalsze kroki optymalizacji (jeśli cel nie osiągnięty)

## Parametry do optymalizacji

| Parametr | Zakres typowy | Wpływ |
|----------|---------------|-------|
| `lr` | 1e-4 — 1e-2 | Szybkość uczenia; za duże = niestabilność, za małe = wolna konwergencja |
| `epsilon_decay` | 0.98 — 0.999 | Tempo przejścia z eksploracji na eksploatację |
| `hidden_layers` | [32,32] — [256,256] | Pojemność sieci; za mała = underfitting, za duża = wolne + overfitting |
| `batch_size` | 32 — 256 | Stabilność gradientów; większy = stabilniej, wolniej |
| `tau` | 0.001 — 0.05 | Soft update target network; mniejszy = stabilniej, wolniej |
| `gamma` | 0.9 — 0.999 | Discount factor; mniejszy = skupienie na krótkim horyzoncie |
| `memory_size` | 5000 — 100000 | Rozmiar replay buffer; większy = więcej doświadczeń |
| `num_episodes` | 500 — 5000 | Budżet treningowy |
| `train_every_steps` | 1 — 8 | Co ile kroków update sieci; mniejszy = częstsze updaty, wolniejszy trening |
| `use_dueling` | true/false | Architektura Dueling DQN |
| `use_per` | true/false | Prioritized Experience Replay |
| `per_alpha` | 0.0 — 1.0 | Siła priorytetyzacji PER; 0 = uniform, 1 = pełna priorytetyzacja |
| `per_beta_frames` | 10000 — 200000 | Kroki do annealingu IS beta do 1.0; mniejszy = szybsza korekcja wag |

## Narzędzia

- **TensorBoard** (`tensorboard --logdir logs`) — wizualizacja i porównanie przebiegów między iteracjami. Każdy run treningu automatycznie loguje metryki treningowe (`train/*`, `episode/*`) i ewaluacyjne (`eval/*`).
- **Pandas + utils/analyze.py** — programatyczna analiza CSV, porównanie runów, diagnoza ostatniego treningu.
- **Jupyter Notebook** (`analysis.ipynb`) — interaktywna analiza, wykresy `avg100`, `epsilon`, `eval/mean_reward`, eksport podsumowań do CSV.
- **CSV** (`metrics/`) — surowe dane wejściowe dla pandas/notebooka.
- **evaluate.py** — standalone ewaluacja greedy policy po treningu.

## Referencje

- [Przewodnik analizy metryk](./references/analysis-guide.md) — jak czytać i interpretować CSV oraz TensorBoard
- [Strategie tuningu](./references/tuning-strategies.md) — szczegółowe strategie dla każdej diagnozy
