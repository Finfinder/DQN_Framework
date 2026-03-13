# Przewodnik analizy metryk DQN

## Wizualizacja — TensorBoard

Do porównania iteracji użyj TensorBoard:

```bash
tensorboard --logdir logs
```

Każdy run treningu tworzy osobny katalog w `logs/` z pełnym zestawem metryk. TensorBoard automatycznie nakłada przebiegi, co pozwala wizualnie porównać:

| Panel TensorBoard | Co pokazuje |
|-------------------|-------------|
| `eval/mean_reward` | Główna metryka sukcesu — greedy policy (epsilon=0) |
| `eval/std_reward` | Stabilność policy |
| `episode/avg100` | Trend uczenia (średnia z 100 epizodów) |
| `episode/epsilon` | Tempo spadku eksploracji |
| `episode/reward` | Surowy reward per epizod |
| `train/loss` | Strata treningowa |
| `train/td_error_mean` | Błąd predykcji wartości |
| `train/q_mean` | Średnia estymowana Q-wartość |
| `train/is_weight_mean` | Balans IS weights (PER) |
| `train/priority_mean` | Średni priorytet w buforze (PER) |

**Tip**: Użyj filtra regex w TensorBoard (np. `CartPole`) żeby zobaczyć tylko runy danego środowiska.

## Formaty plików CSV

### Training CSV (`metrics/<env>_<model>_<timestamp>.csv`)

| Kolumna | Opis | Jak interpretować |
|---------|------|-------------------|
| `episode` | Numer epizodu | Oś X — postęp treningu |
| `reward` | Sumaryczny reward w epizodzie | Surowy wynik; zaszumiony — patrz na avg100 |
| `avg100` | Średnia z ostatnich 100 epizodów | **Trend uczenia** — kluczowa metryka treningowa |
| `epsilon` | Aktualny epsilon | Tempo eksploracji — spada exponencjalnie |
| `beta` | IS weight annealing (PER) | Powinno rosnąć od ~0.4 do ~1.0 |
| `is_weight_mean` | Średnia IS weight w epizodzie | Bliski 1.0 = dobrze zbalansowane priorytety |
| `td_error_mean` | Średni |TD error| w epizodzie | Spadek = agent lepiej przewiduje wartości |
| `priority_mean` | Średni priorytet w buforze (PER) | Spadek = mniej zaskakujących przejść |

### Eval CSV (`metrics/<env>_<model>_<timestamp>_eval.csv`)

| Kolumna | Opis | Jak interpretować |
|---------|------|-------------------|
| `episode` | Epizod treningowy w którym wykonano eval | Punkty kontrolne |
| `mean_reward` | Średni reward z N greedy epizodów | **Główna metryka sukcesu** |
| `std_reward` | Odchylenie standardowe | Niższe = bardziej stabilna policy |
| `min_reward` | Najgorszy epizod eval | Dno wydajności — ważne dla reliability |
| `max_reward` | Najlepszy epizod eval | Sufit wydajności |

## Jak czytać metryki

### 1. Overview — szybka diagnoza

Wczytaj ostatni plik CSV i sprawdź:

```
Początek treningu (pierwsze 10%):
  - avg100 — punkt startowy (random policy)
  - epsilon — powinno być ~1.0

Środek treningu (30-70%):
  - avg100 — czy rośnie? (poprawa)
  - epsilon — czy spada? (przejście na eksploatację)
  - td_error_mean — czy spada? (lepsze predykcje)

Koniec treningu (ostatnie 10%):
  - avg100 — finalna wydajność treningowa
  - epsilon — bliskie epsilon_min
  - td_error_mean — stabilne lub niskie
```

### 2. Wzorce diagnostyczne

#### Wzorzec A: Brak uczenia
```
avg100: płaski od początku do końca
td_error_mean: nie spada lub rośnie
```
**Przyczyna**: lr za niski, sieć za mała, lub reward shaping potrzebny
**Rozwiązanie**: ↑ lr, ↑ hidden_layers, sprawdź train_reward

#### Wzorzec B: Wczesne plateau
```
avg100: rośnie szybko w pierwszych 20% epizodów, potem stagnacja
epsilon: osiągnął epsilon_min za wcześnie
```
**Przyczyna**: epsilon spada za szybko — agent przestał eksplorować
**Rozwiązanie**: ↑ epsilon_decay (bliżej 1.0), ↑ num_episodes

#### Wzorzec C: Niestabilność
```
avg100: oscyluje — rośnie i spada naprzemiennie
td_error_mean: chaotyczny, nie konwerguje
```
**Przyczyna**: lr za wysoki, tau za duży, batch za mały
**Rozwiązanie**: ↓ lr, ↓ tau, ↑ batch_size

#### Wzorzec D: Wolna konwergencja
```
avg100: rośnie powoli ale stabilnie
Po num_episodes epizodów nadal nie osiągnął celu
```
**Przyczyna**: za mało updatów, za mały bufor, za mało epizodów
**Rozwiązanie**: ↓ train_every_steps, ↑ memory_size, ↑ num_episodes

#### Wzorzec E: Eval << Avg100 (policy misalignment)
```
eval mean_reward znacząco gorszy niż avg100
eval std_reward wysoki
```
**Przyczyna**: policy działa dobrze z noise (epsilon), ale źle bez niego
**Rozwiązanie**: ↓ epsilon_min, ↓ tau (stabilniejsza target network), dłuższy trening

#### Wzorzec F: Dobry wynik, wysoki std
```
eval mean_reward blisko celu
eval std_reward > 20% mean_reward
```
**Przyczyna**: policy niestabilna — niektóre epizody dobre, inne złe
**Rozwiązanie**: ↑ batch_size, ↓ tau, dłuższy trening

### 3. Porównanie między iteracjami

Porównuj zawsze te same metryki z tych samych punktów treningu:

```
Iteracja N:   eval mean_reward = X1,  std = S1,  avg100(final) = A1
Iteracja N+1: eval mean_reward = X2,  std = S2,  avg100(final) = A2

Delta eval:  X2 - X1  (>0 = poprawa)
Delta std:   S2 - S1  (<0 = poprawa stabilności)
```

### 4. Referencyjne wartości per środowisko

| Środowisko | Random policy | Dobry wynik | Solved |
|------------|---------------|-------------|--------|
| CartPole-v1 | ~20 | >400 | 500 (max) |
| MountainCar-v0 | -200 | >-150 | >-100 |
| Acrobot-v1 | -500 | >-100 | >-80 |
