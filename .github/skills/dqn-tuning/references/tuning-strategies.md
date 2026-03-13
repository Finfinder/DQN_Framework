# Strategie tuningu hiperparametrów DQN

## Zasady ogólne

1. **Maksymalnie 2-3 parametry na iterację** — inaczej nie wiadomo, co zadziałało
2. **Małe kroki** — mnożnik ×0.5 lub ×2 dla lr/tau, delta ±0.005 dla decay
3. **Nie ruszaj seeda** — zachowaj porównywalność
4. **Cofaj regresję** — jeśli wynik gorszy, cofnij zmiany i spróbuj czegoś innego

## Strategie per diagnoza

### S1: Agent nie uczy się wcale

**Objawy**: avg100 płaski, td_error nie spada

**Krok 1** — Zwiększ learning rate:
```python
"lr": aktualne * 2  # np. 0.0005 → 0.001
```

**Krok 2** — Jeśli nadal nie pomaga, zwiększ sieć:
```python
"hidden_layers": [256, 256]  # z [128, 128]
```

**Krok 3** — Sprawdź czy PER nie przeszkadza:
```python
"use_per": False  # wyłącz PER i porównaj
```

---

### S2: Wczesne plateau (epsilon za szybko spada)

**Objawy**: avg100 rośnie w pierwszych 20% epizodów, potem stagnacja; epsilon blisko epsilon_min

**Krok 1** — Spowolnij spadek epsilon:
```python
"epsilon_decay": aktualne + 0.003  # np. 0.985 → 0.988
```

**Krok 2** — Dodaj bufor treningowy:
```python
"num_episodes": aktualne * 1.5  # więcej czasu na naukę
```

**Krok 3** — Obniż epsilon_min (jeśli jest > 0.01):
```python
"epsilon_min": 0.005  # pozwól na minimalną eksplorację nawet na końcu
```

---

### S3: Niestabilny trening (oscylacje)

**Objawy**: avg100 skacze góra-dół, td_error chaotyczny

**Krok 1** — Zmniejsz learning rate:
```python
"lr": aktualne * 0.5  # np. 0.001 → 0.0005
```

**Krok 2** — Zmniejsz tau (wolniejszy soft update):
```python
"tau": aktualne * 0.5  # np. 0.01 → 0.005
```

**Krok 3** — Zwiększ batch size:
```python
"batch_size": aktualne * 2  # np. 64 → 128
```

---

### S4: Wolna konwergencja

**Objawy**: avg100 rośnie stabilnie ale powoli; nie osiąga celu w danym budżecie

**Krok 1** — Częstsze updaty:
```python
"train_every_steps": max(1, aktualne - 1)  # np. 4 → 2
```

**Krok 2** — Większy replay buffer:
```python
"memory_size": aktualne * 2  # więcej doświadczeń do sampli
```

**Krok 3** — Więcej epizodów:
```python
"num_episodes": aktualne + 500
```

---

### S5: Eval gorszy niż training (policy misalignment)

**Objawy**: eval mean_reward << avg100; policy potrzebuje noise żeby dobrze działać

**Krok 1** — Mniejszy epsilon_min:
```python
"epsilon_min": 0.005  # z 0.01
```

**Krok 2** — Wolniejszy target update:
```python
"tau": aktualne * 0.5  # stabilniejsza policy
```

**Krok 3** — Dłuższy trening z niskim epsilon:
```python
"num_episodes": aktualne + 500
"epsilon_decay": aktualne + 0.001  # wolniej schodzi
```

---

### S6: Dobry wynik ale niestabilny (wysoki std)

**Objawy**: eval mean_reward bliski celu, ale std_reward wysoki

**Krok 1** — Większy batch:
```python
"batch_size": aktualne * 2  # stabilniejsze gradienty
```

**Krok 2** — Mniejszy tau:
```python
"tau": aktualne * 0.5
```

**Krok 3** — Dłuższy trening:
```python
"num_episodes": aktualne + 300
```

---

### S7: PER zdominowany przez pojedyncze priorytety

**Objawy**: priority_mean wysoki i nie spada; is_weight_mean daleki od 1.0

**Krok 1** — Mniej agresywna priorytetyzacja:
```python
"per_alpha": aktualne - 0.1  # np. 0.7 → 0.6
```

**Krok 2** — Szybszy annealing beta:
```python
"per_beta_frames": aktualne * 0.7  # szybciej wyrównaj wagi IS
```

---

### S8: Architektura — Dueling vs Standard

**Kiedy włączyć Dueling** (`use_dueling: True`):
- Środowiska z wieloma akcjami gdzie nie wszystkie mają znaczenie w danym stanie
- Gdy standard DQN osiąga plateau

**Kiedy wyłączyć Dueling** (`use_dueling: False`):
- Proste środowiska (np. CartPole z 2 akcjami)
- Gdy Dueling nie poprawia wyników po kilku iteracjach

**Kiedy włączyć PER** (`use_per: True`):
- Sparse rewards (np. MountainCar)
- Gdy uniform sampling daje wolną konwergencję

**Kiedy wyłączyć PER** (`use_per: False`):
- Dense rewards (np. CartPole)
- Gdy PER powoduje niestabilność

## Kolejność eksperymentów

Sugerowana kolejność eksploracji parametrów (od najbardziej wpływowych):

1. `lr` — najsilniejszy wpływ na uczenie
2. `epsilon_decay` — kontroluje balans eksploracja/eksploatacja
3. `tau` — stabilność target network
4. `batch_size` — stabilność gradientów
5. `hidden_layers` — pojemność sieci
6. `use_dueling` / `use_per` — architektura
7. `gamma` — horyzont planowania (rzadko trzeba ruszać)
8. `memory_size` — zwykle wystarczy domyślna wartość

## Przykładowe ścieżki optymalizacji

### CartPole-v1 (cel: eval mean > 490)
```
Iteracja 1: baseline → eval ~350
Iteracja 2: epsilon_decay 0.985→0.990, lr 0.001→0.0005 → eval ~430
Iteracja 3: tau 0.01→0.005, num_episodes 800→1000 → eval ~480
Iteracja 4: batch_size 64→128 → eval ~495 ✓
```

### MountainCar-v0 (cel: eval mean > -120)
```
Iteracja 1: baseline → eval ~-195
Iteracja 2: use_per True, per_alpha 0.7 → eval ~-160
Iteracja 3: lr 0.0005→0.001, epsilon_decay 0.997→0.998 → eval ~-130
Iteracja 4: memory_size 50000→80000, num_episodes 2500→3000 → eval ~-115 ✓
```
