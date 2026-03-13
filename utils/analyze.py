"""Narzędzia do analizy metryk DQN z plików CSV (pandas)."""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


METRICS_DIR = Path(__file__).resolve().parent.parent / "metrics"


def list_runs(env_name=None, eval_only=False, train_only=False):
    """Zwraca DataFrame z listą dostępnych runów.

    Kolumny: file, env, model, timestamp, type (train/eval)
    """
    rows = []
    for f in sorted(METRICS_DIR.glob("*.csv")):
        name = f.stem
        is_eval = name.endswith("_eval")

        if eval_only and not is_eval:
            continue
        if train_only and is_eval:
            continue

        # Parse: <env>_<model>_<timestamp>[_eval]
        if is_eval:
            name = name[: -len("_eval")]

        # Timestamp is last 15 chars: YYYYMMDD-HHMMSS
        timestamp_str = name[-15:]
        rest = name[: -(15 + 1)]  # strip _timestamp

        # env is up to the first underscore that starts model name (dqn_)
        idx = rest.find("_dqn_")
        if idx == -1:
            continue
        env = rest[:idx]
        model = rest[idx + 1 :]

        try:
            ts = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
        except ValueError:
            continue

        if env_name and env != env_name:
            continue

        # Detect standalone eval files from evaluate.py
        if "_standalone_eval" in model:
            model = model.replace("_standalone_eval", "")
            run_type = "standalone_eval"
        else:
            run_type = "eval" if is_eval else "train"

        rows.append({
            "file": f.name,
            "path": str(f),
            "env": env,
            "model": model,
            "timestamp": ts,
            "type": run_type,
        })

    return pd.DataFrame(rows)


def load_run(file_or_path):
    """Wczytuje pojedynczy plik CSV jako DataFrame."""
    path = Path(file_or_path)
    if not path.is_absolute():
        path = METRICS_DIR / path
    return pd.read_csv(path)


def load_latest(env_name, run_type="train"):
    """Wczytuje najnowszy run dla danego środowiska.

    Args:
        env_name: np. "CartPole-v1"
        run_type: "train" lub "eval"

    Returns:
        (DataFrame, metadata_dict) lub (None, None) jeśli brak
    """
    runs = list_runs(env_name=env_name)
    runs = runs[runs["type"] == run_type]
    if runs.empty:
        return None, None

    latest = runs.sort_values("timestamp").iloc[-1]
    df = load_run(latest["path"])
    return df, latest.to_dict()


def compare_runs(env_name, run_type="train", last_n=None):
    """Porównuje wszystkie runy danego środowiska.

    Returns:
        DataFrame z podsumowaniem każdego runu:
        - timestamp, model
        - final_avg100 (train) lub mean_reward (eval)
        - best_avg100/best_mean_reward
        - num_episodes
    """
    runs = list_runs(env_name=env_name)
    runs = runs[runs["type"] == run_type]

    if runs.empty:
        return pd.DataFrame()

    runs = runs.sort_values("timestamp")
    if last_n:
        runs = runs.tail(last_n)

    summaries = []
    for _, run in runs.iterrows():
        df = load_run(run["path"])
        summary = {
            "timestamp": run["timestamp"],
            "model": run["model"],
            "num_episodes": len(df),
        }

        if run_type == "train":
            summary["final_reward"] = float(df["reward"].iloc[-1])
            summary["final_avg100"] = float(df["avg100"].iloc[-1])
            summary["best_avg100"] = float(df["avg100"].max())
            summary["final_epsilon"] = float(df["epsilon"].iloc[-1])
            if "td_error_mean" in df.columns:
                summary["final_td_error"] = float(df["td_error_mean"].iloc[-1])
        elif run_type == "eval":
            summary["final_mean_reward"] = float(df["mean_reward"].iloc[-1])
            summary["best_mean_reward"] = float(df["mean_reward"].max())
            if "std_reward" in df.columns:
                summary["final_std_reward"] = float(df["std_reward"].iloc[-1])

        summaries.append(summary)

    return pd.DataFrame(summaries)


def run_summary(env_name):
    """Wyświetla pełne podsumowanie dla środowiska — train + eval."""
    print(f"=== {env_name} ===\n")

    train_cmp = compare_runs(env_name, "train")
    if not train_cmp.empty:
        print("TRAINING RUNS:")
        print(train_cmp.to_string(index=False))
        print()

    eval_cmp = compare_runs(env_name, "eval")
    if not eval_cmp.empty:
        print("EVAL RUNS:")
        print(eval_cmp.to_string(index=False))
        print()

    if train_cmp.empty and eval_cmp.empty:
        print("Brak danych.\n")


def diagnose(env_name):
    """Automatyczna diagnoza ostatniego runu — zwraca listę obserwacji."""
    df_train, meta_train = load_latest(env_name, "train")
    if df_train is None:
        return ["Brak danych treningowych dla tego środowiska."]

    observations = []
    n = len(df_train)

    # 1. Trend avg100
    early = df_train["avg100"].iloc[: n // 10].mean() if n >= 10 else df_train["avg100"].iloc[0]
    mid = df_train["avg100"].iloc[n // 3 : 2 * n // 3].mean()
    late = df_train["avg100"].iloc[-n // 10 :].mean() if n >= 10 else df_train["avg100"].iloc[-1]

    scale = max(abs(early), abs(late), 1)
    improve_total = (late - early) / scale
    improve_first = (mid - early) / scale
    improve_second = (late - mid) / scale

    if improve_total < 0.1:
        observations.append("BRAK UCZENIA: avg100 nie rośnie — rozważ ↑lr lub ↑hidden_layers")
    elif improve_first > 0.2 and improve_second < 0.05:
        observations.append("WCZESNE PLATEAU: avg100 rósł, ale stagnacja — rozważ ↑epsilon_decay, ↑num_episodes")
    elif improve_second > 0.1:
        observations.append("DOBRY TREND: avg100 rośnie stabilnie")

    # 2. Epsilon
    final_eps = df_train["epsilon"].iloc[-1]
    mid_eps = df_train["epsilon"].iloc[n // 2]
    if mid_eps < 0.1:
        observations.append(f"SZYBKI SPADEK EPSILON: epsilon={mid_eps:.3f} w połowie treningu — rozważ ↑epsilon_decay")

    # 3. TD error trend
    if "td_error_mean" in df_train.columns:
        td_early = df_train["td_error_mean"].iloc[n // 10 : n // 5].mean()
        td_late = df_train["td_error_mean"].iloc[-n // 10 :].mean()
        if td_early > 0 and td_late > td_early * 1.5:
            observations.append("ROSNĄCY TD ERROR: model traci stabilność — rozważ ↓lr, ↓tau")
        elif td_early > 0 and td_late < td_early * 0.5:
            observations.append("SPADAJĄCY TD ERROR: dobre predykcje wartości ✓")

    # 4. Eval vs training
    df_eval, meta_eval = load_latest(env_name, "eval")
    if df_eval is None:
        df_eval, meta_eval = load_latest(env_name, "standalone_eval")
    if df_eval is not None and not df_eval.empty:
        eval_mean = df_eval["mean_reward"].iloc[-1]
        train_avg = df_train["avg100"].iloc[-1]
        gap = train_avg - eval_mean
        if gap > 0 and gap / max(abs(train_avg), 1) > 0.3:
            observations.append(
                f"EVAL << TRAIN: eval={eval_mean:.1f} vs avg100={train_avg:.1f} "
                "— policy misalignment, rozważ ↓epsilon_min, ↓tau"
            )
        elif eval_mean - train_avg > 0 and (eval_mean - train_avg) / max(abs(train_avg), 1) > 0.1:
            observations.append(
                f"EVAL > TRAIN: eval={eval_mean:.1f} vs avg100={train_avg:.1f} "
                "— greedy policy lepsza niż epsilon-greedy ✓"
            )
        if "std_reward" in df_eval.columns:
            std = df_eval["std_reward"].iloc[-1]
            if abs(eval_mean) > 0 and std / abs(eval_mean) > 0.2:
                observations.append(
                    f"WYSOKI STD: eval std={std:.1f} ({std/abs(eval_mean)*100:.0f}% mean) "
                    "— niestabilna policy, rozważ ↑batch_size, ↓tau"
                )

    if not observations:
        observations.append("Brak wyraźnych problemów — sprawdź czy cel został osiągnięty.")

    return observations


def build_summary_report(env_name):
    """Buduje połączony raport train+eval dla danego środowiska."""
    summary_train = compare_runs(env_name, "train")
    summary_eval = compare_runs(env_name, "eval")

    if summary_train.empty and summary_eval.empty:
        return pd.DataFrame()

    summary = summary_train.copy()
    if summary.empty:
        summary = summary_eval.copy()
        return summary

    if not summary_eval.empty:
        eval_cols = [
            column
            for column in ["timestamp", "model", "final_mean_reward", "best_mean_reward", "final_std_reward"]
            if column in summary_eval.columns
        ]
        summary = summary.merge(summary_eval[eval_cols], on=["timestamp", "model"], how="left")

    return summary


def export_summary_report(env_name, output_path=None):
    """Eksportuje raport porównawczy do CSV i zwraca (DataFrame, Path)."""
    summary = build_summary_report(env_name)
    if summary.empty:
        return summary, None

    if output_path is None:
        output_path = METRICS_DIR / f"runs_summary_{env_name}.csv"
    else:
        output_path = Path(output_path)

    summary.to_csv(output_path, index=False)
    return summary, output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analiza metryk DQN z metrics/*.csv i generowanie raportu porównawczego."
    )
    parser.add_argument(
        "env_name",
        nargs="?",
        help="Nazwa środowiska, np. CartPole-v1, MountainCar-v0, Acrobot-v1",
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="Pokaż dostępne środowiska wykryte w katalogu metrics/",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Ogranicz porównanie do ostatnich N runów",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Zapisz raport porównawczy do metrics/runs_summary_<env>.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ścieżka wyjściowa dla eksportu CSV (opcjonalna)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runs = list_runs()

    if args.list_envs:
        envs = sorted(runs["env"].unique()) if not runs.empty else []
        if envs:
            print("Dostępne środowiska:")
            for env in envs:
                print(f"- {env}")
        else:
            print("Brak danych w metrics/.")
        return

    if not args.env_name:
        raise SystemExit("Podaj env_name albo użyj --list-envs.")

    print(f"=== {args.env_name} ===\n")

    train_cmp = compare_runs(args.env_name, "train", last_n=args.last_n)
    if not train_cmp.empty:
        print("TRAINING RUNS:")
        print(train_cmp.to_string(index=False))
        print()
    else:
        print("Brak runów treningowych.\n")

    eval_cmp = compare_runs(args.env_name, "eval", last_n=args.last_n)
    if not eval_cmp.empty:
        print("EVAL RUNS:")
        print(eval_cmp.to_string(index=False))
        print()
    else:
        print("Brak runów ewaluacyjnych.\n")

    print("DIAGNOZA:")
    for observation in diagnose(args.env_name):
        print(f"- {observation}")

    if args.export:
        summary, output_path = export_summary_report(args.env_name, args.output)
        if summary.empty:
            print("\nBrak danych do eksportu.")
        else:
            print(f"\nZapisano raport: {output_path}")


if __name__ == "__main__":
    main()
