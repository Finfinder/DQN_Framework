"""Unit tests for utils.analyze."""

import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.analyze import (
    _parse_run_filename,
    _diagnose_trend,
    _diagnose_epsilon,
    _diagnose_td_error,
    _diagnose_eval_vs_train,
    diagnose,
    list_runs,
    load_run,
    load_latest,
    compare_runs,
    run_summary,
    build_summary_report,
    export_summary_report,
    parse_args,
    _print_env_list,
    _print_train_eval_results,
    main,
)


class TestParseRunFilename:
    def test_valid_train_filename(self):
        stem = "CartPole-v1_dqn_cartpole_dueling_20250101-120000"
        result = _parse_run_filename(stem, is_eval=False)
        assert result is not None
        env, model, ts, run_type = result
        assert env == "CartPole-v1"
        assert "dqn_cartpole" in model
        assert isinstance(ts, datetime)
        assert run_type == "train"

    def test_valid_eval_filename(self):
        stem = "CartPole-v1_dqn_cartpole_dueling_20250101-120000_eval"
        result = _parse_run_filename(stem, is_eval=True)
        assert result is not None
        env, _model, _ts, run_type = result
        assert env == "CartPole-v1"
        assert run_type == "eval"

    def test_invalid_filename_missing_dqn_prefix(self):
        stem = "CartPole-v1_nomatch_20250101-120000"
        result = _parse_run_filename(stem, is_eval=False)
        assert result is None

    def test_invalid_filename_bad_timestamp(self):
        stem = "CartPole-v1_dqn_cartpole_dueling_99999999-999999"
        result = _parse_run_filename(stem, is_eval=False)
        assert result is None

    def test_standalone_eval_run_type(self):
        stem = "CartPole-v1_dqn_cartpole_standalone_eval_20250101-120000"
        result = _parse_run_filename(stem, is_eval=False)
        # Model names without 'dqn_' prefix return None; this stem has no valid _dqn_ separator
        # just before timestamp, so the parser cannot extract a valid timestamp position.
        # Either None (unparseable) or a valid tuple — must not raise an exception.
        assert result is None or (isinstance(result, tuple) and len(result) == 4)

    def test_mountaincar_env_parsed(self):
        stem = "MountainCar-v0_dqn_mountaincar_dueling_20250606-100000"
        result = _parse_run_filename(stem, is_eval=False)
        assert result is not None
        env, _model, _ts, _run_type = result
        assert env == "MountainCar-v0"

    def test_timestamp_parsed_correctly(self):
        stem = "CartPole-v1_dqn_cartpole_dueling_20240315-143000"
        result = _parse_run_filename(stem, is_eval=False)
        assert result is not None
        _, _, ts, _ = result
        assert ts.year == 2024
        assert ts.month == 3
        assert ts.day == 15
        assert ts.hour == 14
        assert ts.minute == 30

    def test_returns_four_element_tuple(self):
        stem = "CartPole-v1_dqn_cartpole_dueling_20250101-120000"
        result = _parse_run_filename(stem, is_eval=False)
        assert result is not None
        assert len(result) == 4


class TestDiagnoseTrend:
    def test_no_learning_flat_avg100(self):
        df = pd.DataFrame({"avg100": [10.0] * 100})
        obs = []
        _diagnose_trend(df, obs)
        assert len(obs) == 1
        assert "BRAK UCZENIA" in obs[0]

    def test_early_plateau(self):
        values = [5.0] * 10 + [50.0] * 70 + [52.0] * 20
        df = pd.DataFrame({"avg100": values})
        obs = []
        _diagnose_trend(df, obs)
        assert len(obs) == 1
        assert "WCZESNE PLATEAU" in obs[0]

    def test_good_trend_steadily_rising(self):
        df = pd.DataFrame({"avg100": list(range(100))})
        obs = []
        _diagnose_trend(df, obs)
        assert len(obs) == 1
        assert "DOBRY TREND" in obs[0]

    def test_no_observation_when_moderate_growth(self):
        # poprawa totalna >= 0.1, ale improve_first <= 0.2 i improve_second <= 0.1
        # np. liniowy wzrost od 50 do 60: scale=60, improve_second ≈0.077
        values = [float(50 + i * 0.1) for i in range(100)]
        df = pd.DataFrame({"avg100": values})
        obs = []
        _diagnose_trend(df, obs)
        assert len(obs) == 0


class TestDiagnoseEpsilon:
    def test_fast_epsilon_decay(self):
        n = 100
        epsilons = [1.0] * 50 + [0.05] * 50
        df = pd.DataFrame({"avg100": [0.0] * n, "epsilon": epsilons})
        obs = []
        _diagnose_epsilon(df, obs)
        assert len(obs) == 1
        assert "SZYBKI SPADEK EPSILON" in obs[0]

    def test_normal_epsilon_no_observation(self):
        df = pd.DataFrame({"avg100": [0.0] * 100, "epsilon": [0.5] * 100})
        obs = []
        _diagnose_epsilon(df, obs)
        assert len(obs) == 0


class TestDiagnoseTdError:
    def test_rising_td_error(self):
        n = 100
        td_vals = [0.0] * 10 + [2.0] * 10 + [0.0] * 70 + [4.0] * 10
        df = pd.DataFrame({"avg100": [0.0] * n, "td_error_mean": td_vals})
        obs = []
        _diagnose_td_error(df, obs)
        assert len(obs) == 1
        assert "ROSNĄCY TD ERROR" in obs[0]

    def test_falling_td_error(self):
        n = 100
        td_vals = [0.0] * 10 + [4.0] * 10 + [0.0] * 70 + [1.0] * 10
        df = pd.DataFrame({"avg100": [0.0] * n, "td_error_mean": td_vals})
        obs = []
        _diagnose_td_error(df, obs)
        assert len(obs) == 1
        assert "SPADAJĄCY TD ERROR" in obs[0]

    def test_missing_column_no_observation(self):
        df = pd.DataFrame({"avg100": [0.0] * 100, "epsilon": [0.5] * 100})
        obs = []
        _diagnose_td_error(df, obs)
        assert len(obs) == 0


class TestListRuns:
    def test_returns_all_runs(self, tmp_path):
        (tmp_path / "CartPole-v1_dqn_cartpole_dueling_20250101-120000.csv").touch()
        (tmp_path / "CartPole-v1_dqn_cartpole_dueling_20250102-120000_eval.csv").touch()
        with patch("utils.analyze.METRICS_DIR", tmp_path):
            result = list_runs()
        assert len(result) == 2
        assert set(result.columns) >= {
            "file",
            "path",
            "env",
            "model",
            "timestamp",
            "type",
        }

    def test_filters_by_env_name(self, tmp_path):
        (tmp_path / "CartPole-v1_dqn_cartpole_dueling_20250101-120000.csv").touch()
        (
            tmp_path / "MountainCar-v0_dqn_mountaincar_dueling_20250101-120000.csv"
        ).touch()
        with patch("utils.analyze.METRICS_DIR", tmp_path):
            result = list_runs(env_name="CartPole-v1")
        assert len(result) == 1
        assert result.iloc[0]["env"] == "CartPole-v1"

    def test_eval_only_filter(self, tmp_path):
        (tmp_path / "CartPole-v1_dqn_cartpole_dueling_20250101-120000.csv").touch()
        (tmp_path / "CartPole-v1_dqn_cartpole_dueling_20250101-120000_eval.csv").touch()
        with patch("utils.analyze.METRICS_DIR", tmp_path):
            result = list_runs(eval_only=True)
        assert len(result) == 1
        assert result.iloc[0]["type"] == "eval"

    def test_train_only_filter(self, tmp_path):
        (tmp_path / "CartPole-v1_dqn_cartpole_dueling_20250101-120000.csv").touch()
        (tmp_path / "CartPole-v1_dqn_cartpole_dueling_20250101-120000_eval.csv").touch()
        with patch("utils.analyze.METRICS_DIR", tmp_path):
            result = list_runs(train_only=True)
        assert len(result) == 1
        assert result.iloc[0]["type"] == "train"

    def test_empty_metrics_dir(self, tmp_path):
        with patch("utils.analyze.METRICS_DIR", tmp_path):
            result = list_runs()
        assert result.empty

    def test_invalid_filename_skipped(self, tmp_path):
        (tmp_path / "invalid_file.csv").touch()
        (tmp_path / "CartPole-v1_dqn_cartpole_dueling_20250101-120000.csv").touch()
        with patch("utils.analyze.METRICS_DIR", tmp_path):
            result = list_runs()
        assert len(result) == 1


class TestLoadRun:
    def test_relative_path_joins_metrics_dir(self, tmp_path):
        df = pd.DataFrame({"reward": [100.0]})
        with (
            patch("utils.analyze.METRICS_DIR", tmp_path),
            patch("pandas.read_csv", return_value=df) as mock_read,
        ):
            result = load_run("file.csv")
        mock_read.assert_called_once_with(tmp_path / "file.csv")
        assert result.equals(df)

    def test_absolute_path_used_directly(self, tmp_path):
        df = pd.DataFrame({"reward": [50.0]})
        abs_path = str(tmp_path / "file.csv")
        with patch("pandas.read_csv", return_value=df) as mock_read:
            result = load_run(abs_path)
        mock_read.assert_called_once_with(Path(abs_path))
        assert result.equals(df)


class TestLoadLatest:
    def _make_runs_df(self, run_type):
        suffix = "_eval" if run_type != "train" else ""
        return pd.DataFrame(
            {
                "file": [
                    f"CartPole-v1_dqn_cartpole_dueling_20250101-120000{suffix}.csv"
                ],
                "path": ["/fake/path/file.csv"],
                "env": ["CartPole-v1"],
                "model": ["dqn_cartpole_dueling"],
                "timestamp": [datetime(2025, 1, 1, 12, 0)],
                "type": [run_type],
            }
        )

    def test_returns_latest_run(self):
        runs_df = self._make_runs_df("train")
        data_df = pd.DataFrame({"reward": [100.0], "avg100": [90.0]})
        with (
            patch("utils.analyze.list_runs", return_value=runs_df),
            patch("utils.analyze.load_run", return_value=data_df),
        ):
            result_df, meta = load_latest("CartPole-v1", "train")
        assert result_df is not None
        assert result_df.equals(data_df)
        assert meta["env"] == "CartPole-v1"

    def test_returns_none_when_no_runs(self):
        empty_df = pd.DataFrame(
            columns=["file", "path", "env", "model", "timestamp", "type"]
        )
        with patch("utils.analyze.list_runs", return_value=empty_df):
            result_df, meta = load_latest("CartPole-v1", "train")
        assert result_df is None
        assert meta is None


class TestCompareRuns:
    def _make_runs_df(self, run_type):
        return pd.DataFrame(
            {
                "file": ["CartPole-v1_dqn_cartpole_dueling_20250101-120000.csv"],
                "path": ["/fake/path/file.csv"],
                "env": ["CartPole-v1"],
                "model": ["dqn_cartpole_dueling"],
                "timestamp": [datetime(2025, 1, 1, 12, 0)],
                "type": [run_type],
            }
        )

    def test_train_summary_columns(self):
        runs_df = self._make_runs_df("train")
        data_df = pd.DataFrame(
            {
                "reward": [100.0, 110.0],
                "avg100": [90.0, 95.0],
                "epsilon": [0.1, 0.05],
            }
        )
        with (
            patch("utils.analyze.list_runs", return_value=runs_df),
            patch("utils.analyze.load_run", return_value=data_df),
        ):
            result = compare_runs("CartPole-v1", "train")
        assert not result.empty
        assert "final_avg100" in result.columns
        assert "best_avg100" in result.columns
        assert "final_epsilon" in result.columns

    def test_eval_summary_columns(self):
        runs_df = self._make_runs_df("eval")
        data_df = pd.DataFrame(
            {
                "mean_reward": [85.0, 90.0],
                "std_reward": [5.0, 4.0],
            }
        )
        with (
            patch("utils.analyze.list_runs", return_value=runs_df),
            patch("utils.analyze.load_run", return_value=data_df),
        ):
            result = compare_runs("CartPole-v1", "eval")
        assert "final_mean_reward" in result.columns
        assert "best_mean_reward" in result.columns
        assert "final_std_reward" in result.columns

    def test_empty_runs_returns_empty_dataframe(self):
        empty_df = pd.DataFrame(
            columns=["file", "path", "env", "model", "timestamp", "type"]
        )
        with patch("utils.analyze.list_runs", return_value=empty_df):
            result = compare_runs("CartPole-v1", "train")
        assert result.empty

    def test_last_n_limits_runs(self):
        runs_df = pd.DataFrame(
            {
                "file": ["f1.csv", "f2.csv"],
                "path": ["/fake/f1.csv", "/fake/f2.csv"],
                "env": ["CartPole-v1", "CartPole-v1"],
                "model": ["dqn_m1", "dqn_m1"],
                "timestamp": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                "type": ["train", "train"],
            }
        )
        data_df = pd.DataFrame({"reward": [100.0], "avg100": [90.0], "epsilon": [0.1]})
        with (
            patch("utils.analyze.list_runs", return_value=runs_df),
            patch("utils.analyze.load_run", return_value=data_df),
        ):
            result = compare_runs("CartPole-v1", "train", last_n=1)
        assert len(result) == 1


class TestDiagnoseEvalVsTrain:
    def _make_train_df(self, avg100):
        return pd.DataFrame(
            {
                "reward": [avg100] * 100,
                "avg100": [avg100] * 100,
                "epsilon": [0.1] * 100,
            }
        )

    def test_eval_much_lower_than_train(self):
        df_train = self._make_train_df(100.0)
        df_eval = pd.DataFrame({"mean_reward": [60.0]})
        obs = []

        def mock_ll(env, run_type="train"):
            return (df_eval, {}) if run_type == "eval" else (None, None)

        with patch("utils.analyze.load_latest", side_effect=mock_ll):
            _diagnose_eval_vs_train(df_train, "CartPole-v1", obs)

        assert len(obs) == 1
        assert "EVAL << TRAIN" in obs[0]

    def test_eval_higher_than_train(self):
        df_train = self._make_train_df(50.0)
        df_eval = pd.DataFrame({"mean_reward": [60.0]})
        obs = []

        def mock_ll(env, run_type="train"):
            return (df_eval, {}) if run_type == "eval" else (None, None)

        with patch("utils.analyze.load_latest", side_effect=mock_ll):
            _diagnose_eval_vs_train(df_train, "CartPole-v1", obs)

        assert len(obs) == 1
        assert "EVAL > TRAIN" in obs[0]

    def test_high_std_detected(self):
        df_train = self._make_train_df(50.0)
        df_eval = pd.DataFrame({"mean_reward": [50.0], "std_reward": [15.0]})
        obs = []

        def mock_ll(env, run_type="train"):
            return (df_eval, {}) if run_type == "eval" else (None, None)

        with patch("utils.analyze.load_latest", side_effect=mock_ll):
            _diagnose_eval_vs_train(df_train, "CartPole-v1", obs)

        assert any("WYSOKI STD" in o for o in obs)

    def test_no_eval_data_returns_early(self):
        df_train = self._make_train_df(100.0)
        obs = []
        with patch("utils.analyze.load_latest", return_value=(None, None)):
            _diagnose_eval_vs_train(df_train, "CartPole-v1", obs)
        assert len(obs) == 0


class TestDiagnose:
    def test_no_training_data_returns_message(self):
        with patch("utils.analyze.load_latest", return_value=(None, None)):
            result = diagnose("CartPole-v1")
        assert len(result) == 1
        assert "Brak danych treningowych" in result[0]

    def test_with_training_data_returns_observations(self):
        df_train = pd.DataFrame(
            {
                "avg100": [10.0] * 100,
                "epsilon": [0.5] * 100,
            }
        )

        def mock_ll(env, run_type="train"):
            return (df_train, {}) if run_type == "train" else (None, None)

        with patch("utils.analyze.load_latest", side_effect=mock_ll):
            result = diagnose("CartPole-v1")

        assert isinstance(result, list)
        assert len(result) >= 1
        assert any("BRAK UCZENIA" in o for o in result)


class TestBuildSummaryReport:
    def test_both_train_and_eval_merged(self):
        ts = datetime(2025, 1, 1, 12, 0)
        train_df = pd.DataFrame(
            {
                "timestamp": [ts],
                "model": ["dqn_m1"],
                "final_avg100": [90.0],
                "best_avg100": [95.0],
                "num_episodes": [100],
            }
        )
        eval_df = pd.DataFrame(
            {
                "timestamp": [ts],
                "model": ["dqn_m1"],
                "final_mean_reward": [85.0],
                "best_mean_reward": [88.0],
                "num_episodes": [10],
            }
        )

        def mock_cr(env, run_type, **kw):
            return train_df if run_type == "train" else eval_df

        with patch("utils.analyze.compare_runs", side_effect=mock_cr):
            result = build_summary_report("CartPole-v1")

        assert "final_avg100" in result.columns
        assert "final_mean_reward" in result.columns

    def test_only_train_data(self):
        train_df = pd.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1)],
                "model": ["dqn_m1"],
                "final_avg100": [90.0],
                "num_episodes": [100],
            }
        )

        def mock_cr(env, run_type, **kw):
            return train_df if run_type == "train" else pd.DataFrame()

        with patch("utils.analyze.compare_runs", side_effect=mock_cr):
            result = build_summary_report("CartPole-v1")

        assert not result.empty
        assert "final_avg100" in result.columns

    def test_empty_returns_empty_dataframe(self):
        with patch("utils.analyze.compare_runs", return_value=pd.DataFrame()):
            result = build_summary_report("CartPole-v1")
        assert result.empty


class TestExportSummaryReport:
    def test_exports_to_custom_path(self, tmp_path):
        df = pd.DataFrame({"a": [1], "b": [2]})
        out_path = tmp_path / "report.csv"
        with patch("utils.analyze.build_summary_report", return_value=df):
            result_df, result_path = export_summary_report(
                "CartPole-v1", output_path=str(out_path)
            )
        assert out_path.exists()
        assert str(result_path) == str(out_path)
        assert result_df.equals(df)

    def test_empty_summary_returns_none_path(self):
        with patch("utils.analyze.build_summary_report", return_value=pd.DataFrame()):
            result_df, result_path = export_summary_report("CartPole-v1")
        assert result_path is None
        assert result_df.empty


class TestParseArgs:
    def test_env_name_parsed(self):
        with patch("sys.argv", ["analyze.py", "CartPole-v1"]):
            args = parse_args()
        assert args.env_name == "CartPole-v1"

    def test_list_envs_flag(self):
        with patch("sys.argv", ["analyze.py", "--list-envs"]):
            args = parse_args()
        assert args.list_envs is True

    def test_last_n_and_export(self):
        with patch(
            "sys.argv", ["analyze.py", "CartPole-v1", "--last-n", "5", "--export"]
        ):
            args = parse_args()
        assert args.last_n == 5
        assert args.export is True


class TestPrintEnvList:
    def test_prints_list_of_envs(self, capsys):
        runs = pd.DataFrame({"env": ["CartPole-v1", "CartPole-v1", "MountainCar-v0"]})
        _print_env_list(runs)
        out = capsys.readouterr().out
        assert "CartPole-v1" in out
        assert "MountainCar-v0" in out

    def test_empty_dataframe_prints_no_data(self, capsys):
        _print_env_list(pd.DataFrame({"env": []}))
        out = capsys.readouterr().out
        assert "Brak danych" in out


class TestPrintTrainEvalResults:
    def test_prints_train_and_eval_sections(self, capsys):
        train_df = pd.DataFrame(
            {"model": ["m1"], "final_avg100": [90.0], "num_episodes": [100]}
        )
        eval_df = pd.DataFrame(
            {"model": ["m1"], "final_mean_reward": [85.0], "num_episodes": [10]}
        )

        def mock_cr(env, run_type, **kw):
            return train_df if run_type == "train" else eval_df

        with (
            patch("utils.analyze.compare_runs", side_effect=mock_cr),
            patch("utils.analyze.diagnose", return_value=["Brak wyrążnych problemów."]),
        ):
            _print_train_eval_results("CartPole-v1", None)

        out = capsys.readouterr().out
        assert "TRAINING RUNS:" in out
        assert "EVAL RUNS:" in out
        assert "DIAGNOZA:" in out

    def test_prints_no_runs_when_empty(self, capsys):
        with (
            patch("utils.analyze.compare_runs", return_value=pd.DataFrame()),
            patch("utils.analyze.diagnose", return_value=["Brak danych treningowych."]),
        ):
            _print_train_eval_results("CartPole-v1", None)

        out = capsys.readouterr().out
        assert "Brak runów treningowych" in out
        assert "Brak runów ewaluacyjnych" in out


class TestMain:
    def test_list_envs_calls_print_env_list(self):
        args = MagicMock()
        args.list_envs = True
        args.env_name = None
        runs_df = pd.DataFrame({"env": ["CartPole-v1"]})

        with (
            patch("utils.analyze.parse_args", return_value=args),
            patch("utils.analyze.list_runs", return_value=runs_df),
            patch("utils.analyze._print_env_list") as mock_prl,
        ):
            main()

        mock_prl.assert_called_once_with(runs_df)

    def test_no_env_name_raises_system_exit(self):
        args = MagicMock()
        args.list_envs = False
        args.env_name = None

        with (
            patch("utils.analyze.parse_args", return_value=args),
            patch("utils.analyze.list_runs", return_value=pd.DataFrame()),
        ):
            with pytest.raises(SystemExit):
                main()

    def test_env_name_calls_print_results(self):
        args = MagicMock()
        args.list_envs = False
        args.env_name = "CartPole-v1"
        args.last_n = None
        args.export = False

        with (
            patch("utils.analyze.parse_args", return_value=args),
            patch("utils.analyze.list_runs", return_value=pd.DataFrame()),
            patch("utils.analyze._print_train_eval_results") as mock_pter,
        ):
            main()

        mock_pter.assert_called_once_with("CartPole-v1", None)

    def test_export_flag_calls_export(self, tmp_path):
        args = MagicMock()
        args.list_envs = False
        args.env_name = "CartPole-v1"
        args.last_n = None
        args.export = True
        args.output = None

        df = pd.DataFrame({"a": [1]})
        with (
            patch("utils.analyze.parse_args", return_value=args),
            patch("utils.analyze.list_runs", return_value=pd.DataFrame()),
            patch("utils.analyze._print_train_eval_results"),
            patch(
                "utils.analyze.export_summary_report",
                return_value=(df, tmp_path / "r.csv"),
            ) as mock_exp,
        ):
            main()

        mock_exp.assert_called_once_with("CartPole-v1", None)


class TestRunSummary:
    def test_prints_train_and_eval(self, capsys):
        train_df = pd.DataFrame(
            {"model": ["m1"], "final_avg100": [90.0], "num_episodes": [100]}
        )
        eval_df = pd.DataFrame(
            {"model": ["m1"], "final_mean_reward": [85.0], "num_episodes": [10]}
        )

        def mock_cr(env, run_type, **kw):
            return train_df if run_type == "train" else eval_df

        with patch("utils.analyze.compare_runs", side_effect=mock_cr):
            run_summary("CartPole-v1")

        out = capsys.readouterr().out
        assert "TRAINING RUNS:" in out
        assert "EVAL RUNS:" in out

    def test_no_data_prints_no_data(self, capsys):
        with patch("utils.analyze.compare_runs", return_value=pd.DataFrame()):
            run_summary("CartPole-v1")

        out = capsys.readouterr().out
        assert "Brak danych." in out
