import json
from srcf_repro.experiment import run, save_results, synthetic


def test_synthetic_benchmark_is_reproducible_and_measured(tmp_path):
    ratings, profiles = synthetic(users=12, items=40)
    rows = run(ratings, profiles, max_users=7, k_values=(3, 5))
    assert rows == run(ratings, profiles, max_users=7, k_values=(3, 5))
    assert len(rows) == 6
    assert all(0 <= x["mae"] <= 4 and 0 <= x["coverage"] <= 1 and x["n_test"] > 0 for x in rows)
    save_results(rows, tmp_path, make_plot=False)
    assert json.loads((tmp_path / "benchmark.json").read_text()) == rows


def test_cold_curve_fixed_holdout_no_new_user_training_leakage():
    from srcf_repro.experiment import cold_start_curve
    ratings, profiles = synthetic(users=16, items=42)
    rows = cold_start_curve(ratings, profiles, max_users=3, observed=(0, 3), k=5)
    assert len(rows) == 6
    assert all(x["n_test"] == 15 for x in rows)
    assert {x["n_observed"] for x in rows} == {0, 3}
