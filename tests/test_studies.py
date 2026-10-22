import pytest

from srcf_repro.experiment import synthetic
from srcf_repro.studies import (METHODS, build_cases, method_args,
                               paired_ci, run_study, scheduled_weights, summarize)


def test_predeclared_weights_are_valid_and_change_with_budget():
    assert scheduled_weights(0) == (1, 0, 0)
    assert scheduled_weights(10) == (.2, .3, .5)
    for method in METHODS:
        for observed in (0, 1, 3, 5, 10):
            _, w = method_args(method, observed)
            assert sum(w) == pytest.approx(1)
    with pytest.raises(ValueError):
        method_args("unknown", 3)


def test_all_target_users_excluded_and_fixed_holdout():
    ratings, _ = synthetic(users=23, items=45)
    reference, batches = build_cases(ratings, study_seed=20, batches=2, users_per_batch=5)
    all_ids = [u for group in batches for _, u, _, _ in group]
    assert len(set(all_ids)) == 10
    assert not (set(all_ids) & set(reference))
    for group in batches:
        for cid, user, test, train_ids in group:
            assert len(test) == 5
            assert set(test).isdisjoint(train_ids)
            assert len(train_ids) >= 10
            assert cid.startswith("case_")
            assert user not in reference


def test_small_study_reproducible_and_paired_bootstrap():
    ratings, profiles = synthetic(users=22, items=45)
    kw = dict(study_seed=11, batches=2, users_per_batch=3, observations=(0, 3, 10))
    a = run_study(ratings, profiles, **kw)
    b = run_study(ratings, profiles, **kw)
    assert a == b
    assert len(a) == 2 * 3 * 3 * len(METHODS)
    assert len(summarize(a)) == 3 * len(METHODS)
    c = paired_ci(a, "sr", "dr", 3, draws=300)
    assert c["n_users"] == 6
    assert c["ci_low"] <= c["mean_diff"] <= c["ci_high"]
    assert 0 <= c["wins_a"] <= c["n_users"]
    with pytest.raises(ValueError):
        paired_ci(a[:-1], "sr_schedule", "sr", 10)
