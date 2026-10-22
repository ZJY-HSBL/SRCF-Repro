from srcf_repro.experiment import synthetic
from srcf_repro.seed_diagnostics import (METHODS, SEED_COUNTS, aggregate,
                                         prepare_cases, run, seed_support)


def test_seed_support_requires_two_shared_nonconstant_ratings():
    reference = {2: {1: 1, 2: 5, 3: 4}, 3: {1: 5, 2: 1}}
    assert seed_support({1: 5}, reference, frozenset({1, 2}))['candidate_pairs_with_nonzero_seed_similarity'] == 0
    a = seed_support({1: 1, 2: 5}, reference, frozenset({1, 2}))
    assert a['observed_seed_items'] == 2
    assert a['candidate_pairs_with_2_shared_seed_ratings'] == 2
    assert a['candidate_pairs_with_nonzero_seed_similarity'] == 2
    assert seed_support({1: 3, 2: 3}, reference, frozenset({1, 2}))['candidate_pairs_with_nonzero_seed_similarity'] == 0


def test_fresh_cohort_is_disjoint_from_prior_cohort_and_reference():
    ratings, _ = synthetic(users=50, items=42)
    ref, groups, previous = prepare_cases(ratings, old_b=2, old_n=4,
                                           batches=2, users_per_batch=5)
    ids = [u for batch in groups for _, u, _, _ in batch]
    assert previous == 8
    assert len(set(ids)) == 10
    assert not (set(ids) & set(ref))
    for batch in groups:
        for _, _, heldout, visible in batch:
            assert set(heldout).isdisjoint(visible)


def test_seed_sensitivity_is_deterministic_and_complete():
    ratings, profiles = synthetic(users=50, items=42)
    kwargs = dict(batches=2, users_per_batch=3, old_b=2, old_n=4)
    a, diag, old = run(ratings, profiles, **kwargs)
    b, diag2, old2 = run(ratings, profiles, **kwargs)
    assert (a, diag, old) == (b, diag2, old2)
    assert len(a) == 2*3*5*len(METHODS)
    assert len(diag) == 2*3*5
    for record in diag:
        assert record['seed12_seen'] <= record['seed40_seen'] <= record['seed100_seen']
    summary, support, ci = aggregate(a, diag)
    assert len(summary) == 5*len(METHODS)
    assert len(support) == 5*len(SEED_COUNTS)
    assert len(ci) == 5*3
