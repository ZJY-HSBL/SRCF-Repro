import pytest
from srcf_repro.data import popularity_seeds, split_by_user, load_movielens100k


def test_split_disjoint_exhaustive_deterministic():
    original = {1: {i: float(i) for i in range(1, 6)}, 2: {1: 3.0}}
    a, b = split_by_user(original, seed=3)
    assert (a, b) == split_by_user(original, seed=3)
    for u in original:
        assert set(a[u]).isdisjoint(b[u])
        assert a[u] | b[u] == original[u]
        assert a[u]


def test_seeds_selected_using_only_training():
    train = {1: {2: 5, 3: 3}, 2: {2: 4, 1: 3}, 3: {2: 3, 1: 5}}
    assert popularity_seeds(train, 1) == frozenset({2})
    assert popularity_seeds(train, 0) == frozenset()


def test_missing_real_dataset_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_movielens100k(tmp_path)
