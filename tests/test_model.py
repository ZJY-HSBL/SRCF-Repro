import pytest
from srcf_repro.model import Profile, Recommender, adjusted_cosine, demographic_similarity


def test_similarity_equal_opposite_and_undefined():
    assert adjusted_cosine({1: 1, 2: 3}, {1: 2, 2: 4}) == pytest.approx(1)
    assert adjusted_cosine({1: 1, 2: 3}, {1: 4, 2: 2}) == pytest.approx(-1)
    assert adjusted_cosine({1: 1}, {1: 1}) == 0
    assert adjusted_cosine({1: 3, 2: 3}, {1: 4, 2: 2}) == 0


def test_demographics_from_paper_age_bins():
    assert demographic_similarity(Profile(26, "M", "a"), Profile(26, "M", "a")) == pytest.approx(1)
    assert 0 < demographic_similarity(Profile(26), Profile(63)) < .01
    assert demographic_similarity(None, Profile(26)) == 0


def toy(prediction="normalized", mode="sr"):
    return Recommender({1: {1: 1, 2: 5}, 2: {1: 1, 2: 5, 3: 4},
                        3: {1: 5, 2: 1, 3: 1}},
                       {1: Profile(26, "M", "dev"), 2: Profile(27, "M", "dev"),
                        3: Profile(63, "F", "farmer")}, mode=mode,
                       weights=(.2, .3, .5), prediction=prediction)


def test_original_paper_formula_is_not_normalized():
    m = toy(prediction="paper", mode="ub")
    assert m.predict(1, 3, 2) == pytest.approx(3.0)
    assert m.predict(1, 3, 1) == pytest.approx(4.0)


def test_normalized_formula_is_bounded_and_excludes_seen():
    m = toy()
    assert 1 <= m.predict(1, 3, 2) <= 5
    assert [i for i, _ in m.recommend(1, k=2)] == [3]


def test_cold_user_works_with_demographics_only():
    m = toy()
    m.profiles[4] = Profile(26, "M", "dev")
    assert 1 <= m.predict(4, 3, k=2) <= 5


def test_weights_validate_and_deterministic_neighbors():
    with pytest.raises(ValueError):
        Recommender({1: {1: 3}}, weights=(.1, .2, .8))
    m = toy()
    assert m.neighbors(1, 2) == m.neighbors(1, 2)
    with pytest.raises(ValueError):
        m.neighbors(1, 0)
