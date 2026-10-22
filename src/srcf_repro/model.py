"""User-based, demographic, and mixed similarity recommenders.

Source: Qiao Yu (2018), chapter 4, equations (4.3)-(4.9).
The thesis leaves several implementation decisions unspecified; these are
explicitly documented in docs/REPLICATION.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp, isfinite
from typing import Mapping


@dataclass(frozen=True)
class Profile:
    age: int | None = None
    gender: str | None = None
    occupation: str | None = None


def adjusted_cosine(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    """Equation (4.3): mean-centred similarity on co-rated items.

    Return zero if fewer than two co-ratings or zero variance; no imputation.
    """
    common = left.keys() & right.keys()
    if len(common) < 2:
        return 0.0
    a = [left[i] for i in common]
    b = [right[i] for i in common]
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    denom = (sum(x*x for x in da)*sum(x*x for x in db))**0.5
    if denom == 0:
        return 0.0
    return max(-1.0, min(1.0, sum(x*y for x, y in zip(da, db))/denom))


def demographic_similarity(left: Profile | None, right: Profile | None) -> float:
    """Equation (4.5)/(4.7) with uniform weights on *available* attributes.

    Occupation/gender are categorical equality. The paper's hobby-tag
    denominator is ambiguous; MovieLens 100K has no hobby tags.
    """
    if left is None or right is None:
        return 0.0
    parts = []
    if left.age is not None and right.age is not None:
        # 26 -> segment 6; 63 -> segment 13, as in the thesis.
        parts.append(exp(-abs((left.age // 5 + 1) - (right.age // 5 + 1))))
    if left.gender is not None and right.gender is not None:
        parts.append(float(left.gender == right.gender))
    if left.occupation is not None and right.occupation is not None:
        parts.append(float(left.occupation == right.occupation))
    return sum(parts)/len(parts) if parts else 0.0


class Recommender:
    """Deterministic, reference-scale neighborhood model.

    mode: 'ub' historical ratings, 'dr' demographics, 'sr' blended SR-CF.
    prediction: 'paper' literal Eq (4.4), or 'normalized' bounded weighted
    mean over positive-similarity neighbors (a clearly labeled correction).
    """

    def __init__(self, ratings: Mapping[int, Mapping[int, float]],
                 profiles: Mapping[int, Profile] | None = None,
                 seeds: frozenset[int] | set[int] | None = None,
                 mode: str = "sr", weights: tuple[float, float, float] = (.2, .3, .5),
                 prediction: str = "normalized") -> None:
        if mode not in {"ub", "dr", "sr"}:
            raise ValueError("mode must be ub, dr or sr")
        if prediction not in {"paper", "normalized"}:
            raise ValueError("prediction must be paper or normalized")
        if len(weights) != 3 or any(not isfinite(w) or w < 0 for w in weights) or abs(sum(weights)-1) > 1e-9:
            raise ValueError("weights must be nonnegative and sum to 1")
        self.ratings = {int(u): dict(items) for u, items in ratings.items()}
        if any(not 1 <= r <= 5 for items in self.ratings.values() for r in items.values()):
            raise ValueError("expected MovieLens ratings between 1 and 5")
        self.profiles = dict(profiles or {})
        self.seeds = frozenset(seeds or ())
        self.mode, self.weights, self.prediction = mode, weights, prediction
        self.users = tuple(sorted(u for u, rated in self.ratings.items() if rated))
        items = {i for rated in self.ratings.values() for i in rated}
        if not items:
            raise ValueError("training ratings must contain at least one item")
        self.items = tuple(sorted(items))
        by_item: dict[int, list[float]] = {}
        for rated in self.ratings.values():
            for i, rating in rated.items():
                by_item.setdefault(i, []).append(rating)
        self.item_mean = {i: sum(v)/len(v) for i, v in by_item.items()}
        all_ratings = [v for values in by_item.values() for v in values]
        self.global_mean = sum(all_ratings)/len(all_ratings)

    @lru_cache(maxsize=None)
    def similarity(self, user: int, other: int) -> float:
        if user == other:
            return 1.0
        left = self.ratings.get(user, {})
        right = self.ratings.get(other, {})
        if self.mode == "dr":
            return demographic_similarity(self.profiles.get(user), self.profiles.get(other))
        historical = adjusted_cosine(left, right)
        if self.mode == "ub":
            return historical
        seed_left = {i: left[i] for i in self.seeds if i in left}
        seed_right = {i: right[i] for i in self.seeds if i in right}
        seed_sim = adjusted_cosine(seed_left, seed_right)
        a, b, c = self.weights
        return (a * demographic_similarity(self.profiles.get(user), self.profiles.get(other))
                + b * seed_sim + c * historical)

    @lru_cache(maxsize=None)
    def neighbors(self, user: int, k: int) -> tuple[tuple[int, float], ...]:
        if k <= 0:
            raise ValueError("k must be positive")
        scored = [(v, self.similarity(user, v)) for v in self.users if v != user]
        scored.sort(key=lambda t: (-t[1], t[0]))
        return tuple(scored[:k])

    def predict(self, user: int, item: int, k: int = 30) -> float:
        relevant = [(s, self.ratings.get(v, {})[item]) for v, s in self.neighbors(user, k)
                    if item in self.ratings.get(v, {})]
        if self.prediction == "paper":
            # Literal Eq. (4.4), without any normalization or clipping.
            return sum(s*r for s, r in relevant)
        positive = [(s, r) for s, r in relevant if s > 0]
        if positive:
            prediction = sum(s*r for s, r in positive)/sum(s for s, _ in positive)
            return max(1.0, min(5.0, prediction))
        return self.item_mean.get(item, self.global_mean)

    def recommend(self, user: int, k: int = 30, top_n: int = 10) -> list[tuple[int, float]]:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        seen = self.ratings.get(user, {})
        candidates = ((i, self.predict(user, i, k)) for i in self.items if i not in seen)
        return sorted(candidates, key=lambda item: (-item[1], item[0]))[:top_n]
