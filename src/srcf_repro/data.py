"""Strictly local dataset loading and deterministic per-user holdout."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random

from .model import Profile


def load_movielens100k(folder: str | Path) -> tuple[dict[int, dict[int, float]], dict[int, Profile]]:
    folder = Path(folder)
    scores_path, users_path = folder / "u.data", folder / "u.user"
    if not scores_path.is_file() or not users_path.is_file():
        raise FileNotFoundError("MovieLens-100K requires u.data and u.user in --data-dir")
    ratings: dict[int, dict[int, float]] = defaultdict(dict)
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        user, item, score, _ = line.split("\t")
        ratings[int(user)][int(item)] = float(score)
    profiles: dict[int, Profile] = {}
    for line in users_path.read_text(encoding="utf-8").splitlines():
        user, age, gender, occupation, _ = line.split("|")
        profiles[int(user)] = Profile(int(age), gender, occupation)
    return dict(ratings), profiles


def split_by_user(ratings: dict[int, dict[int, float]], test_fraction: float = .2,
                  seed: int = 2026) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    rng = random.Random(seed)
    train, test = {}, {}
    for user, items in sorted(ratings.items()):
        ids = sorted(items)
        rng.shuffle(ids)
        n = min(len(ids) - 1, max(1, round(len(ids)*test_fraction))) if len(ids) > 1 else 0
        test_ids = set(ids[:n])
        train[user] = {i: items[i] for i in sorted(items) if i not in test_ids}
        test[user] = {i: items[i] for i in sorted(items) if i in test_ids}
    return train, test


def popularity_seeds(train: dict[int, dict[int, float]], count: int = 12) -> frozenset[int]:
    if count < 0:
        raise ValueError("count cannot be negative")
    counts = Counter(item for user in train.values() for item in user)
    return frozenset(i for i, _ in sorted(counts.items(), key=lambda t: (-t[1], t[0]))[:count])
