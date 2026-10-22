"""Confidence-aware SR-CF; research hypothesis, NOT part of the 2018 thesis.

A per-pair support-dependent similarity fusion shrinks unreliable co-rating
Pearson components and renormalizes the sum of active component weights.
"""

from __future__ import annotations

from functools import lru_cache

from .model import Recommender, adjusted_cosine, demographic_similarity


class ConfidenceRecommender(Recommender):
    """Shrunk similarity with pair-wise reliability; prediction remains normalized.

    lambda_support is set using *validation users*, never the test users.
    Missing or constant Pearson components receive zero reliability.
    A no-seed ablation is available to test whether seed information adds value.
    """

    def __init__(self, *args, lambda_support: float = 10.0,
                 use_seed: bool = True, **kwargs):
        if not 0 <= lambda_support < float('inf'):
            raise ValueError('lambda_support must be finite and nonnegative')
        if kwargs.get('mode', 'sr') != 'sr':
            raise ValueError('confidence variant only supports SR mode')
        super().__init__(*args, **kwargs)
        if self.prediction != 'normalized':
            raise ValueError('confidence variant requires the normalized predictor')
        self.lambda_support = float(lambda_support)
        self.use_seed = bool(use_seed)

    @lru_cache(maxsize=None)
    def similarity(self, user: int, other: int) -> float:
        if user == other:
            return 1.0
        left = self.ratings.get(user, {})
        right = self.ratings.get(other, {})
        demographic = demographic_similarity(self.profiles.get(user), self.profiles.get(other))
        common = left.keys() & right.keys()
        historical = adjusted_cosine(left, right)
        w_d, w_s, w_h = self.weights
        h_rel = len(common) / (len(common) + self.lambda_support) if historical else 0.0
        se = left.keys() & right.keys() & self.seeds if self.use_seed else set()
        if len(se) >= 2:
            seed_sim = adjusted_cosine({i:left[i] for i in se}, {i:right[i] for i in se})
        else:
            seed_sim = 0.0
        s_rel = len(se) / (len(se) + self.lambda_support) if seed_sim else 0.0
        numer = w_d * demographic + w_s * s_rel * seed_sim + w_h * h_rel * historical
        active = w_d + w_s * s_rel + w_h * h_rel
        return numer / active if active else 0.0
