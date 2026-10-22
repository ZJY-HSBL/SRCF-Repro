# MovieLens-100K measured experiments / 真实数据实验

**Status / 状态: measured using the official MovieLens-100K dataset (not synthetic).**

This report is generated from `artifacts/real/ml100k/benchmark.csv` and `artifacts/real/cold/cold_start.csv` after executing the versioned implementation on downloaded **official MovieLens-100K** data. The raw dataset is not redistributed.

**Important scientific limitation / 科研限制:** These are independent, protocol-defined real-data experiments, **not exact reproduction of the thesis's numeric plots**. The thesis does not specify its seed-set construction, train/test partition, user sample or all demographic feature weights; the predictor here is the explicitly corrected normalized variant, not the literal Eq. (4.4) implementation.

## Warm holdout / 常规留出集

- Source: MovieLens 100K; required counts: 943 users, 1,682 movies, 100,000 ratings.
- Split: per-user train/test = approximately 80/20; PRNG seed 2026, train-only top-12 popularity seeds.
- Evaluation users: the first **200** users with held-out ratings, sorted by ID. **Not** the paper's randomly sampled 200 users.
- Prediction: positive-similarity normalized scoring with item-mean fallback; neighbors k = 5/10/20.

| Neighbors (k) | UB-CF MAE ↓ | DR-CF MAE ↓ | SR-CF MAE ↓ |
| ---: | ---: | ---: | ---: |
| 5 | 0.8644 | 0.9311 | 0.8698 |
| 10 | 0.8805 | 0.9576 | 0.8891 |
| 20 | 0.8859 | 0.9354 | 0.8995 |

![Actual MovieLens-100K warm benchmark](../artifacts/real/ml100k/demo_mae.svg)

## Cold-start / 新用户冷启动

- Reference neighborhood: user-disjoint (no interactions from test users in the reference population).
- Target users: first **30** users with at least 18 ratings, ID order; evaluation holds out the *same five item ratings* for every observation budget.
- The observation budgets 0/1/3/5/10 refer only to revealed target-user feedback; holdout outcomes remain unseen.
- Predictors: normalized variants; neighbor k=10. No validation-tuned fusion weights.

| Observed ratings | UB-CF MAE ↓ | DR-CF MAE ↓ | SR-CF MAE ↓ |
| ---: | ---: | ---: | ---: |
| 0 | 0.7830 | 0.8996 | 0.8996 |
| 1 | 0.7830 | 0.8818 | 0.8818 |
| 3 | 0.8587 | 0.8799 | 0.9175 |
| 5 | 0.8933 | 0.8802 | 0.9209 |
| 10 | 0.8996 | 0.8793 | 1.0131 |

![Actual MovieLens-100K cold-start experiment](../artifacts/real/cold/cold_start.svg)

## What to investigate next / 后续验证

The observed ranking varies with k and target history length. To test whether adaptive weights improve cold-start robustness, use **user-disjoint train/validation/test partitions**, paired sampling and confidence intervals before selecting a rule. Verify baseline fairness and seed-item overlap; do not claim that a single holdout proves an algorithmic improvement.
