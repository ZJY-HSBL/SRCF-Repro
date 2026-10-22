# Seed-signal availability and sensitivity / 种子评分信号可用性研究

**Data source / 数据源：MOVIELENS-100K. Measured, exploratory, and not an exact thesis replication.**

本实验专门检验固定 12 部种子电影是否会因为新用户交互太少而导致种子评分相似度无法计算。
The test asks whether a small set of popular seed movies can provide enough *co-rated seed items* for the SR-CF similarity component to be meaningful in cold-start settings.

## Frozen protocol / 预设实验方案

- Excluded all **96** target users from the earlier study, both as reference neighbors and as candidates; selected **48 previously unevaluated target users** with seed 271828.
- All current targets are excluded from the persistent reference pool. Within each batch, targets with revealed ratings may serve as neighbors of other batch targets, but **no target's held-out ratings** are ever exposed to model training. This is *not* cross-dataset validation.
- Same five held-out ratings for each user at observation budgets **0, 1, 3, 5, 10**; 10 nearest neighbors; **fixed** SR-CF mixing weights (0.2,0.3,0.5). We only vary seed-set size **12/40/100**, and compare against UB-CF and DR-CF.
- Seeds selected from reference ratings only, by popularity; no tuning on target scores. Evaluation is averaged *per target user*. Predictions for held-out items with no training vocabulary entry are excluded from these MAE estimates.

## Model MAE / 模型比较 (lower is better)

| Known ratings | UB-CF | DR-CF | SR(12) | SR(40) | SR(100) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.8002 | 0.9330 | 0.9330 | 0.9330 | 0.9330 |
| 1 | 0.8002 | 0.9252 | 0.9252 | 0.9252 | 0.9252 |
| 3 | 0.8653 | 0.9251 | 0.9290 | 0.9287 | 0.9225 |
| 5 | 0.8757 | 0.9250 | 0.8333 | 0.8401 | 0.8345 |
| 10 | 0.9155 | 0.9249 | 0.8513 | 0.8700 | 0.8853 |

![Measured MAE versus seed-set size](../artifacts/real/seed-diagnostics/seed_count_mae.svg)

## Seed support / 种子信息可用率

A target can contribute a **nonzero** seed-rating Pearson similarity to at least one reference neighbor only when the seed overlap spans at least two shared items with nonconstant co-ratings. The table reports the percentage of targets with at least one such candidate reference user; it does **not** imply that this candidate is ranked in top-k.

| Known ratings | Seed movies | Targets with usable seed similarities | Mean usable reference neighbors |
|---:|---:|---:|---:|
| 0 | 12 | 0.0% | 0.0 |
| 0 | 40 | 0.0% | 0.0 |
| 0 | 100 | 0.0% | 0.0 |
| 1 | 12 | 0.0% | 0.0 |
| 1 | 40 | 0.0% | 0.0 |
| 1 | 100 | 0.0% | 0.0 |
| 3 | 12 | 0.0% | 0.0 |
| 3 | 40 | 4.2% | 7.3 |
| 3 | 100 | 12.5% | 14.3 |
| 5 | 12 | 2.1% | 4.1 |
| 5 | 40 | 16.7% | 23.7 |
| 5 | 100 | 41.7% | 55.1 |
| 10 | 12 | 14.6% | 26.9 |
| 10 | 40 | 45.8% | 83.8 |
| 10 | 100 | 81.2% | 156.4 |

![Measured seed similarity availability](../artifacts/real/seed-diagnostics/seed_signal_availability.svg)

## Paired error differences / 同一目标用户配对误差

(A − B) below zero favors A. User-level 2,000-draw bootstrap, **exploratory unadjusted** confidence intervals; multiple hypotheses and budgets were inspected.

| Budget | Difference (A − B) | Mean ΔMAE | Exploratory 95% bootstrap CI |
|---:|---|---:|---:|
| 3 | sr12 − dr | +0.0039 | [-0.0840, +0.0912] |
| 3 | sr40 − sr12 | -0.0003 | [-0.0009, +0.0000] |
| 3 | sr100 − sr12 | -0.0066 | [-0.0178, +0.0000] |
| 5 | sr12 − dr | -0.0916 | [-0.1856, -0.0007] |
| 5 | sr40 − sr12 | +0.0067 | [-0.0083, +0.0247] |
| 5 | sr100 − sr12 | +0.0012 | [-0.0332, +0.0325] |
| 10 | sr12 − dr | -0.0736 | [-0.1591, +0.0152] |
| 10 | sr40 − sr12 | +0.0187 | [-0.0299, +0.0715] |
| 10 | sr100 − sr12 | +0.0340 | [-0.0121, +0.0804] |

## Research implications / 研究启示

1. A near-zero seed similarity term may be uninformative even when assigned a nominal mixing weight of 0.3. Increase in seed size changes information availability **and** the statistical comparison target, so do not treat size effects as evidence of a new method without external validation.
2. Model rankings depend on neighborhood selection, data sparsity and seed choice. Analyze the tradeoff between *shared seed coverage* and candidate similarity reliability before fitting any adaptive weight rule.
3. For a confirmatory follow-up, freeze a single seed-selection policy on a **validation-only** cohort and evaluate on users and/or data not used to select it. Clearly report effect sizes and confidence intervals.

Raw evidence / 全部原始数据：[`per_user.csv`](../artifacts/real/seed-diagnostics/per_user.csv), [`seed_diagnostics.csv`](../artifacts/real/seed-diagnostics/seed_diagnostics.csv), [`seed_support.csv`](../artifacts/real/seed-diagnostics/seed_support.csv), [`paired_bootstrap.csv`](../artifacts/real/seed-diagnostics/paired_bootstrap.csv), [`protocol.json`](../artifacts/real/seed-diagnostics/protocol.json).
