# Repeated cold-start ablation / 多批次冷启动消融实验

**Actually measured: SYNTHETIC.** The full source dataset is not checked into the repository.

**Protocol / 方法：** 2 non-overlapping target-user batches, 8 users per batch (16 unique users); fixed five-rating test holdout per user; 0/1/3/5/10 *nested* visible interactions; all target users excluded from the reference population across all batches. Global study seed **2026**, neighborhoods **k=10**, train-only top-12 popular seed items, corrected normalized predictions. Mean MAE is **user-averaged**, so users with more usable test ratings do not receive more weight. No model was chosen using test ratings.

本实验严格隔离参考用户与目标用户，所有批次的目标用户互不重叠；每位目标用户固定五条测试评分，仅逐步公开训练交互记录。固定权重和预设分段权重均属于**探索性假设**，并未在测试集上调参。

## MAE vs observation budget / MAE 与历史交互量

| Visible | ub | dr | sr | sr_no_seed | sr_no_history | sr_schedule |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.1087 | 0.6762 | 0.6762 | 0.6762 | 0.6762 | 0.6762 |
| 1 | 1.1099 | 0.6648 | 0.6648 | 0.6648 | 0.6648 | 0.6648 |
| 3 | 1.0505 | 0.6699 | 0.7281 | 0.7285 | 0.6710 | 0.6789 |
| 5 | 1.0265 | 0.6764 | 0.8417 | 0.8398 | 0.6767 | 0.6688 |
| 10 | 0.7818 | 0.6570 | 0.7759 | 0.7482 | 0.7411 | 0.7759 |


![Measured cold-start ablations](../artifacts/experimental/synthetic-repeated/cold_start_ablations.svg)

## Paired comparisons / 配对比较

Per-user paired MAE difference (A − B, negative favors A). Nonparametric subject-level percentile bootstrap with 2,000 resamples, seed 9301. CIs are exploratory and **not** multiplicity adjusted.

| Observed | A − B | Mean difference | 95% bootstrap CI | Wins A / users |
|---:|---|---:|---:|---:|
| 3 | sr − ub | -0.3224 | [-0.4506, -0.1931] | 15 / 16 |
| 10 | sr − ub | -0.0059 | [-0.0549, +0.0589] | 11 / 16 |
| 3 | sr − dr | +0.0582 | [-0.0463, +0.1901] | 6 / 16 |
| 10 | sr − dr | +0.1189 | [+0.0547, +0.1831] | 4 / 16 |
| 3 | sr_schedule − sr | -0.0493 | [-0.1410, +0.0320] | 9 / 16 |
| 10 | sr_schedule − sr | +0.0000 | [+0.0000, +0.0000] | 0 / 16 |


## Caveats and next experiment / 解释与下一步

- This is **not** exact replication of the source thesis (seed selection and evaluation split were not fully specified there).
- Predeclared `sr_schedule` weights use (1,0,0) at 0 ratings; (.7,.2,.1) at 1; (.5,.2,.3) at 3/5; and (.2,.3,.5) at 10. These have not been fitted to a validation set; if they improve, that is only an exploratory finding.
- Holdouts missing from the reference item's vocabulary are excluded by the predictor, and `n_test` records the number retained. This is a conditional, not full-catalog new-item evaluation.
- Bootstrap intervals condition on this dataset and protocol; use a *separate* held-out dataset and randomized users for confirmatory analysis.
- Next: investigate overlapping seed ratings, profile weights, negative-neighbor handling, and nested cross-validation (validation-only model selection), then test on unseen datasets.

Raw audit / 原始记录：[per-user metrics](../artifacts/experimental/synthetic-repeated/per_user.csv) · [summary](../artifacts/experimental/synthetic-repeated/summary.csv) · [pairwise bootstrap](../artifacts/experimental/synthetic-repeated/paired_bootstrap.csv) · [protocol](../artifacts/experimental/synthetic-repeated/protocol.json).
