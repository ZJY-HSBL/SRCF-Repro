# Replication protocol / 复现规范

**Reference:** Qiao Yu (乔雨), *Research on Recommendation Algorithms Against Cold-Start Problem* /《面向冷启动问题的推荐算法研究》, Nanjing University of Posts and Telecommunications, February 2018. Chapter 4, printed pp. 29–37.

## What is thesis-defined / 论文明确给出的内容

- Age similarity (Eq. 4.5) is `exp(-abs(segment(age_u)-segment(age_v)))`, with 5-year segments; sex equality and tag similarity are discussed in §4.2.1.
- Demographic similarity is a weighted sum (Eq. 4.7). **Weights of individual attributes are not specified**, so our reference implementation uses the mean of available age, gender and occupation similarities. MovieLens 100K provides no hobby fields.
- Seed-item similarity and rating-history similarity use Eq. 4.3, a mean-centred co-rated similarity. Undefined correlations are set to zero (implementation choice).
- Fused similarity is Eq. 4.8: `0.2 D + 0.3 S + 0.5 R`, or any weights on the unit simplex. We use the fixed reported triple without test-set tuning.
- The **literal** rating predictor in Eq. 4.4 is the *un-normalized* sum of neighbor similarity times ratings. `--prediction paper` preserves it, including out-of-range predictions and zeros.
- `--prediction normalized` is **our explicit variant**, not an original thesis formula: positive-similarity weighted average, fallback to item/global training mean, clipped to [1,5]. Never pool metrics from the two variants.

## Unspecified decisions / 未明确的实验细节

1. The thesis calls its dataset “ml-1m” but gives **943 users, 1,682 items and 100,000 ratings** (MovieLens 100K statistics, not 1M). This project uses **MovieLens 100K** and does not claim exact numeric replication.
2. Neither the seed selection rule nor its cardinality is fixed. Our configurable prototype chooses the **12 most frequently rated items based on training observations alone**.
3. The user demographics include age, sex and occupation. Hobby tags are not present in MovieLens 100K; we do **not** fabricate them.
4. The thesis does not provide a fully reproducible train/test split or random seeds. We use deterministic 80/20 per-user holdout, default seed **2026**; users with only one rating stay in train. No held-out rating is used to select neighbors, seeds or fit weights.
5. The thesis chart reports the MAE trend without a downloadable experiment record. Its coverage table is transcribed into `artifacts/paper_coverage.csv` with *reported* provenance, not measured provenance.
6. Baselines share the same train/holdout and candidate universe. `UB-CF` uses **all** historical co-ratings; `DR-CF` uses demographic similarity; `SR-CF` combines demographic, seed and full-history similarities. The original text does not specify whether seed ratings should be removed from historical ratings; we retain them in both, documenting the double use.
7. Coverage here means `number of distinct items in top-5 recommendations across evaluated users / number of catalog items with training ratings`. This is not guaranteed identical to the paper's coverage protocol.
8. Current reference implementation computes user-neighbor similarities in Python and is for reproducibility, not industrial-scale throughput. Report wall time separately from predictive accuracy.

## Experiment scope / 实验范围

`synthetic(seed=2026)` is a constructed 42-user, 60-item toy dataset with 1,092 interactions. Its behavior is a **software smoke experiment**, not evidence of superiority or a replication of the paper's test scores. Its MAE and coverage plots are labeled SYNTHETIC.

For a real-data benchmark download *ml-100k* from the official GroupLens site, unpack it locally (do not commit it), then run:

```bash
python -m srcf_repro.experiment --data-dir ./ml-100k --max-users 200 --out artifacts/local/ml100k
python -m srcf_repro.experiment --data-dir ./ml-100k --max-users 200 --prediction paper --out artifacts/local/ml100k-paper
```

The first uses an explicitly corrected predictor; the second implements Eq. 4.4 literally. The 200 users are the **first 200 by ID** for deterministic sampling, **not** the paper's unspecified random 200. With real data, validate identifiers, sizes, ratings and run independent seed replicates before drawing conclusions.

## Upcoming research / 后续实验方向

1. Implement a separate user-disjoint cold-start protocol (`cold_start_curve` is provided as a base), varying the number of observed target-user interactions in {0,1,3,5,10} against a **fixed** held-out target set.
2. Tune fusion weights only on a validation split, freeze them, and evaluate once on independent test users; report paired confidence intervals instead of only a best observed score.
3. Test seed-set construction (popularity, diversity, entropy) separately from the fusion rule.
4. Perform mechanism ablations (D only, S only, R only, D+S, D+R, full), speed/memory profiling, and larger contemporary baselines.
