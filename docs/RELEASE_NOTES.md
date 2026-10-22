# v0.4.0 — Mobile-readable formulas and seed-set diagnostic

- Removed raw LaTeX code blocks that the GitHub iPad/iOS client displayed literally; used fixed-background SVG mathematical equations as the only visual formula representation.
- Added regression tests to ensure math images have a white background and README does not contain raw math fences.
- Evaluated (upon verified workflow completion) an entirely new, previously unevaluated target-user cohort from official MovieLens-100K, excluding the 96 old target users from both target and reference pools.
- Introduced predeclared comparisons of top 12, 40 and 100 most-popular train-only seed movies, along with DR-CF and UB-CF, under the same fixed five-rating test holdouts and nested cold-start budgets.
- Logged per-user prediction error, the fraction of users with usable seed-rating similarities, paired exploratory bootstraps and visualized curves. No claims of numerical replication or statistically confirmed improvement.

See `docs/SEED_DIAGNOSTICS.md` for actual measured numbers only after the official-data study workflow has completed.
