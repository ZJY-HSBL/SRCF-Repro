# v0.3.0 — GitHub mathematical typesetting and repeated MovieLens-100K evaluation

## Verified implementation and release evidence

- Changed README equations from ambiguous display delimiters to GitHub-renderable `math` fences; included checked-in mathematical SVG fallbacks.
- Ran **18 local automated tests**, plus passing GitHub Actions tests and the newly added repeated-study workflow.
- Official MovieLens-100K dataset verified on a GitHub runner: **943 users, 1,682 movies, 100,000 ratings**. Original data are not redistributed.
- Four non-overlapping batches of 24 target users, **96 unique target users** total, each with five fixed held-out ratings and 0/1/3/5/10 observed training ratings.
- Six predeclared configurations: UB-CF, DR-CF, original SR-CF, without seeds, without history, and fixed schedule. Produced **2,880 subject-method-budget rows** and anonymized per-user MAE.
- User-level paired 2,000-resample bootstrap intervals and SVG charts are stored in `docs/REPEATED_RESULTS.md` and `artifacts/real/repeated/`.

## Actual observed results, normalized prediction, k=10

- 3 visible ratings: SR-CF MAE **0.9481**, DR-CF **0.9541**, UB-CF **0.9223**.
- 5 visible ratings: SR-CF **0.9349**, DR-CF **0.9542**, UB-CF **0.9498**.
- 10 visible ratings: SR-CF **0.9888**, DR-CF **0.9540**, UB-CF **1.0219**.
- At 10, user-level SR-CF minus DR-CF difference **+0.0348**, bootstrap 95% interval **[-0.0421, +0.1138]**, so no confirmed superiority.
- Predeclared schedule is not a validated improvement.

The original paper's exact experiments remain underspecified; these independently defined real-data tests are **not** advertised as exact replication. Neither the experimental model choice nor the weight schedule was tuned on target outcomes.
