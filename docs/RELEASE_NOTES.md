# v0.3.0 — README equation repair and repeated cold-start experiments

- Render mathematical expressions with GitHub-compatible `math` fenced blocks and checked-in SVG fallbacks.
- Add a predeclared user-disjoint cold-start ablation protocol using five feedback budgets and six algorithms.
- Include four nonoverlapping target-user batches (96 distinct users) on *official MovieLens-100K*, with anonymized user-averaged MAE, subject-level paired percentile-bootstrap intervals, and measured SVG figures.
- Clearly distinguish prior v0.2 warm/cold results, synthetic development checks, paper-reported results, and this release's independently measured results.
- No test-set parameter tuning, and no claim of a complete numerical reproduction of the thesis.

Find measured results in `docs/REPEATED_RESULTS.md` and `artifacts/real/repeated/`. The raw MovieLens data are downloaded by CI, not redistributed.
