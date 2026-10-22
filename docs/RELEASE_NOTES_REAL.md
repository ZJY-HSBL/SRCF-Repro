# v0.2.0 — Measured MovieLens-100K benchmark

- Downloaded the official MovieLens-100K dataset in a GitHub Actions runner and verified the complete user, item and rating counts.
- Ran SR-CF, UB-CF and DR-CF on a deterministic per-user holdout using the explicitly normalized rating prediction formula.
- Ran a user-disjoint cold-start experiment with a fixed 5-rating test target per user, varying visible user feedback over {0, 1, 3, 5, 10}.
- Stored actual experimental CSV, JSON, and SVG figures plus a newly generated bilingual measured-results report.
- Clearly distinguish the independent real-data validation from the paper's incompletely specified original experimental protocol.

Data were not committed or redistributed. Refer to `docs/REAL_RESULTS.md` for limitations and values.
