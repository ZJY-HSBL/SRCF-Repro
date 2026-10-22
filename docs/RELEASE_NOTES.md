# v0.1.0 — SR-CF reference reproduction scaffold

- Implemented thesis §4.2 similarity fusion, demographic similarity, seed ratings, and three comparable UB-CF / DR-CF / SR-CF models.
- Exposed the literal, unnormalized Eq. (4.4) predictor separately from an explicitly labeled bounded/normalized variant.
- Added deterministic train/test separation, MovieLens-100K loading, a synthetic smoke benchmark, tests, and bilingual documentation.
- Included an SVG methods flow diagram, paper-reported coverage visualization, and measured SYNTHETIC MAE/coverage artifacts.
- Documented missing protocol details and the MovieLens 100K vs 1M naming contradiction.

This is **not** a claim of reproducing the paper's exact numerical results. MovieLens evaluation remains to be run independently.
