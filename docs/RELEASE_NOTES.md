# v0.5.0 — Native LaTeX and validation-frozen confidence study

- README equations converted to editable, native GitHub LaTeX (`$$...$$`), rather than SVG screenshots or fenced code. GitHub **web** rendering is documented; not all mobile clients are guaranteed to display MathJax.
- Introduced reliability-dependent Pearson component shrinkage with a per-pair normalized mixture, explicitly **not** part of the original thesis.
- Strictly excluded the earlier 144 exploratory users, including them neither in new reference nor evaluation cohorts.
- Selected shrinkage parameter **only** on 32 new validation users; evaluate the frozen candidate on **48 untouched test users**, 0/1/3/5/10 observed ratings, compared against SR-CF, UB-CF, DR-CF and a no-seed ablation.
- Added per-user CSV, full protocol, exploratory paired bootstrap (clustered by user across observation budgets), SVG test curves and a bilingual analysis report.
- Do **not** claim positive performance improvements until actually measured, and do not confuse in-dataset independent-user holdout with external-dataset replication.
