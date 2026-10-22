"""Predeclared exploratory SR-CF cold-start study with user-disjoint splits.

Never tunes model weights on the evaluation subjects. MovieLens data are
loaded locally/on CI, not distributed with this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .data import load_movielens100k, popularity_seeds
from .experiment import synthetic
from .model import Recommender

OBSERVATIONS = (0, 1, 3, 5, 10)
METHODS = ("ub", "dr", "sr", "sr_no_seed", "sr_no_history", "sr_schedule")
LABELS = {
    "ub": "UB-CF (history)",
    "dr": "DR-CF (profile)",
    "sr": "SR-CF (0.2/0.3/0.5)",
    "sr_no_seed": "SR-CF - seed component",
    "sr_no_history": "SR-CF - history component",
    "sr_schedule": "SR-CF preset schedule",
}


def scheduled_weights(n: int) -> tuple[float, float, float]:
    """Exploratory, preset *before* seeing results, never fitted on targets."""
    if n == 0:
        return (1.0, 0.0, 0.0)
    if n <= 2:
        return (.7, .2, .1)
    if n <= 5:
        return (.5, .2, .3)
    return (.2, .3, .5)


def method_args(method: str, observed: int) -> tuple[str, tuple[float, float, float]]:
    if method == "ub":
        return "ub", (.2, .3, .5)
    if method == "dr":
        return "dr", (.2, .3, .5)
    if method == "sr":
        return "sr", (.2, .3, .5)
    if method == "sr_no_seed":
        return "sr", (.2 / .7, 0.0, .5 / .7)
    if method == "sr_no_history":
        return "sr", (.2 / .5, .3 / .5, 0.0)
    if method == "sr_schedule":
        return "sr", scheduled_weights(observed)
    raise ValueError(f"unknown method: {method}")


def build_cases(ratings: dict, *, study_seed: int, batches: int, users_per_batch: int,
                holdout: int = 5) -> tuple[dict, list[list[tuple[str, int, dict, list]]]]:
    """Draw disjoint users once, excluding *all* targets from reference ratings.

    Each target has a fixed holdout of 5 ratings. The observed training
    prefixes are nested and cannot intersect the holdout. Cases are anonymized.
    """
    if batches <= 0 or users_per_batch <= 0 or holdout < 1:
        raise ValueError("positive batches, users per batch and holdout required")
    rng = random.Random(study_seed)
    eligible = sorted(u for u, items in ratings.items() if len(items) >= 10 + holdout)
    size = batches * users_per_batch
    if size >= len(eligible):
        raise ValueError("target users must leave at least one reference user")
    picked = rng.sample(eligible, size)
    reference = {u: dict(items) for u, items in ratings.items() if u not in set(picked)}
    batch_cases = []
    for batch in range(batches):
        cases = []
        for offset, u in enumerate(picked[batch * users_per_batch:(batch+1)*users_per_batch]):
            shuffled = sorted(ratings[u]); rng.shuffle(shuffled)
            test_ids = shuffled[-holdout:]
            training_ids = shuffled[:-holdout]
            cid = f"case_{batch+1:02d}_{offset+1:03d}"
            cases.append((cid, u, {i: ratings[u][i] for i in test_ids}, training_ids))
        batch_cases.append(cases)
    return reference, batch_cases


def run_study(ratings: dict, profiles: dict, *, study_seed: int = 2026,
              batches: int = 4, users_per_batch: int = 24, k: int = 10,
              observations: tuple[int, ...] = OBSERVATIONS,
              methods: tuple[str, ...] = METHODS) -> list[dict]:
    """Return one row per anonymized evaluation subject, budget and method."""
    reference, case_batches = build_cases(ratings, study_seed=study_seed,
                                          batches=batches, users_per_batch=users_per_batch)
    seeds = popularity_seeds(reference, count=12)
    result = []
    for batch_idx, cases in enumerate(case_batches, start=1):
        for n in observations:
            if n < 0 or n > 10:
                raise ValueError("observation budget outside predefined 0..10 range")
            train = {u: dict(items) for u, items in reference.items()}
            for _, u, _, training_ids in cases:
                train[u] = {i: ratings[u][i] for i in training_ids[:n]}
            for method in methods:
                mode, weights = method_args(method, n)
                model = Recommender(train, profiles, seeds=seeds, mode=mode,
                                    weights=weights, prediction="normalized")
                for cid, u, holdouts, _ in cases:
                    errors = [abs(model.predict(u, i, k) - r)
                              for i, r in holdouts.items() if i in model.item_mean]
                    if not errors:
                        raise ValueError("no held-out item exists in reference training")
                    result.append({"case": cid, "batch": batch_idx,
                                   "observed": n, "method": method,
                                   "mae": round(mean(errors), 9),
                                   "n_test": len(errors), "k": k,
                                   "study_seed": study_seed})
    return result


def paired_ci(rows: list[dict], a: str, b: str, n: int,
              *, draws: int = 2000, bootstrap_seed: int = 9301) -> dict:
    """Cluster bootstrap across unique target users: A minus B (negative better)."""
    grouped = defaultdict(dict)
    for row in rows:
        if row["observed"] == n and row["method"] in (a, b):
            if row["method"] in grouped[row["case"]]:
                raise ValueError("duplicate subject/method/budget")
            grouped[row["case"]][row["method"]] = row["mae"]
    if not grouped or any(set(vals) != {a, b} for vals in grouped.values()):
        raise ValueError("unpaired data")
    ds = [p[a] - p[b] for _, p in sorted(grouped.items())]
    observed = mean(ds)
    if draws <= 0:
        raise ValueError("draws must be positive")
    rng = random.Random(bootstrap_seed)
    rep = sorted(mean(rng.choices(ds, k=len(ds))) for _ in range(draws))
    return {"a": a, "b": b, "observed": n, "n_users": len(ds),
            "mean_diff": round(observed, 6),
            "ci_low": round(rep[int(.025 * (draws-1))], 6),
            "ci_high": round(rep[int(.975 * (draws-1))], 6),
            "wins_a": sum(d < 0 for d in ds),
            "ties": sum(d == 0 for d in ds)}


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for n in sorted({x["observed"] for x in rows}):
        for m in METHODS:
            scores = [row["mae"] for row in rows
                      if row["observed"] == n and row["method"] == m]
            if scores:
                out.append({"observed": n, "method": m,
                            "user_mean_mae": round(mean(scores), 6),
                            "n_users": len(scores)})
    return out


def render_svg(rows: list[dict], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summarize(rows)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for method in METHODS:
        group = [r for r in s if r["method"] == method]
        if group:
            ax.plot([r["observed"] for r in group],
                    [r["user_mean_mae"] for r in group], marker="o", label=LABELS[method])
    ax.set(title=title, xlabel="Visible target-user ratings", ylabel="User-average MAE (lower better)")
    ax.grid(alpha=.25); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg"); plt.close(fig)


def export(rows: list[dict], out: Path, *, source: str, batches: int,
           users_per_batch: int, study_seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    comparisons = [paired_ci(rows, "sr", baseline, n) for baseline in ("ub", "dr")
                   for n in OBSERVATIONS]
    comparisons += [paired_ci(rows, "sr_schedule", "sr", n) for n in OBSERVATIONS]
    for filename, collection in (("per_user.csv", rows), ("summary.csv", summary),
                                 ("paired_bootstrap.csv", comparisons)):
        with (out / filename).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(collection[0]))
            w.writeheader(); w.writerows(collection)
    config = {"source": source, "seed": study_seed, "batches": batches,
              "users_per_batch": users_per_batch, "k": rows[0]["k"],
              "holdout_ratings_per_user": 5, "observations": OBSERVATIONS,
              "model_weights": {m: {str(n): method_args(m, n)[1] for n in OBSERVATIONS}
                                for m in METHODS},
              "bootstrap_draws": 2000, "bootstrap_seed": 9301,
              "protocol": "all targets excluded from reference; unique targets across batches; fixed heldouts; per-user MAE; no tuned weights"}
    (out / "protocol.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    render_svg(rows, out / "cold_start_ablations.svg", f"Measured {source}: independent cold-start evaluation")
    header = "| Visible | " + " | ".join(METHODS) + " |\n"
    header += "|---:|" + "---:|" * len(METHODS) + "\n"
    for n in OBSERVATIONS:
        vals = {r["method"]: r["user_mean_mae"] for r in summary if r["observed"] == n}
        header += f"| {n} | " + " | ".join(f"{vals[m]:.4f}" for m in METHODS) + " |\n"
    ci_table = "| Observed | A − B | Mean difference | 95% bootstrap CI | Wins A / users |\n|---:|---|---:|---:|---:|\n"
    for c in comparisons:
        if c["observed"] not in (3, 10):
            continue
        ci_table += (f"| {c['observed']} | {c['a']} − {c['b']} | {c['mean_diff']:+.4f} | "
                     f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                     f"{c['wins_a']} / {c['n_users']} |\n")
    report = f"""# Repeated cold-start ablation / 多批次冷启动消融实验

**Actually measured: {source}.** The full source dataset is not checked into the repository.

**Protocol / 方法：** {batches} non-overlapping target-user batches, {users_per_batch} users per batch ({batches*users_per_batch} unique users); fixed five-rating test holdout per user; 0/1/3/5/10 *nested* visible interactions; all target users excluded from the reference population across all batches. Global study seed **{study_seed}**, neighborhoods **k=10**, train-only top-12 popular seed items, corrected normalized predictions. Mean MAE is **user-averaged**, so users with more usable test ratings do not receive more weight. No model was chosen using test ratings.

本实验严格隔离参考用户与目标用户，所有批次的目标用户互不重叠；每位目标用户固定五条测试评分，仅逐步公开训练交互记录。固定权重和预设分段权重均属于**探索性假设**，并未在测试集上调参。

## MAE vs observation budget / MAE 与历史交互量

{header}

![Measured cold-start ablations](../{out.as_posix()}/cold_start_ablations.svg)

## Paired comparisons / 配对比较

Per-user paired MAE difference (A − B, negative favors A). Nonparametric subject-level percentile bootstrap with 2,000 resamples, seed 9301. CIs are exploratory and **not** multiplicity adjusted.

{ci_table}

## Caveats and next experiment / 解释与下一步

- This is **not** exact replication of the source thesis (seed selection and evaluation split were not fully specified there).
- Predeclared `sr_schedule` weights use (1,0,0) at 0 ratings; (.7,.2,.1) at 1; (.5,.2,.3) at 3/5; and (.2,.3,.5) at 10. These have not been fitted to a validation set; if they improve, that is only an exploratory finding.
- Holdouts missing from the reference item's vocabulary are excluded by the predictor, and `n_test` records the number retained. This is a conditional, not full-catalog new-item evaluation.
- Bootstrap intervals condition on this dataset and protocol; use a *separate* held-out dataset and randomized users for confirmatory analysis.
- Next: investigate overlapping seed ratings, profile weights, negative-neighbor handling, and nested cross-validation (validation-only model selection), then test on unseen datasets.

Raw audit / 原始记录：[per-user metrics](../{out.as_posix()}/per_user.csv) · [summary](../{out.as_posix()}/summary.csv) · [pairwise bootstrap](../{out.as_posix()}/paired_bootstrap.csv) · [protocol](../{out.as_posix()}/protocol.json).
"""
    Path("docs").mkdir(exist_ok=True)
    Path("docs/REPEATED_RESULTS.md").write_text(report, encoding="utf-8")
    marker = "<!-- STUDY_RESULTS_START -->"
    end = "<!-- STUDY_RESULTS_END -->"
    p = Path("README.md")
    readme = p.read_text(encoding="utf-8")
    if marker not in readme or end not in readme:
        raise ValueError("README study markers missing")
    snippet = ("### 多批次冷启动实验 · Repeated cold-start study\n\n"
               f"**{source}**, {batches*users_per_batch} independent target users, "
               "fixed holdouts, no test-set tuning. 结果全部保留，既包括支持假设也包括反例。\n\n"
               + header + "\n"
               f"![Measured {source} cold-start comparisons]({out.as_posix()}/cold_start_ablations.svg)\n\n"
               "[完整实验报告、配对 Bootstrap 置信区间 / full study and CIs](docs/REPEATED_RESULTS.md)\n")
    p.write_text(readme.split(marker)[0] + marker + "\n" + snippet + "\n" + end + readme.split(end,1)[1], encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Repeated user-disjoint cold-start study")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--out", type=Path, default=Path("artifacts/local/repeated"))
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--users-per-batch", type=int, default=24)
    args = p.parse_args(argv)
    ratings, profiles = (load_movielens100k(args.data_dir) if args.data_dir else synthetic(args.seed))
    source = "MOVIELENS-100K" if args.data_dir else "SYNTHETIC"
    rows = run_study(ratings, profiles, study_seed=args.seed,
                     batches=args.batches, users_per_batch=args.users_per_batch)
    export(rows, args.out, source=source, batches=args.batches,
           users_per_batch=args.users_per_batch, study_seed=args.seed)
    print(json.dumps({"source": source, "rows": len(rows),
                      "subjects": args.batches * args.users_per_batch,
                      "output": args.out.as_posix()}))


if __name__ == "__main__":
    main()
