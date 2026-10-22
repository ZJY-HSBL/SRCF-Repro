"""Publish true-data benchmark reports from artifacts; never fabricate results."""

from __future__ import annotations
import argparse
import csv
from pathlib import Path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        result = list(csv.DictReader(f))
    if not result:
        raise ValueError(f"empty measured experiment: {path}")
    for row in result:
        if not 0 <= float(row["coverage"]) <= 1 or float(row["mae"]) < 0 or int(row["n_test"]) <= 0:
            raise ValueError(f"invalid metrics: {row}")
    return result


def _table(rows: list[dict[str, str]], x: str) -> str:
    column = "Neighbors (k)" if x == "k" else "Observed ratings"
    values = sorted({int(row[x]) for row in rows})
    lines = [f"| {column} | UB-CF MAE ↓ | DR-CF MAE ↓ | SR-CF MAE ↓ |",
             "| ---: | ---: | ---: | ---: |"]
    for v in values:
        matches = {r["mode"]: r for r in rows if int(r[x]) == v}
        if set(matches) != {"ub", "dr", "sr"}:
            raise ValueError(f"missing model results for {x}={v}")
        lines.append(f"| {v} | {float(matches['ub']['mae']):.4f} | "
                     f"{float(matches['dr']['mae']):.4f} | {float(matches['sr']['mae']):.4f} |")
    return "\n".join(lines)


def build(warm_path: Path, cold_path: Path, doc_path: Path, readme_path: Path) -> None:
    warm, cold = _rows(warm_path), _rows(cold_path)
    if {r["prediction"] for r in warm} != {"normalized"}:
        raise ValueError("warm benchmark must use the labeled normalized predictor")
    if {r["seed"] for r in warm + cold} != {"2026"}:
        raise ValueError("unexpected seed, edit protocol and outputs together")
    warm_users = {int(r["n_users"]) for r in warm}
    cold_users = {int(r["n_users"]) for r in cold}
    if len(warm_users) != 1 or len(cold_users) != 1:
        raise ValueError("inconsistent user counts")
    text = f"""# MovieLens-100K measured experiments / 真实数据实验

**Status / 状态: measured using the official MovieLens-100K dataset (not synthetic).**

This report is generated from `artifacts/real/ml100k/benchmark.csv` and `artifacts/real/cold/cold_start.csv` after executing the versioned implementation on downloaded **official MovieLens-100K** data. The raw dataset is not redistributed.

**Important scientific limitation / 科研限制:** These are independent, protocol-defined real-data experiments, **not exact reproduction of the thesis's numeric plots**. The thesis does not specify its seed-set construction, train/test partition, user sample or all demographic feature weights; the predictor here is the explicitly corrected normalized variant, not the literal Eq. (4.4) implementation.

## Warm holdout / 常规留出集

- Source: MovieLens 100K; required counts: 943 users, 1,682 movies, 100,000 ratings.
- Split: per-user train/test = approximately 80/20; PRNG seed 2026, train-only top-12 popularity seeds.
- Evaluation users: the first **{next(iter(warm_users))}** users with held-out ratings, sorted by ID. **Not** the paper's randomly sampled 200 users.
- Prediction: positive-similarity normalized scoring with item-mean fallback; neighbors k = 5/10/20.

{_table(warm, 'k')}

![Actual MovieLens-100K warm benchmark](../artifacts/real/ml100k/demo_mae.svg)

## Cold-start / 新用户冷启动

- Reference neighborhood: user-disjoint (no interactions from test users in the reference population).
- Target users: first **{next(iter(cold_users))}** users with at least 18 ratings, ID order; evaluation holds out the *same five item ratings* for every observation budget.
- The observation budgets 0/1/3/5/10 refer only to revealed target-user feedback; holdout outcomes remain unseen.
- Predictors: normalized variants; neighbor k=10. No validation-tuned fusion weights.

{_table(cold, 'n_observed')}

![Actual MovieLens-100K cold-start experiment](../artifacts/real/cold/cold_start.svg)

## What to investigate next / 后续验证

The observed ranking varies with k and target history length. To test whether adaptive weights improve cold-start robustness, use **user-disjoint train/validation/test partitions**, paired sampling and confidence intervals before selecting a rule. Verify baseline fairness and seed-item overlap; do not claim that a single holdout proves an algorithmic improvement.
"""
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(text, encoding="utf-8")
    readme = readme_path.read_text(encoding="utf-8")
    readme = readme.replace("**Not yet executed / 尚未运行**", "**Real experiment complete / 已完成真实实验**")
    section = """## MovieLens-100K 真正实测 · Real-data evaluation

真实数据已在 GitHub Actions 环境中运行，含常规留出集和固定测试项目的 0/1/3/5/10 条历史行为冷启动实验。完整表格、方法约束及 SVG 图请阅读 [真实数据实验报告 / real-data results](docs/REAL_RESULTS.md)。These are independent experiments using an explicit split, **not** an assertion that the thesis's original numerical results have been exactly reproduced.

![Official MovieLens-100K benchmark](artifacts/real/ml100k/demo_mae.svg)

![Official MovieLens-100K cold-start benchmark](artifacts/real/cold/cold_start.svg)

"""
    marker = "## 安装与复现 · Installation & reproduction"
    if marker not in readme:
        raise ValueError("README section marker not found")
    if "## MovieLens-100K 真正实测" not in readme:
        readme = readme.replace(marker, section + marker)
    readme = readme.replace("- **MovieLens-100K**", "- **MovieLens-100K**")
    readme_path.write_text(readme, encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--warm", type=Path, required=True)
    p.add_argument("--cold", type=Path, required=True)
    p.add_argument("--doc", type=Path, default=Path("docs/REAL_RESULTS.md"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    args = p.parse_args(argv)
    build(args.warm, args.cold, args.doc, args.readme)
    print(f"Wrote {args.doc} and updated {args.readme}")


if __name__ == "__main__":
    main()
