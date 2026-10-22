"""Predeclared seed-support and seed-set-size sensitivity experiment.

This is an *independent exploratory cohort* from the earlier user evaluation:
target users appearing in the earlier study are excluded entirely. Dataset
population is shared, so this is not an external-dataset validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from .data import load_movielens100k, popularity_seeds
from .experiment import synthetic
from .model import Recommender, adjusted_cosine
from .studies import OBSERVATIONS, build_cases, paired_ci

STUDY_SEED = 271828
PREVIOUS_SEED = 2026
SEED_COUNTS = (12, 40, 100)
METHODS = ('ub', 'dr', 'sr12', 'sr40', 'sr100')


def prepare_cases(ratings: dict, *, old_b=4, old_n=24,
                  batches=2, users_per_batch=24, study_seed=STUDY_SEED):
    """Exclude all users used by the previous 96-user exploratory study."""
    _, old_cases = build_cases(ratings, study_seed=PREVIOUS_SEED,
                               batches=old_b, users_per_batch=old_n)
    prior_ids = {u for group in old_cases for _, u, _, _ in group}
    pool = {u: items for u, items in ratings.items() if u not in prior_ids}
    reference, cases = build_cases(pool, study_seed=study_seed,
                                   batches=batches, users_per_batch=users_per_batch)
    new_ids = {u for group in cases for _, u, _, _ in group}
    assert not (new_ids & prior_ids)
    assert not (new_ids & set(reference))
    assert not (prior_ids & set(reference))
    return reference, cases, len(prior_ids)


def seed_support(observed: dict, reference: dict, seeds: frozenset[int]) -> dict:
    """Report support *before* ranking: minimum two common, nonconstant seed ratings."""
    on_seed = {i: r for i, r in observed.items() if i in seeds}
    common_pairs = 0
    useful_pairs = 0
    if len(on_seed) >= 2:
        for history in reference.values():
            intersect = on_seed.keys() & history.keys()
            if len(intersect) >= 2:
                common_pairs += 1
                if abs(adjusted_cosine(on_seed, history)) > 1e-12:
                    useful_pairs += 1
    return {"observed_seed_items": len(on_seed),
            "candidate_pairs_with_2_shared_seed_ratings": common_pairs,
            "candidate_pairs_with_nonzero_seed_similarity": useful_pairs}


def run(ratings: dict, profiles: dict, *, batches: int = 2,
        users_per_batch: int = 24, study_seed: int = STUDY_SEED,
        old_b: int = 4, old_n: int = 24, k: int = 10):
    reference, groups, n_excluded = prepare_cases(
        ratings, old_b=old_b, old_n=old_n,
        batches=batches, users_per_batch=users_per_batch, study_seed=study_seed)
    seeds = {size: popularity_seeds(reference, count=size) for size in SEED_COUNTS}
    all_scores, diagnostics = [], []
    for batch, group in enumerate(groups, 1):
        for obs in OBSERVATIONS:
            train = {u: dict(v) for u, v in reference.items()}
            for _, u, _, train_ids in group:
                train[u] = {i: ratings[u][i] for i in train_ids[:obs]}
            models = {}
            for name in METHODS:
                nseed = int(name[2:]) if name.startswith('sr') else 12
                models[name] = Recommender(train, profiles, seeds[nseed],
                                          mode='sr' if name.startswith('sr') else name,
                                          prediction='normalized')
            for cid, u, holdout, train_ids in group:
                uid = f'group{batch}_{cid}'
                observed_ratings = {i: ratings[u][i] for i in train_ids[:obs]}
                rec = {"case": uid, "batch": batch, "observed": obs,
                       "study_seed": study_seed}
                for count in SEED_COUNTS:
                    support = seed_support(observed_ratings, reference, seeds[count])
                    rec[f'seed{count}_seen'] = support['observed_seed_items']
                    rec[f'seed{count}_2shared'] = support['candidate_pairs_with_2_shared_seed_ratings']
                    rec[f'seed{count}_usable'] = support['candidate_pairs_with_nonzero_seed_similarity']
                diagnostics.append(rec)
                for name, model in models.items():
                    errors = [abs(model.predict(u, item, k) - r)
                              for item, r in holdout.items() if item in model.item_mean]
                    if not errors:
                        raise ValueError('no scored holdouts; cannot compare model')
                    all_scores.append({"case": uid, "batch": batch, "observed": obs,
                                       "method": name, "mae": round(mean(errors), 9),
                                       "n_test": len(errors), "k": k,
                                       "study_seed": study_seed})
    return all_scores, diagnostics, n_excluded


def aggregate(rows: list[dict], diagnostics: list[dict]):
    scores = []
    support = []
    for budget in OBSERVATIONS:
        for model in METHODS:
            vals = [r['mae'] for r in rows if r['observed'] == budget and r['method'] == model]
            scores.append({"observed": budget, "method": model,
                           "user_mean_mae": round(mean(vals), 6), "n_users": len(vals)})
        group = [r for r in diagnostics if r['observed'] == budget]
        for count in SEED_COUNTS:
            support.append({"observed": budget, "seeds": count,
                            "n_users": len(group),
                            "fraction_users_with_2_or_more_seed_ratings": round(
                                mean(r[f'seed{count}_seen'] >= 2 for r in group), 6),
                            "fraction_users_with_usable_seed_similarity": round(
                                mean(r[f'seed{count}_usable'] > 0 for r in group), 6),
                            "mean_nonzero_seed_neighbors": round(
                                mean(r[f'seed{count}_usable'] for r in group), 4)})
    cis = [paired_ci(rows, a, b, obs)
           for obs in OBSERVATIONS for a, b in
           [('sr12','dr'), ('sr40','sr12'), ('sr100','sr12')]]
    return scores, support, cis


def _write_csv(path: Path, rows: list[dict]):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary: list[dict], support: list[dict], out: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    for name in METHODS:
        s = [r for r in summary if r['method'] == name]
        ax.plot([x['observed'] for x in s], [x['user_mean_mae'] for x in s],
                '-o', label=name.upper())
    ax.set(xlabel='Visible user ratings', ylabel='User-average MAE (lower is better)',
           title='Measured seed-set size sensitivity (not thesis numeric replication)')
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out/'seed_count_mae.svg'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    for num in SEED_COUNTS:
        grp = [r for r in support if r['seeds'] == num]
        ax.plot([x['observed'] for x in grp],
                [x['fraction_users_with_usable_seed_similarity']*100 for x in grp],
                '-o', label=f'{num} seed movies')
    ax.set(xlabel='Visible user ratings', ylabel='Users with nonzero seed similarity (%)',
           title='Diagnostic: when can the seed component contribute?')
    ax.set_ylim(-3, 103); ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out/'seed_signal_availability.svg'); plt.close(fig)


def export(rows: list[dict], diagnostics: list[dict], excluded: int,
           out: Path, *, source: str, seed: int, batches: int, users_per_batch: int):
    out.mkdir(parents=True, exist_ok=True)
    summary, support, ci = aggregate(rows, diagnostics)
    for name, value in [('per_user.csv', rows), ('seed_diagnostics.csv', diagnostics),
                        ('summary.csv', summary), ('seed_support.csv', support),
                        ('paired_bootstrap.csv', ci)]:
        _write_csv(out/name, value)
    (out/'protocol.json').write_text(json.dumps({
        'data_source': source, 'seed': seed, 'cohort_target_count': batches*users_per_batch,
        'previous_cohort_excluded': excluded, 'prior_cohort_seed': PREVIOUS_SEED,
        'reference_and_new_target_disjoint': True, 'prior_cohort_fully_excluded': True,
        'old_cohort_shape': [4,24] if excluded == 96 else 'synthetic override',
        'observed_budgets': OBSERVATIONS, 'seed_counts': SEED_COUNTS,
        'methods': METHODS, 'k': rows[0]['k'], 'ratings_held_out_per_user': 5,
        'weights_sr': [0.2, 0.3, 0.5], 'predictor': 'normalized',
        'bootstrap_draws': 2000, 'bootstrap_seed': 9301,
        'exploratory_no_validation_tuning': True}, indent=2)+'\n', encoding='utf-8')
    _plot(summary, support, out)
    by = {(r['observed'],r['method']):r for r in summary}
    tab = '| Known ratings | UB-CF | DR-CF | SR(12) | SR(40) | SR(100) |\n|---:|---:|---:|---:|---:|---:|\n'
    for obs in OBSERVATIONS:
        tab += '| ' + str(obs) + ' | ' + ' | '.join(f"{by[(obs,m)]['user_mean_mae']:.4f}" for m in METHODS) + ' |\n'
    sup_tab = '| Known ratings | Seed movies | Targets with usable seed similarities | Mean usable reference neighbors |\n|---:|---:|---:|---:|\n'
    for v in support:
        sup_tab += f"| {v['observed']} | {v['seeds']} | {v['fraction_users_with_usable_seed_similarity']*100:.1f}% | {v['mean_nonzero_seed_neighbors']:.1f} |\n"
    ci_tab = '| Budget | Difference (A − B) | Mean ΔMAE | Exploratory 95% bootstrap CI |\n|---:|---|---:|---:|\n'
    for c in ci:
        if c['observed'] not in (3, 5, 10):
            continue
        ci_tab += f"| {c['observed']} | {c['a']} − {c['b']} | {c['mean_diff']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] |\n"
    pathprefix = '../' + out.as_posix()
    report = f'''# Seed-signal availability and sensitivity / 种子评分信号可用性研究

**Data source / 数据源：{source}. Measured, exploratory, and not an exact thesis replication.**

本实验专门检验固定 12 部种子电影是否会因为新用户交互太少而导致种子评分相似度无法计算。
The test asks whether a small set of popular seed movies can provide enough *co-rated seed items* for the SR-CF similarity component to be meaningful in cold-start settings.

## Frozen protocol / 预设实验方案

- Excluded all **{excluded}** target users from the earlier study, both as reference neighbors and as candidates; selected **{batches*users_per_batch} previously unevaluated target users** with seed {seed}.
- All current targets are excluded from the persistent reference pool. Within each batch, targets with revealed ratings may serve as neighbors of other batch targets, but **no target's held-out ratings** are ever exposed to model training. This is *not* cross-dataset validation.
- Same five held-out ratings for each user at observation budgets **0, 1, 3, 5, 10**; 10 nearest neighbors; **fixed** SR-CF mixing weights (0.2,0.3,0.5). We only vary seed-set size **12/40/100**, and compare against UB-CF and DR-CF.
- Seeds selected from reference ratings only, by popularity; no tuning on target scores. Evaluation is averaged *per target user*. Predictions for held-out items with no training vocabulary entry are excluded from these MAE estimates.

## Model MAE / 模型比较 (lower is better)

{tab}
![Measured MAE versus seed-set size]({pathprefix}/seed_count_mae.svg)

## Seed support / 种子信息可用率

A target can contribute a **nonzero** seed-rating Pearson similarity to at least one reference neighbor only when the seed overlap spans at least two shared items with nonconstant co-ratings. The table reports the percentage of targets with at least one such candidate reference user; it does **not** imply that this candidate is ranked in top-k.

{sup_tab}
![Measured seed similarity availability]({pathprefix}/seed_signal_availability.svg)

## Paired error differences / 同一目标用户配对误差

(A − B) below zero favors A. User-level 2,000-draw bootstrap, **exploratory unadjusted** confidence intervals; multiple hypotheses and budgets were inspected.

{ci_tab}
## Research implications / 研究启示

1. A near-zero seed similarity term may be uninformative even when assigned a nominal mixing weight of 0.3. Increase in seed size changes information availability **and** the statistical comparison target, so do not treat size effects as evidence of a new method without external validation.
2. Model rankings depend on neighborhood selection, data sparsity and seed choice. Analyze the tradeoff between *shared seed coverage* and candidate similarity reliability before fitting any adaptive weight rule.
3. For a confirmatory follow-up, freeze a single seed-selection policy on a **validation-only** cohort and evaluate on users and/or data not used to select it. Clearly report effect sizes and confidence intervals.

Raw evidence / 全部原始数据：[`per_user.csv`]({pathprefix}/per_user.csv), [`seed_diagnostics.csv`]({pathprefix}/seed_diagnostics.csv), [`seed_support.csv`]({pathprefix}/seed_support.csv), [`paired_bootstrap.csv`]({pathprefix}/paired_bootstrap.csv), [`protocol.json`]({pathprefix}/protocol.json).
'''
    Path('docs/SEED_DIAGNOSTICS.md').write_text(report, encoding='utf-8')
    readme = Path('README.md')
    txt = readme.read_text(encoding='utf-8')
    begin, end = '<!-- SEED_STUDY_START -->', '<!-- SEED_STUDY_END -->'
    if begin not in txt or end not in txt:
        raise ValueError('seed study markers missing')
    snippet = (f'### MovieLens-100K 新目标用户种子集实验 · Seed-set follow-up\n\n'
               f'**{source}**, {batches*users_per_batch} new target users (earlier {excluded} excluded), '
               'fixed user holdouts and seed-set-size ablations.\n\n'
               +tab+'\n'
               +f'![Seed-set sensitivity, measured]({out.as_posix()}/seed_count_mae.svg)\n\n'
               +f'![Seed similarity support, measured]({out.as_posix()}/seed_signal_availability.svg)\n\n'
               +'[实验方案、配对区间与完整 CSV / protocol, CIs and data](docs/SEED_DIAGNOSTICS.md)\n')
    readme.write_text(txt.split(begin)[0]+begin+'\n'+snippet+'\n'+end+txt.split(end,1)[1], encoding='utf-8')


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description='Seed similarity diagnostic on a fresh target cohort')
    ap.add_argument('--data-dir', type=Path)
    ap.add_argument('--out', type=Path, default=Path('artifacts/local/seed-diagnostics'))
    ap.add_argument('--seed', type=int, default=STUDY_SEED)
    ap.add_argument('--batches', type=int, default=2)
    ap.add_argument('--users-per-batch', type=int, default=24)
    ap.add_argument('--old-b', type=int, default=4,
                    help='number of previously studied target batches')
    ap.add_argument('--old-n', type=int, default=24)
    args = ap.parse_args(argv)
    ratings, profiles = (load_movielens100k(args.data_dir) if args.data_dir else synthetic(args.seed, users=80))
    rows, diagnostics, excluded = run(ratings, profiles,
                                      batches=args.batches,
                                      users_per_batch=args.users_per_batch,
                                      old_b=args.old_b, old_n=args.old_n,
                                      study_seed=args.seed)
    export(rows, diagnostics, excluded, args.out,
           source='MOVIELENS-100K' if args.data_dir else 'SYNTHETIC',
           seed=args.seed, batches=args.batches,
           users_per_batch=args.users_per_batch)
    print(json.dumps({'rows':len(rows), 'diagnostics':len(diagnostics),
                      'targets':args.batches*args.users_per_batch,
                      'excluded_previous':excluded, 'out':str(args.out)}))


if __name__ == '__main__':
    main()
