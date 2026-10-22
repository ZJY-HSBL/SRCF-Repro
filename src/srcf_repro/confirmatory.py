"""Validation-only support-shrinkage selection, followed by untouched-user evaluation.

The dataset is MovieLens-100K or explicitly synthetic smoke data. Independent
user cohorts within one dataset are NOT equivalent to an external dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .confidence import ConfidenceRecommender
from .data import load_movielens100k, popularity_seeds
from .experiment import synthetic
from .model import Recommender
from .studies import build_cases

OBSERVATIONS = (0, 1, 3, 5, 10)
SELECTION_BUDGETS = (3, 5, 10)
CANDIDATE_SHRINKAGE = (2.0, 10.0, 30.0)
METHODS = ('ub', 'dr', 'sr', 'confidence_2', 'confidence_10', 'confidence_30', 'confidence_10_no_seed')
COHORT_SEED = 314159


def cohorts(ratings: dict, *, val_n: int = 32, test_n: int = 48,
            prior_batches: int = 4, prior_per_batch: int = 24,
            prior_seed_batches: int = 2, prior_seed_batch_size: int = 24,
            seed: int = COHORT_SEED, holdout: int = 5):
    """Exclude ALL earlier sampled targets before drawing validation/test users.

    Excluded cohorts are not available as reference neighbors. Prior cohort
    reproduction uses the exact RNG/protocol from studies and seed_diagnostics.
    """
    if val_n < 1 or test_n < 1:
        raise ValueError('validation and test must contain users')
    _, prior = build_cases(ratings, study_seed=2026,
                           batches=prior_batches, users_per_batch=prior_per_batch)
    prior_users = {u for batch in prior for _, u, _, _ in batch}
    # v0.4 used a fresh 271828-seeded sample from the *remaining* users.
    remaining = {u: rs for u, rs in ratings.items() if u not in prior_users}
    _, prior_seed = build_cases(remaining, study_seed=271828,
                                batches=prior_seed_batches,
                                users_per_batch=prior_seed_batch_size)
    prior_users.update(u for batch in prior_seed for _, u, _, _ in batch)
    candidates = sorted(u for u, rs in ratings.items()
                        if u not in prior_users and len(rs) >= 10 + holdout)
    if len(candidates) <= val_n + test_n:
        raise ValueError('not enough unused users for validation/test/reference')
    rng = random.Random(seed)
    sampled = rng.sample(candidates, val_n + test_n)
    val_ids, test_ids = set(sampled[:val_n]), set(sampled[val_n:])
    excluded = prior_users | val_ids | test_ids
    reference = {u: dict(r) for u, r in ratings.items() if u not in excluded}

    def make_cases(ids):
        cases = []
        for u in sorted(ids):
            ids_r = sorted(ratings[u]); rng.shuffle(ids_r)
            held = {i: ratings[u][i] for i in ids_r[-holdout:]}
            remaining_ids = ids_r[:-holdout]
            cases.append((u, held, remaining_ids))
        return cases

    assert not ((val_ids & test_ids) or (set(reference) & excluded))
    return reference, make_cases(val_ids), make_cases(test_ids), len(prior_users)


def _model(method: str, training: dict, profiles: dict, seeds: frozenset):
    if method in ('ub', 'dr', 'sr'):
        return Recommender(training, profiles, seeds=seeds,
                           mode=method, prediction='normalized')
    if method in METHODS:
        suffix = method.removeprefix('confidence_')
        no_seed = suffix.endswith('_no_seed')
        level = float(suffix.removesuffix('_no_seed'))
        return ConfidenceRecommender(training, profiles, seeds=seeds,
                                     mode='sr', prediction='normalized',
                                     lambda_support=level, use_seed=not no_seed)
    raise ValueError(f'unknown model: {method}')


def evaluate_cohort(reference: dict, cases: list, ratings: dict,
                    profiles: dict, methods=METHODS, *, k: int = 10,
                    seeds_count: int = 40, partition: str = 'validation'):
    """Reference + exactly *one* target user per fit: no target-to-target edges."""
    seeds = popularity_seeds(reference, count=seeds_count)
    rows = []
    for u, holdouts, candidate_ratings in cases:
        for observed in OBSERVATIONS:
            training = dict(reference)
            training[u] = {i: ratings[u][i] for i in candidate_ratings[:observed]}
            test_items = [(i, r) for i, r in holdouts.items()
                          if any(i in rs for rs in reference.values())]
            if not test_items:
                raise ValueError('test user has no reference-observed heldout items')
            for method in methods:
                model = _model(method, training, profiles, seeds)
                errors = [abs(model.predict(u, item, k) - r) for item, r in test_items]
                rows.append({'cohort': partition, 'case': f'{partition}-{u:04d}',
                             'observed': observed, 'method': method,
                             'mae': round(mean(errors), 9),
                             'n_test': len(errors), 'k': k})
    return rows


def select_on_validation(rows: list[dict]) -> dict:
    """Single prespecified selection metric: mean user MAE across 3/5/10."""
    scores = {}
    for lam in CANDIDATE_SHRINKAGE:
        method = f'confidence_{int(lam)}'
        vals = [r['mae'] for r in rows if r['method'] == method
                and r['observed'] in SELECTION_BUDGETS]
        expected = sum(1 for r in rows if r['method'] == method) * 3 // 5
        if len(vals) != expected or not vals:
            raise ValueError('missing validation score for candidate')
        scores[method] = mean(vals)
    chosen = min(scores, key=lambda name: (scores[name], name))
    return {'selected': chosen, 'validation_mean_mae': {k: round(v, 8) for k,v in scores.items()},
            'criterion': 'equal-weight average of per-user MAE over budgets 3,5,10'}


def paired_interval(rows, a: str, b: str, *, budgets=SELECTION_BUDGETS,
                    draws: int = 3000, seed: int = 314160):
    """User-cluster bootstrap, preserving dependence across interaction budgets."""
    data = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row['observed'] in budgets and row['method'] in (a, b):
            data[row['case']][row['observed']][row['method']] = row['mae']
    deltas = []
    for case in sorted(data):
        if set(data[case]) != set(budgets) or any(set(pair) != {a,b} for pair in data[case].values()):
            raise ValueError('unpaired user/budget data')
        deltas.append(mean(data[case][n][a] - data[case][n][b] for n in budgets))
    if not deltas or draws < 100:
        raise ValueError('missing paired cases or too few draws')
    rng = random.Random(seed)
    boot = sorted(mean(rng.choices(deltas, k=len(deltas))) for _ in range(draws))
    return {'comparison': a + ' - ' + b, 'n_users': len(deltas),
            'budgets': ','.join(map(str, budgets)),
            'difference_mae': round(mean(deltas), 6),
            'ci_low': round(boot[int(.025*(draws-1))], 6),
            'ci_high': round(boot[int(.975*(draws-1))], 6),
            'wins_a': sum(d < 0 for d in deltas), 'ties': sum(d == 0 for d in deltas)}


def _csv(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def publish(validation, test, selection, out: Path, *, source: str,
            prior_excluded: int, reference_users: int, seed: int):
    out.mkdir(parents=True, exist_ok=True)
    _csv(out/'validation_per_user.csv',validation)
    _csv(out/'test_per_user.csv',test)
    methods = ('ub','dr','sr','confidence_2','confidence_10','confidence_30','confidence_10_no_seed')
    summary = []
    for cohort, rows in (('validation',validation),('test',test)):
        for obs in OBSERVATIONS:
            for method in methods:
                v=[r['mae'] for r in rows if r['observed']==obs and r['method']==method]
                if v:
                    summary.append({'cohort':cohort,'observed':obs,'method':method,
                                    'mean_user_mae':round(mean(v),6),'n_users':len(v)})
    _csv(out/'summary.csv', summary)
    selected=selection['selected']
    comparisons=[paired_interval(test, selected, 'sr'),
                 paired_interval(test, selected, 'dr'),
                 paired_interval(test, selected, 'ub'),
                 paired_interval(test, 'confidence_10', 'confidence_10_no_seed')]
    _csv(out/'paired_intervals.csv', comparisons)
    protocol={'source':source, 'seed':seed, 'prior_users_excluded':prior_excluded,
              'reference_users':reference_users,'n_val_users':len({r['case'] for r in validation}),
              'n_test_users':len({r['case'] for r in test}),
              'budgets':OBSERVATIONS,'selection_budgets':SELECTION_BUDGETS,
              'lambda_candidates':CANDIDATE_SHRINKAGE,'selected':selected,
              'selection':selection, 'seed_count':40, 'k':10,
              'no_reference_target_overlap':True,'no_validation_test_overlap':True,
              'test_access_after_freezing_selection':True,
              'user_cluster_bootstrap_draws':3000,
              'statistical_status':'exploratory holdout within one dataset; no multiple comparisons correction'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.8,4.8))
    for method in ('ub','dr','sr',selected):
        s=[x for x in summary if x['cohort']=='test' and x['method']==method]
        ax.plot([x['observed'] for x in s],[x['mean_user_mae'] for x in s],'-o',label=method)
    ax.set(xlabel='Observed ratings per new user',ylabel='MAE (lower is better)',
           title='Untouched target users: validation-selected confidence method')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(out/'holdout_mae.svg');plt.close(fig)
    def markdown_table(cohort):
        ss={(r['observed'],r['method']):r['mean_user_mae'] for r in summary if r['cohort']==cohort}
        names=['ub','dr','sr',selected]
        t='| Observed | '+' | '.join(names)+' |\n|---:|'+'---:|'*len(names)+'\n'
        for b in OBSERVATIONS:
            t+='| '+str(b)+' | '+' | '.join(f'{ss[b,m]:.4f}' for m in names)+' |\n'
        return t
    comparison_table='| Test comparison | ΔMAE | Exploratory 95% CI | Paired users |\n|---|---:|---:|---:|\n'
    for c in comparisons:
        comparison_table+=f"| {c['comparison']} | {c['difference_mae']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['n_users']} |\n"
    report=f'''# Validation-frozen confidence study / 可信度收缩验证实验

**{source}; measured experiment, not the source thesis's numerical replication.**

## Precommitted question / 科研假设

Does support-dependent shrinkage of the seed-rating and historical-rating
similarity terms improve prediction under 3/5/10 observed ratings? The
method is an exploratory modification proposed in this repo, **not** SR-CF
as defined by Qiao Yu (2018).

Users from earlier exploratory studies (n={prior_excluded}) are **excluded**
from reference, validation, and test. New users are assigned once with RNG
seed {seed}; reference {reference_users}, validation {protocol['n_val_users']}
and untouched test {protocol['n_test_users']}; the same five holdout ratings
per target remain hidden at each interaction budget. Each model's neighbors
are exclusively *reference users*: new validation/test users cannot act as
one another's neighbors.

## Validation-only hyperparameter selection

Candidate support shrinks: `λ=2`, `λ=10`, `λ=30`; seed item count **40**,
neighbor count **10**, SR-CF original weights `(0.2,0.3,0.5)`. Candidate
selection uses only equal-weight per-user validation MAE averaged over
budgets `{SELECTION_BUDGETS}`; tie-break lexicographic.

{json.dumps(selection,ensure_ascii=False,indent=2)}

## Validation scores (diagnostic only)

{markdown_table('validation')}

## Untouched-user test results / 独立目标用户测试

{markdown_table('test')}

![Test MAE across observation budgets](../{out.as_posix()}/holdout_mae.svg)

## Paired test-user comparisons

Differences = A − B; negative favors A. 3,000-resample user-cluster percentile
bootstrap; all three observation budgets per user resampled together.
Intervals are **exploratory**, unadjusted for multiple comparisons, and
conditional on MovieLens-100K; this is not cross-dataset confirmation.

{comparison_table}

## Limitations / 局限性

- Earlier cohorts are excluded, but reference data and new-user holdouts come from
  the same MovieLens-100K dataset; this does not demonstrate external generalization.
- This is a user-level fixed-k rating-error experiment; it does not directly
  establish ranking quality (NDCG@K), coverage or computational scalability.
- The old SR-CF Pearson term includes seed-rated items again as history; seed
  and history are correlated and not independent evidence.
- The current confidence rule also rescales effective fusion weights; ablate
  confidence shrinkage versus renormalization in future experiments.
- If CIs include zero, report no confirmed gain. Negative results remain visible.

Raw audit / 原始数据：`validation_per_user.csv`, `test_per_user.csv`,
`summary.csv`, `paired_intervals.csv`, `protocol.json`.
'''
    Path('docs/CONFIDENCE_STUDY.md').write_text(report,encoding='utf-8')
    return comparisons,summary


def main(argv=None):
    p=argparse.ArgumentParser(description='Validation-only confidence-aware SR-CF study')
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--out',type=Path,default=Path('artifacts/local/confidence'))
    p.add_argument('--validation-users',type=int,default=32)
    p.add_argument('--test-users',type=int,default=48)
    p.add_argument('--prior-batches',type=int,default=4)
    p.add_argument('--prior-per-batch',type=int,default=24)
    p.add_argument('--prior-seed-batches',type=int,default=2)
    p.add_argument('--prior-seed-per-batch',type=int,default=24)
    p.add_argument('--seed',type=int,default=COHORT_SEED)
    args=p.parse_args(argv)
    ratings,profiles=(load_movielens100k(args.data_dir) if args.data_dir else
                      synthetic(args.seed,users=64,items=55))
    reference,val,test,previous=cohorts(ratings,val_n=args.validation_users,
                                       test_n=args.test_users,
                                       prior_batches=args.prior_batches,
                                       prior_per_batch=args.prior_per_batch,
                                       prior_seed_batches=args.prior_seed_batches,
                                       prior_seed_batch_size=args.prior_seed_per_batch,
                                       seed=args.seed)
    validation=evaluate_cohort(reference,val,ratings,profiles,partition='validation')
    selection=select_on_validation(validation)
    # Untouched test: only now are its predictions computed.
    holdout=evaluate_cohort(reference,test,ratings,profiles,partition='test')
    comparisons,summary=publish(validation,holdout,selection,args.out,
                                source='MOVIELENS-100K' if args.data_dir else 'SYNTHETIC',
                                prior_excluded=previous,reference_users=len(reference),seed=args.seed)
    print(json.dumps({'prior_excluded':previous,'reference':len(reference),
                      'val_users':len(val),'test_users':len(test),
                      'selected':selection,'paired':comparisons,'out':str(args.out)}))

if __name__=='__main__':
    main()
