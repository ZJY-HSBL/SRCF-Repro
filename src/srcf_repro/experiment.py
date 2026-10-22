"""Small fully reproducible benchmark. Real MovieLens files are never bundled."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from statistics import mean
from math import sqrt

from .data import load_movielens100k, popularity_seeds, split_by_user
from .model import Profile, Recommender


def synthetic(seed: int = 2026, users: int = 42, items: int = 60):
    """Generate explicitly synthetic ratings; NOT evidence for the thesis claims."""
    rng = random.Random(seed)
    profiles, ratings = {}, {}
    for user in range(1, users + 1):
        group = (user - 1) % 3
        profiles[user] = Profile(23+group*8 + rng.randrange(5),
                                 "M" if group == 0 else "F",
                                 ("student", "artist", "engineer")[group])
        watched = rng.sample(range(1, items + 1), 26)
        ratings[user] = {}
        for item in watched:
            affinity = 4.4 if (item - 1) % 3 == group else 2.1
            ratings[user][item] = float(max(1, min(5, round(affinity + rng.gauss(0, .7)))))
    return ratings, profiles


def evaluate(model: Recommender, test: dict[int, dict[int, float]], k: int,
             users: list[int], top_n: int = 5) -> dict:
    errors = [model.predict(u, i, k) - r for u in users for i, r in test.get(u, {}).items()
              if i in model.item_mean]
    predicted = {i for u in users for i, _ in model.recommend(u, k, top_n)}
    return {"mae": round(mean(abs(e) for e in errors), 6) if errors else None,
            "rmse": round(sqrt(mean(e*e for e in errors)), 6) if errors else None,
            "coverage": round(len(predicted)/len(model.items), 6),
            "n_test": len(errors), "n_users": len(users)}


def run(ratings, profiles, seed: int = 2026, max_users: int = 32,
        prediction: str = "normalized", k_values=(5, 10, 20)) -> list[dict]:
    train, test = split_by_user(ratings, seed=seed)
    seeds = popularity_seeds(train)
    users = sorted(u for u in test if test[u])[:max_users]
    results = []
    for mode in ("ub", "dr", "sr"):
        model = Recommender(train, profiles, seeds=seeds, mode=mode, prediction=prediction)
        for k in k_values:
            result = evaluate(model, test, k, users)
            results.append({"mode": mode, "k": k, "prediction": prediction,
                            "seed": seed, "seed_items": len(seeds), **result})
    return results


def save_results(rows: list[dict], output: Path, make_plot: bool = True,
                 label: str = "SYNTHETIC") -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "benchmark.json").write_text(json.dumps(rows, indent=2)+"\n", encoding="utf-8")
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        for mode in ("ub", "dr", "sr"):
            selected = [x for x in rows if x["mode"] == mode]
            ax.plot([x["k"] for x in selected], [x["mae"] for x in selected], "-o", label=mode.upper())
        ax.set(xlabel="Number of neighbors (k)", ylabel="MAE (lower is better)",
               title=f"Measured {label} benchmark (not thesis numeric replication)")
        ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / "demo_mae.svg")
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproducible SR-CF benchmark")
    parser.add_argument("--data-dir", type=Path, help="local MovieLens-100K ml-100k directory")
    parser.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-users", type=int, default=32)
    parser.add_argument("--prediction", choices=["paper", "normalized"], default="normalized")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--cold", action="store_true", help="fixed-test user-disjoint cold-start curve")
    args = parser.parse_args(argv)
    if args.max_users <= 0:
        parser.error("--max-users must be positive")
    ratings, profiles = load_movielens100k(args.data_dir) if args.data_dir else synthetic(args.seed)
    label = "MOVIELENS-100K" if args.data_dir else "SYNTHETIC"
    if args.cold:
        rows = cold_start_curve(ratings, profiles, seed=args.seed, max_users=args.max_users)
        save_cold_results(rows, args.out, make_plot=not args.no_plot, label=label)
        for row in rows:
            print(f"{label} cold={row['n_observed']:>2} {row['mode']} MAE={row['mae']} n={row['n_test']}")
        return
    rows = run(ratings, profiles, seed=args.seed, max_users=args.max_users, prediction=args.prediction)
    save_results(rows, args.out, make_plot=not args.no_plot, label=label)
    print(f"source={label}; users={len(ratings)}; interactions={sum(map(len, ratings.values()))}; seed={args.seed}")
    for row in rows:
        print(f"{row['mode']:>2} k={row['k']:>2} MAE={row['mae']} RMSE={row['rmse']} "
              f"coverage={row['coverage']} n={row['n_test']}")
    print(f"Wrote {args.out}/benchmark.csv and benchmark.json")


def cold_start_curve(ratings, profiles, seed: int = 2026, max_users: int = 15,
                     observed=(0, 1, 3, 5, 10), k: int = 10) -> list[dict]:
    """User-disjoint reference users; fixed target test items across history levels.

    Never selects observation indices from the test portion. Unlike the warm
    benchmark, reference training contains no target-user ratings.
    """
    rng = random.Random(seed)
    users = [u for u in sorted(ratings) if len(ratings[u]) >= 18][:max_users]
    if not users:
        raise ValueError("no users have enough interactions for cold-start evaluation")
    target = set(users)
    reference = {u: dict(items) for u, items in ratings.items() if u not in target}
    seeds = popularity_seeds(reference)
    target_splits = {}
    for u in users:
        shuffled = sorted(ratings[u])
        rng.shuffle(shuffled)
        holdout = set(shuffled[-5:])
        seen = [i for i in shuffled if i not in holdout]
        target_splits[u] = (seen, {i: ratings[u][i] for i in holdout})
    output = []
    for n in observed:
        training = {u: dict(v) for u, v in reference.items()}
        for u, (seen, _) in target_splits.items():
            training[u] = {i: ratings[u][i] for i in seen[:n]}
        holdouts = {u: p[1] for u, p in target_splits.items()}
        for mode in ("ub", "dr", "sr"):
            m = Recommender(training, profiles, seeds, mode, prediction="normalized")
            row = evaluate(m, holdouts, k, users)
            output.append({"mode": mode, "n_observed": n, "k": k, "seed": seed, **row})
    return output


def save_cold_results(rows: list[dict], output: Path, make_plot: bool = True,
                      label: str = "SYNTHETIC") -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cold_start.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        for mode in ("ub", "dr", "sr"):
            group = [r for r in rows if r["mode"] == mode]
            ax.plot([r["n_observed"] for r in group], [r["mae"] for r in group], "-o", label=mode.upper())
        ax.set(title=f"{label}: cold-start simulation, fixed target holdouts",
               xlabel="Observed target-user ratings", ylabel="MAE (lower is better)")
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(output / "cold_start.svg"); plt.close(fig)


if __name__ == "__main__":
    main()
