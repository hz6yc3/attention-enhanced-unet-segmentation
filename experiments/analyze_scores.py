"""
Diagnostics for the synthetic-image disagreement scores
=======================================================

Checks whether the disagreement ranking is confounded with road density and
shows what the filtered / antifiltered subsets look like.

Usage:
    python -m experiments.analyze_scores --results results --k 250
"""

import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--k", type=int, default=250)
    parser.add_argument("--score-key", default="seed_disagreement")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, "scores", "*.csv")))
    if not files:
        raise SystemExit("no score files found")
    for f in files:
        df = pd.read_csv(f)
        low = df.nsmallest(args.k, args.score_key)
        high = df.nlargest(args.k, args.score_key)
        print(f"\n=== {os.path.basename(f)} ({len(df)} images) ===")
        print(f"Spearman corr of {args.score_key} with road_fraction: "
              f"{df[args.score_key].corr(df['road_fraction'], method='spearman'):+.3f}")
        print(f"Spearman corr of {args.score_key} with label_disagreement: "
              f"{df[args.score_key].corr(df['label_disagreement'], method='spearman'):+.3f}")
        print(f"{'subset':14s} {'score':>8s} {'road_frac':>10s} {'label_dis':>10s} {'prob_var':>9s}")
        for name, sub in [("filtered", low), ("antifiltered", high), ("all", df)]:
            print(f"{name:14s} {sub[args.score_key].mean():8.4f} {sub['road_fraction'].mean():10.4f} "
                  f"{sub['label_disagreement'].mean():10.4f} {sub['prob_variance'].mean():9.5f}")

    # Stability of the ranking across folds: overlap of the low-k sets
    if len(files) > 1:
        sets = [set(pd.read_csv(f).nsmallest(args.k, args.score_key)["name"]) for f in files]
        pairs = [(len(a & b) / args.k) for i, a in enumerate(sets) for b in sets[i + 1:]]
        print(f"\nMean overlap of the {args.k} lowest-score images between folds: {sum(pairs)/len(pairs):.2f}")


if __name__ == "__main__":
    main()
