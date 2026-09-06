"""
Aggregate cross-validation results into paper-ready tables
==========================================================

Reads every results/runs/*/result.json and produces:

    results/summary.csv         mean ± std, 95% CI and seed sensitivity per
                                (arch, condition, k) on the REAL test set
    results/paired_tests.csv    paired comparisons between conditions over the
                                same (fold, seed) replicates: mean difference,
                                Wilcoxon signed-rank p and paired t-test p
    results/runs_table.csv      one row per run (for plotting)

and prints Markdown versions of the first two.

Usage:
    python -m experiments.aggregate --results results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["dice", "iou", "precision", "recall", "pooled_dice"]


def load_runs(results_dir: str) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(results_dir).glob("runs/*/result.json")):
        with open(p) as f:
            r = json.load(f)
        row = {
            "run": r["run_name"], "arch": r["arch"], "condition": r["condition"],
            "k": int(r["k"]), "fold": r["fold"], "seed": r["seed"],
            "n_synthetic": r["n_train_synthetic"], "best_step": r["best_step"],
            "val_dice": r["val"]["dice"],
        }
        for m in METRICS:
            row[f"test_{m}"] = r["test"][m]
        rows.append(row)
    if not rows:
        raise SystemExit(f"No result.json files found under {results_dir}/runs")
    df = pd.DataFrame(rows)
    df["cell"] = df.apply(lambda x: x["condition"] if x["condition"] in ("real", "all") else f"{x['condition']}{x['k']}", axis=1)
    return df


def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (arch, cell), g in df.groupby(["arch", "cell"], sort=False):
        row = {"arch": arch, "cell": cell, "condition": g["condition"].iloc[0], "k": g["k"].iloc[0],
               "n_runs": len(g), "n_folds": g["fold"].nunique(), "n_seeds": g["seed"].nunique()}
        for m in METRICS:
            col = f"test_{m}"
            row[f"{m}_mean"] = g[col].mean()
            row[f"{m}_std"] = g[col].std(ddof=1) if len(g) > 1 else np.nan
            row[f"{m}_ci95"] = ci95(g[col].values)
        # Seed sensitivity: std across seeds within a fold, averaged over folds.
        per_fold = g.groupby("fold")["test_dice"].std(ddof=1)
        row["dice_seed_std"] = per_fold.mean()
        row["val_dice_mean"] = g["val_dice"].mean()
        out.append(row)
    order = {"real": 0, "all": 1, "random": 2, "filtered": 3, "antifiltered": 4}
    return pd.DataFrame(out).sort_values(["arch", "condition", "k"], key=lambda s: s.map(order) if s.name == "condition" else s).reset_index(drop=True)


def paired_tests(df: pd.DataFrame, metric: str = "test_dice") -> pd.DataFrame:
    """Paired comparisons over identical (fold, seed) replicates."""
    comparisons = [("filtered", "random"), ("filtered", "all"), ("filtered", "real"),
                   ("random", "real"), ("all", "real"), ("antifiltered", "random"), ("antifiltered", "real"), ("all", "random")]
    rows = []
    for arch, ga in df.groupby("arch"):
        ks = sorted(ga.loc[~ga["condition"].isin(["real", "all"]), "k"].unique()) or [0]
        for k in ks:
            def pick(cond):
                sel = ga[ga["condition"] == cond]
                if cond not in ("real", "all"):
                    sel = sel[sel["k"] == k]
                return sel.set_index(["fold", "seed"])[metric]
            for a, b in comparisons:
                xa, xb = pick(a), pick(b)
                common = xa.index.intersection(xb.index)
                if len(common) < 2:
                    continue
                d = (xa.loc[common] - xb.loc[common]).values
                try:
                    w_p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                except ValueError:
                    w_p = np.nan
                t_p = stats.ttest_rel(xa.loc[common], xb.loc[common]).pvalue
                rows.append({
                    "arch": arch, "k": k, "a": a, "b": b, "n_pairs": len(common),
                    "mean_diff": d.mean(), "diff_ci95": ci95(d), "wins_a": int((d > 0).sum()),
                    "wilcoxon_p": w_p, "paired_t_p": t_p,
                })
    return pd.DataFrame(rows)


def to_markdown(summary: pd.DataFrame) -> str:
    lines = ["| arch | condition | n | test Dice | test IoU | seed std (Dice) | val Dice |",
             "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['arch']} | {r['cell']} | {r['n_runs']} | "
            f"{r['dice_mean']:.4f} ± {r['dice_std']:.4f} (±{r['dice_ci95']:.4f}) | "
            f"{r['iou_mean']:.4f} ± {r['iou_std']:.4f} | {r['dice_seed_std']:.4f} | {r['val_dice_mean']:.4f} |"
        )
    return "\n".join(lines)


def tests_to_markdown(tests: pd.DataFrame) -> str:
    if tests.empty:
        return "(no paired comparisons available yet)"
    lines = ["| arch | k | comparison | pairs | Δ Dice (a−b) | a wins | Wilcoxon p | paired t p |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in tests.iterrows():
        lines.append(f"| {r['arch']} | {r['k']} | {r['a']} vs {r['b']} | {r['n_pairs']} | "
                     f"{r['mean_diff']:+.4f} (±{r['diff_ci95']:.4f}) | {r['wins_a']}/{r['n_pairs']} | "
                     f"{r['wilcoxon_p']:.3g} | {r['paired_t_p']:.3g} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    df = load_runs(args.results)
    summary = summarize(df)
    tests = paired_tests(df)

    out = Path(args.results)
    df.to_csv(out / "runs_table.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    tests.to_csv(out / "paired_tests.csv", index=False)

    print(f"Loaded {len(df)} runs from {out / 'runs'}\n")
    print("## Real test-set results (mean ± std across fold×seed replicates; ± 95% CI in parentheses)\n")
    print(to_markdown(summary))
    print("\n## Paired comparisons on test Dice\n")
    print(tests_to_markdown(tests))
    print(f"\nWritten: {out/'summary.csv'}, {out/'paired_tests.csv'}, {out/'runs_table.csv'}")


if __name__ == "__main__":
    main()
