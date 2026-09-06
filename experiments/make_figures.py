"""
Paper figures from the cross-validation results
===============================================

Produces, under <results>/figures/:

    fig_conditions.png   test Dice of every replicate per condition, with paired
                         lines joining the same (fold, seed) across conditions
    fig_paired.png       paired differences in test Dice relative to the random
                         subset, one point per replicate, with 95% CI
    fig_scores.png       committee disagreement vs. label disagreement for all
                         synthetic images (fold 0), coloured by road fraction,
                         with the filtered / antifiltered subsets marked

Usage:
    python -m experiments.make_figures --results results --k 250
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["real", "filtered", "random", "all", "antifiltered"]
LABELS = {"real": "Real only", "filtered": "Most-agreed\n(filtered)", "random": "Random",
          "all": "All synthetic", "antifiltered": "Most-disputed\n(anti-filtered)"}
COLORS = {"real": "#6c757d", "filtered": "#1f77b4", "random": "#2ca02c", "all": "#9467bd", "antifiltered": "#d62728"}


def load(results):
    rows = []
    for p in sorted(glob.glob(os.path.join(results, "runs", "*", "result.json"))):
        r = json.load(open(p))
        rows.append({"arch": r["arch"], "condition": r["condition"], "k": r["k"], "fold": r["fold"],
                     "seed": r["seed"], "dice": r["test"]["dice"], "iou": r["test"]["iou"]})
    return pd.DataFrame(rows)


def fig_conditions(df, out):
    conds = [c for c in ORDER if c in set(df["condition"])]
    piv = df.pivot_table(index=["fold", "seed"], columns="condition", values="dice")
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = {c: i for i, c in enumerate(conds)}
    for _, row in piv.iterrows():
        xs = [x[c] for c in conds if not np.isnan(row.get(c, np.nan))]
        ys = [row[c] for c in conds if not np.isnan(row.get(c, np.nan))]
        ax.plot(xs, ys, color="0.75", lw=0.6, zorder=1)
    for c in conds:
        vals = df.loc[df["condition"] == c, "dice"].values
        jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.18
        ax.scatter(np.full(len(vals), x[c]) + jitter, vals, s=18, color=COLORS[c], zorder=3, edgecolor="white", lw=0.4)
        m, s = vals.mean(), vals.std(ddof=1)
        ax.errorbar(x[c], m, yerr=s, fmt="_", color="black", ms=22, mew=1.8, capsize=5, zorder=4)
        ax.text(x[c], vals.max() + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([LABELS[c] for c in conds], fontsize=9)
    ax.set_ylabel("Test Dice (20 held-out real images)")
    ax.set_ylim(df["dice"].min() - 0.015, df["dice"].max() + 0.02)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig_paired(df, out, ref="random"):
    piv = df.pivot_table(index=["fold", "seed"], columns="condition", values="dice")
    conds = [c for c in ORDER if c in piv.columns and c != ref]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for i, c in enumerate(conds):
        d = (piv[c] - piv[ref]).dropna().values
        ci = 2.145 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0  # t(14) at 97.5%
        jitter = (np.random.RandomState(1).rand(len(d)) - 0.5) * 0.2
        ax.scatter(d, np.full(len(d), i) + jitter, s=16, color=COLORS[c], edgecolor="white", lw=0.4, zorder=3)
        ax.errorbar(d.mean(), i, xerr=ci, fmt="D", color="black", ms=5, capsize=4, zorder=4)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([LABELS[c].replace("\n", " ") for c in conds], fontsize=9)
    ax.set_xlabel(f"Test Dice difference vs. {LABELS[ref].lower()} subset (paired by fold and seed)")
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig_scores(results, out, k, fold=0):
    files = glob.glob(os.path.join(results, "scores", f"*_f{fold}.csv"))
    if not files:
        print("no score file for fold", fold)
        return
    sc = pd.read_csv(files[0])
    low = sc.nsmallest(k, "seed_disagreement")
    high = sc.nlargest(k, "seed_disagreement")
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    s = ax.scatter(sc["seed_disagreement"], sc["label_disagreement"], c=sc["road_fraction"], cmap="viridis",
                   s=10, alpha=0.8, edgecolor="none")
    for sub, col, name in [(low, COLORS["filtered"], f"{k} most-agreed"), (high, COLORS["antifiltered"], f"{k} most-disputed")]:
        ax.scatter(sub["seed_disagreement"], sub["label_disagreement"], s=22, facecolor="none", edgecolor=col, lw=0.7, label=name)
    cb = fig.colorbar(s, ax=ax, pad=0.02)
    cb.set_label("Road fraction of synthetic mask")
    ax.set_xlabel("Committee disagreement (mean pairwise 1 − Dice)")
    ax.set_ylabel("Disagreement with synthetic mask (1 − Dice)")
    rho = sc["seed_disagreement"].corr(sc["label_disagreement"], method="spearman")
    ax.text(0.02, 0.97, f"Spearman ρ = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=9)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--k", type=int, default=250)
    parser.add_argument("--arch", default="attention")
    args = parser.parse_args()
    out = os.path.join(args.results, "figures")
    os.makedirs(out, exist_ok=True)
    df = load(args.results)
    df = df[(df["arch"] == args.arch) & ((df["k"] == args.k) | df["condition"].isin(["real", "all"]))]
    fig_conditions(df, os.path.join(out, "fig_conditions.png"))
    fig_paired(df, os.path.join(out, "fig_paired.png"))
    fig_scores(args.results, os.path.join(out, "fig_scores.png"), args.k)
    print("figures written to", out, ":", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
