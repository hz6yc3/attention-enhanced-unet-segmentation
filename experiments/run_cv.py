"""
Cross-validation driver for the seed-disagreement filtering study
=================================================================

Runs the full experimental grid, skipping any cell whose result.json already
exists, so it can be stopped and resumed across Colab sessions.

Stages
------
1. real      Train every (arch, fold, seed) on the real training fold only.
             These checkpoints are both the "no synthetic data" baseline and
             the scoring committee for stage 2.
2. score     For every (arch, fold), score all synthetic images by the
             disagreement between the real-only seed models of that fold.
3. subsets   Train every (arch, fold, seed) on real + synthetic under each
             requested condition (all / random / filtered / antifiltered) and
             each requested k, with the same step budget as stage 1.

Afterwards run ``python -m experiments.aggregate`` to build the tables.

Example (default grid: 2 archs x 5 folds x 3 seeds x 5 conditions = 150 runs):
    python -m experiments.run_cv --k 250
Smaller pilot on one fold:
    python -m experiments.run_cv --folds 0 --seeds 42 123 456 --k 250
Dose-response over k (adds runs for each k):
    python -m experiments.run_cv --k 100 250 500
"""

import argparse
import gc
import time
from pathlib import Path
from typing import List

import torch

from experiments.score_synthetic import score_fold, scores_path_for
from experiments.train_run import RunConfig, run_experiment, add_common_args, config_from_args


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archs", nargs="+", default=["baseline", "attention"], choices=["baseline", "attention"])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--k", type=int, nargs="+", default=[250], help="synthetic subset sizes")
    parser.add_argument("--conditions", nargs="+", default=["all", "random", "filtered", "antifiltered"],
                        choices=["all", "random", "filtered", "antifiltered"])
    parser.add_argument("--pool-archs", action="store_true",
                        help="score synthetic images with the real-only models of BOTH archs pooled "
                             "(default: each arch uses its own seed models)")
    parser.add_argument("--stage", choices=["all", "real", "score", "subsets"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="list the runs without training")
    parser.add_argument("--quiet", action="store_true")
    add_common_args(parser)
    args = parser.parse_args()
    verbose = not args.quiet

    def cfg_for(**kw) -> RunConfig:
        return config_from_args(args, **kw)

    planned: List[str] = []
    t_start = time.time()

    # ---------------- Stage 1: real-only ----------------
    if args.stage in ("all", "real"):
        for arch in args.archs:
            for fold in args.folds:
                for seed in args.seeds:
                    cfg = cfg_for(arch=arch, fold=fold, seed=seed, condition="real", k=0)
                    done = (cfg.run_dir / "result.json").exists()
                    planned.append(f"{'done ' if done else 'TODO '}{cfg.run_name}")
                    if not args.dry_run and not done:
                        run_experiment(cfg, verbose=verbose)
                        _free()

    # ---------------- Stage 2: scoring ----------------
    score_files = {}
    if args.stage in ("all", "score", "subsets"):
        for arch in args.archs:
            score_archs = list(args.archs) if args.pool_archs else [arch]
            for fold in args.folds:
                out = scores_path_for(args.out_dir, fold, score_archs)
                score_files[(arch, fold)] = out
                if args.stage == "subsets":
                    continue
                done = out.exists()
                planned.append(f"{'done ' if done else 'TODO '}score {out.name}")
                if not args.dry_run and not done:
                    score_fold(cfg_for(fold=fold), score_archs, args.seeds, out, verbose=verbose)
                    _free()

    # ---------------- Stage 3: synthetic conditions ----------------
    if args.stage in ("all", "subsets"):
        for arch in args.archs:
            for fold in args.folds:
                for seed in args.seeds:
                    for cond in args.conditions:
                        ks = [0] if cond == "all" else args.k
                        for k in ks:
                            cfg = cfg_for(arch=arch, fold=fold, seed=seed, condition=cond, k=k,
                                          scores_path=str(score_files.get((arch, fold), "")))
                            done = (cfg.run_dir / "result.json").exists()
                            planned.append(f"{'done ' if done else 'TODO '}{cfg.run_name}")
                            if not args.dry_run and not done:
                                run_experiment(cfg, verbose=verbose)
                                _free()

    n_todo = sum(p.startswith("TODO") for p in planned)
    print("\n".join(planned))
    print(f"\n{len(planned)} cells planned, {n_todo} {'remaining' if args.dry_run else 'were run'}; "
          f"elapsed {(time.time() - t_start) / 60:.1f} min")
    if not args.dry_run:
        print("Next: python -m experiments.aggregate --results", args.out_dir)


if __name__ == "__main__":
    main()
