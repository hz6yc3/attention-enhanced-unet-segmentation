"""
Seed-disagreement scoring of synthetic images
=============================================

The proposed filter: models trained independently (different seeds) on REAL
images only are run on every synthetic image. Where those models disagree with
each other, the synthetic image is far from what real data supports (GAN
artefacts, implausible layouts) and is a candidate for removal. Where they
agree, the image is "safe" to add.

For one fold this script loads the real-only checkpoints of the requested
seeds (and optionally both architectures), predicts every synthetic image, and
writes one row per synthetic image with several candidate scores:

    seed_disagreement        mean pairwise (1 - Dice) between binarised seed predictions
    prob_variance            mean per-pixel variance of the predicted probabilities
    label_disagreement       mean over seeds of (1 - Dice) between prediction and the
                             synthetic ground-truth mask
    ensemble_label_disagreement
                             (1 - Dice) between the seed-averaged prediction and the mask
    road_fraction            fraction of road pixels in the synthetic mask
    pred_road_fraction       mean predicted road fraction across seeds

Only the checkpoints of the SAME fold are used, so the held-out validation
fold and the test set never influence which synthetic images are kept.

Usage:
    python -m experiments.score_synthetic --fold 0 --archs attention --seeds 42 123 456
"""

import argparse
import itertools
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from experiments.train_run import RunConfig, build_model, make_loader, pick_device, add_common_args, config_from_args
from utils.splits import list_synthetic_pairs


def _dice(a: torch.Tensor, b: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Per-image Dice between two binary tensors shaped (B, 1, H, W)."""
    a, b = a.flatten(1), b.flatten(1)
    inter = (a * b).sum(1)
    return (2 * inter + smooth) / (a.sum(1) + b.sum(1) + smooth)


def load_real_only_models(cfg: RunConfig, archs: List[str], seeds: List[int], device: torch.device):
    models = []
    for arch in archs:
        for seed in seeds:
            c = RunConfig(**{**cfg.__dict__, "arch": arch, "seed": seed, "condition": "real", "k": 0})
            ckpt = c.run_dir / "best.pt"
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Missing real-only checkpoint {ckpt}. Train condition 'real' for "
                    f"arch={arch} fold={cfg.fold} seed={seed} first."
                )
            state = torch.load(ckpt, map_location="cpu")
            m = build_model(c)
            m.load_state_dict(state["model_state_dict"])
            m.to(device).eval()
            models.append((f"{arch}_s{seed}", m))
    return models


@torch.no_grad()
def score_fold(cfg: RunConfig, archs: List[str], seeds: List[int], out_path: Path,
               threshold: float = 0.5, verbose: bool = True) -> pd.DataFrame:
    device = pick_device(cfg.device)
    models = load_real_only_models(cfg, archs, seeds, device)
    if len(models) < 2:
        raise ValueError("Need at least two models (seeds and/or archs) to measure disagreement")

    pairs = list_synthetic_pairs(cfg.data_root)
    names = [p[0].name for p in pairs]
    loader = make_loader([str(p[0]) for p in pairs], [str(p[1]) for p in pairs], cfg, train=False, seed=0)

    rows = []
    i = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        probs = torch.stack([torch.sigmoid(m(images)) for _, m in models])   # (S, B, 1, H, W)
        bins = (probs > threshold).float()
        S = probs.shape[0]

        pair_dis = torch.stack([1 - _dice(bins[a], bins[b]) for a, b in itertools.combinations(range(S), 2)]).mean(0)
        prob_var = probs.var(0, unbiased=False).flatten(1).mean(1)
        label_dis = torch.stack([1 - _dice(bins[s], masks) for s in range(S)]).mean(0)
        ens_bin = (probs.mean(0) > threshold).float()
        ens_label_dis = 1 - _dice(ens_bin, masks)
        road_frac = masks.flatten(1).mean(1)
        pred_frac = bins.flatten(2).mean(2).mean(0)

        for j in range(images.shape[0]):
            rows.append({
                "name": names[i + j],
                "seed_disagreement": float(pair_dis[j]),
                "prob_variance": float(prob_var[j]),
                "label_disagreement": float(label_dis[j]),
                "ensemble_label_disagreement": float(ens_label_dis[j]),
                "road_fraction": float(road_frac[j]),
                "pred_road_fraction": float(pred_frac[j]),
            })
        i += images.shape[0]
        if verbose and (i % (cfg.batch_size * 25) == 0 or i == len(names)):
            print(f"  scored {i}/{len(names)}")

    df = pd.DataFrame(rows).sort_values("name").reset_index(drop=True)
    df.attrs["models"] = [n for n, _ in models]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"Wrote {out_path} ({len(df)} synthetic images, {len(models)} models: "
              f"{', '.join(n for n, _ in models)})")
        print(df[["seed_disagreement", "prob_variance", "label_disagreement"]].describe().loc[["mean", "std", "min", "50%", "max"]])
    return df


def scores_path_for(out_dir: str, fold: int, archs: List[str]) -> Path:
    tag = "+".join(sorted(archs))
    return Path(out_dir) / "scores" / f"{tag}_f{fold}.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--archs", nargs="+", default=["attention"], choices=["baseline", "attention"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--out", default=None, help="output CSV (default results/scores/<archs>_f<fold>.csv)")
    add_common_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args, fold=args.fold)
    out = Path(args.out) if args.out else scores_path_for(args.out_dir, args.fold, args.archs)
    score_fold(cfg, args.archs, args.seeds, out)


if __name__ == "__main__":
    main()
