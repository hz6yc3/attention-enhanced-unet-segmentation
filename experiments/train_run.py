"""
Single step-budgeted training run under the real-only evaluation protocol
==========================================================================

One call of ``run_experiment`` trains one model for one
(architecture, fold, seed, synthetic condition) cell and writes:

    <out_dir>/runs/<run_name>/result.json        summary metrics
    <out_dir>/runs/<run_name>/per_image_test.csv  per-image test metrics
    <out_dir>/runs/<run_name>/best.pt             best-on-validation weights

Design decisions that answer the ICTAI reviews:

* Validation and test images are REAL images only (see utils/splits.py).
  Synthetic images are added to the training list and nowhere else.
* Every condition trains for the same number of optimizer steps with the same
  cosine schedule, so "more data" is not confounded with "more updates".
* Model selection uses the real validation fold; the real test set is scored
  exactly once, with the selected weights.

Synthetic conditions:
    real          real training fold only
    all           real + every synthetic image
    random        real + k synthetic images drawn at random (seeded)
    filtered      real + k synthetic images with the LOWEST disagreement score
    antifiltered  real + k synthetic images with the HIGHEST disagreement score
                  (sanity control: if the filter is meaningful this should hurt)

Usage:
    python -m experiments.train_run --arch attention --fold 0 --seed 42 --condition real
    python -m experiments.train_run --arch attention --fold 0 --seed 42 \
        --condition filtered --k 250 --scores results/scores/attention_f0.csv
"""

import argparse
import copy
import functools
import hashlib
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import get_model
from utils.dataset import RoadSegmentationDataset, get_train_transforms, get_val_transforms
from utils.losses import get_loss_function
from utils.metrics import per_image_metrics
from utils.splits import (
    DEFAULT_SPLIT_PATH,
    get_fold_paths,
    list_synthetic_pairs,
    load_or_create_splits,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

CONDITIONS = ("real", "all", "random", "filtered", "antifiltered", "middle")


@dataclass
class RunConfig:
    arch: str = "attention"                 # "baseline" | "attention"
    fold: int = 0
    seed: int = 42
    condition: str = "real"
    k: int = 0                              # number of synthetic images for random/filtered/antifiltered
    scores_path: Optional[str] = None       # CSV from experiments/score_synthetic.py
    score_key: str = "seed_disagreement"    # column used to rank synthetic images
    dedup: bool = False                     # collapse byte-identical synthetic images before selection

    data_root: str = "data"
    splits_path: str = DEFAULT_SPLIT_PATH
    out_dir: str = "results"

    max_steps: int = 3000                   # identical optimizer budget for every condition
    eval_every: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    min_lr: float = 1e-6
    loss_function: str = "bce_dice"
    gradient_clip: float = 1.0
    use_amp: bool = True
    num_workers: int = 4

    image_size: Tuple[int, int] = (256, 256)
    encoder_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    bottleneck_channels: int = 1024
    dropout_rate: float = 0.1

    save_checkpoint: bool = True
    device: Optional[str] = None            # auto if None

    @property
    def run_name(self) -> str:
        cond = self.condition if self.condition in ("real", "all") else f"{self.condition}{self.k}"
        if self.dedup and self.condition != "real":
            cond = "u" + cond
        return f"{self.arch}_{cond}_f{self.fold}_s{self.seed}"

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / "runs" / self.run_name


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dedup_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Keep one representative (first by name) per byte-identical synthetic image."""
    seen, out = set(), []
    for img, mask in pairs:
        with open(img, "rb") as f:
            key = hashlib.md5(f.read()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append((img, mask))
    return out


def select_synthetic(cfg: RunConfig) -> List[Tuple[str, str]]:
    """Return the list of synthetic (image, mask) paths for this condition."""
    if cfg.condition == "real":
        return []
    pairs = [(str(i), str(m)) for i, m in list_synthetic_pairs(cfg.data_root)]
    if len(pairs) == 0:
        raise FileNotFoundError("No synthetic images found but a synthetic condition was requested")
    if cfg.dedup:
        pairs = dedup_pairs(pairs)
    if cfg.condition == "all":
        return pairs
    if cfg.k <= 0 or cfg.k > len(pairs):
        raise ValueError(f"k must be in [1, {len(pairs)}] for condition '{cfg.condition}'")

    if cfg.condition == "random":
        rng = np.random.RandomState(cfg.seed * 1000 + cfg.fold)
        idx = rng.choice(len(pairs), size=cfg.k, replace=False)
        return [pairs[i] for i in sorted(idx)]

    if cfg.condition in ("filtered", "antifiltered", "middle"):
        if not cfg.scores_path or not os.path.exists(cfg.scores_path):
            raise FileNotFoundError(
                f"Condition '{cfg.condition}' needs a scores CSV (got {cfg.scores_path}). "
                "Run experiments/score_synthetic.py first."
            )
        scores = pd.read_csv(cfg.scores_path)
        if cfg.score_key not in scores.columns:
            raise KeyError(f"'{cfg.score_key}' not in {cfg.scores_path} columns: {list(scores.columns)}")
        by_name = {Path(i).name: (i, m) for i, m in pairs}
        scores = scores[scores["name"].isin(by_name)]
        ranked = scores.sort_values([cfg.score_key, "name"], ascending=[True, True])["name"].tolist()
        if cfg.condition == "filtered":        # lowest disagreement
            chosen = ranked[: cfg.k]
        elif cfg.condition == "antifiltered":  # highest disagreement
            chosen = ranked[-cfg.k:]
        else:                                  # middle band centred on the median
            start = (len(ranked) - cfg.k) // 2
            chosen = ranked[start: start + cfg.k]
        return [by_name[n] for n in chosen]

    raise ValueError(f"Unknown condition {cfg.condition}; choose from {CONDITIONS}")


def _seed_worker(seed: int, worker_id: int) -> None:
    np.random.seed((seed + worker_id) % (2 ** 31))
    random.seed(seed + worker_id)


def make_loader(
    imgs: List[str], masks: List[str], cfg: RunConfig, train: bool, seed: int
) -> DataLoader:
    tf = get_train_transforms(cfg.image_size) if train else get_val_transforms(cfg.image_size)
    ds = RoadSegmentationDataset(imgs, masks, transform=tf, image_size=cfg.image_size)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=train,
        drop_last=train and len(ds) >= cfg.batch_size,
        num_workers=min(cfg.num_workers, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
        generator=gen,
        worker_init_fn=functools.partial(_seed_worker, seed),
    )


def infinite(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def build_model(cfg: RunConfig) -> torch.nn.Module:
    return get_model(
        model_type=cfg.arch,
        in_channels=3,
        num_classes=1,
        encoder_channels=list(cfg.encoder_channels),
        bottleneck_channels=cfg.bottleneck_channels,
        use_batch_norm=True,
        dropout_rate=cfg.dropout_rate,
    )


def lr_lambda_factory(warmup: int, total: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return f


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, names: List[str]) -> pd.DataFrame:
    """Per-image metrics for every image in ``loader`` (order must match ``names``)."""
    model.eval()
    rows = []
    i = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        probs = torch.sigmoid(model(images))
        m = per_image_metrics(probs, masks)
        for j in range(images.shape[0]):
            rows.append({"name": names[i + j], **{k: float(v[j]) for k, v in m.items()}})
        i += images.shape[0]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """Mean per-image metrics plus pooled (pixel-level) Dice/IoU."""
    tp, fp, fn = df["tp"].sum(), df["fp"].sum(), df["fn"].sum()
    return {
        "dice": float(df["dice"].mean()),
        "iou": float(df["iou"].mean()),
        "precision": float(df["precision"].mean()),
        "recall": float(df["recall"].mean()),
        "accuracy": float(df["accuracy"].mean()),
        "dice_std_images": float(df["dice"].std(ddof=0)),
        "pooled_dice": float(2 * tp / max(1.0, 2 * tp + fp + fn)),
        "pooled_iou": float(tp / max(1.0, tp + fp + fn)),
        "n_images": int(len(df)),
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def run_experiment(cfg: RunConfig, force: bool = False, verbose: bool = True) -> dict:
    run_dir = cfg.run_dir
    result_path = run_dir / "result.json"
    if result_path.exists() and not force:
        with open(result_path) as f:
            return json.load(f)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    use_amp = cfg.use_amp and device.type == "cuda"

    splits = load_or_create_splits(cfg.data_root, cfg.splits_path)
    fold = get_fold_paths(cfg.data_root, splits, cfg.fold)
    synth = select_synthetic(cfg)

    train_imgs = list(fold["train"][0]) + [s[0] for s in synth]
    train_masks = list(fold["train"][1]) + [s[1] for s in synth]
    val_imgs, val_masks = fold["val"]
    test_imgs, test_masks = fold["test"]

    train_loader = make_loader(train_imgs, train_masks, cfg, train=True, seed=cfg.seed)
    val_loader = make_loader(val_imgs, val_masks, cfg, train=False, seed=cfg.seed)
    test_loader = make_loader(test_imgs, test_masks, cfg, train=False, seed=cfg.seed)

    model = build_model(cfg).to(device)
    criterion = get_loss_function(loss_type=cfg.loss_function, bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(cfg.warmup_steps, cfg.max_steps, cfg.min_lr / cfg.learning_rate)
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    if verbose:
        print(f"[{cfg.run_name}] device={device} train={len(train_imgs)} "
              f"(real={len(fold['train'][0])}, synthetic={len(synth)}) "
              f"val={len(val_imgs)} test={len(test_imgs)} steps={cfg.max_steps}")

    best_val_dice = -1.0
    best_step = 0
    best_state = None
    history = []
    losses = []
    t0 = time.time()
    batches = infinite(train_loader)

    for step in range(1, cfg.max_steps + 1):
        model.train()
        images, masks = next(batches)
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()
        scheduler.step()
        losses.append(float(loss.item()))

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            val_df = evaluate(model, val_loader, device, [Path(p).name for p in val_imgs])
            val_dice = float(val_df["dice"].mean())
            history.append({"step": step, "train_loss": float(np.mean(losses[-cfg.eval_every:])),
                            "val_dice": val_dice, "lr": optimizer.param_groups[0]["lr"]})
            if val_dice > best_val_dice:
                best_val_dice, best_step = val_dice, step
                best_state = copy.deepcopy(model.state_dict())
            if verbose:
                print(f"  step {step:5d} loss {history[-1]['train_loss']:.4f} "
                      f"val_dice {val_dice:.4f} best {best_val_dice:.4f}@{best_step}")

    model.load_state_dict(best_state)
    val_df = evaluate(model, val_loader, device, [Path(p).name for p in val_imgs])
    test_df = evaluate(model, test_loader, device, [Path(p).name for p in test_imgs])
    test_df.to_csv(run_dir / "per_image_test.csv", index=False)
    val_df.to_csv(run_dir / "per_image_val.csv", index=False)
    if cfg.save_checkpoint:
        torch.save({"model_state_dict": best_state, "config": asdict(cfg), "best_step": best_step},
                   run_dir / "best.pt")

    result = {
        "run_name": cfg.run_name,
        "arch": cfg.arch, "fold": cfg.fold, "seed": cfg.seed,
        "condition": cfg.condition, "k": len(synth), "score_key": cfg.score_key, "dedup": cfg.dedup,
        "n_train_real": len(fold["train"][0]), "n_train_synthetic": len(synth),
        "max_steps": cfg.max_steps, "best_step": best_step,
        "val": summarize(val_df), "test": summarize(test_df),
        "history": history,
        "synthetic_names": [Path(s[0]).name for s in synth],
        "elapsed_sec": time.time() - t0,
        "device": str(device),
        "config": asdict(cfg),
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    if True:
        print(f"[{cfg.run_name}] done in {result['elapsed_sec']/60:.1f} min | "
              f"val dice {result['val']['dice']:.4f} | TEST dice {result['test']['dice']:.4f} "
              f"iou {result['test']['iou']:.4f}")
    return result


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default="data")
    parser.add_argument("--splits", default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--encoder-channels", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--bottleneck", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-key", default="seed_disagreement")


def config_from_args(args, **overrides) -> RunConfig:
    cfg = RunConfig(
        data_root=args.data, splits_path=args.splits, out_dir=args.out_dir,
        max_steps=args.max_steps, eval_every=args.eval_every, batch_size=args.batch_size,
        learning_rate=args.lr, image_size=(args.image_size, args.image_size),
        encoder_channels=list(args.encoder_channels), bottleneck_channels=args.bottleneck,
        num_workers=args.num_workers, use_amp=not args.no_amp, device=args.device,
        score_key=args.score_key,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=["baseline", "attention"], default="attention")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition", choices=CONDITIONS, default="real")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--scores", default=None, help="scores CSV for filtered/antifiltered")
    parser.add_argument("--dedup", action="store_true", help="collapse duplicate synthetic images before selection")
    parser.add_argument("--force", action="store_true", help="re-run even if result.json exists")
    add_common_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args, arch=args.arch, fold=args.fold, seed=args.seed,
                           condition=args.condition, k=args.k, scores_path=args.scores, dedup=args.dedup)
    run_experiment(cfg, force=args.force)


if __name__ == "__main__":
    main()
