"""
Frozen real-image split protocol
================================

The original pipeline drew a random validation split from the pooled set of
real and Pix2Pix-generated images, so validation metrics for the "expanded"
condition were dominated by synthetic images. This module defines a fixed,
stratified, real-only protocol that is written to disk once and reused by every
experiment so that:

* a fixed set of real images is held out as the TEST set and never touched
  during training, model selection, or synthetic-sample scoring;
* the remaining real images are divided into K folds for cross-validation
  (one fold is the validation fold, the rest are the real training images);
* synthetic images can only ever enter the training portion of a fold.

Stratification uses the road-pixel fraction of each mask so that every fold and
the test set cover the full range of sparse to dense road scenes.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

SPLIT_VERSION = 1
DEFAULT_SPLIT_PATH = "splits/real_splits.json"


def _road_fraction(mask_path: Path) -> float:
    mask = np.array(Image.open(mask_path).convert("L"))
    return float((mask > 127).mean())


def list_real_pairs(data_root: str) -> List[Tuple[Path, Path]]:
    """Sorted (image, mask) pairs for the real annotated images."""
    root = Path(data_root) / "training"
    pairs = []
    for img in sorted((root / "images").glob("*.png")):
        mask = root / "groundtruth" / img.name
        if mask.exists():
            pairs.append((img, mask))
    return pairs


def list_synthetic_pairs(data_root: str) -> List[Tuple[Path, Path]]:
    """Sorted (image, mask) pairs for the Pix2Pix-generated images."""
    root = Path(data_root) / "training"
    pairs = []
    img_dir = root / "images_generated"
    if not img_dir.exists():
        return pairs
    for img in sorted(img_dir.glob("*.png")):
        mask = root / "groundtruth_generated" / img.name
        if mask.exists():
            pairs.append((img, mask))
    return pairs


def build_splits(
    data_root: str,
    n_test: int = 20,
    n_folds: int = 5,
    seed: int = 2024,
) -> Dict:
    """
    Build a stratified test set plus K stratified folds over the real images.

    Images are sorted by road fraction and grouped into consecutive bins. One
    image per bin (chosen at random with the given seed) forms the test set, so
    the test set spans the full range of road density. The remaining images are
    re-binned and each bin is dealt across the K folds in a random order.
    """
    pairs = list_real_pairs(data_root)
    if len(pairs) == 0:
        raise FileNotFoundError(f"No real image/mask pairs found under {data_root}/training")

    names = [p[0].name for p in pairs]
    fractions = {p[0].name: _road_fraction(p[1]) for p in pairs}
    rng = np.random.RandomState(seed)

    ordered = sorted(names, key=lambda n: fractions[n])
    n_total = len(ordered)
    if n_test <= 0 or n_test >= n_total:
        raise ValueError("n_test must be between 1 and the number of real images minus 1")

    # Stratified test selection: one image from each of n_test equal-size bins.
    bins = np.array_split(np.arange(n_total), n_test)
    test = []
    for b in bins:
        test.append(ordered[rng.choice(b)])
    test_set = set(test)

    remaining = [n for n in ordered if n not in test_set]
    # Deal each consecutive group of n_folds images across the folds.
    folds: List[List[str]] = [[] for _ in range(n_folds)]
    for start in range(0, len(remaining), n_folds):
        group = remaining[start:start + n_folds]
        order = rng.permutation(n_folds)[: len(group)]
        for name, f in zip(group, order):
            folds[int(f)].append(name)

    return {
        "version": SPLIT_VERSION,
        "seed": seed,
        "n_test": n_test,
        "n_folds": n_folds,
        "test": sorted(test),
        "folds": [sorted(f) for f in folds],
        "road_fraction": {n: round(fractions[n], 6) for n in names},
    }


def save_splits(splits: Dict, path: str = DEFAULT_SPLIT_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str = DEFAULT_SPLIT_PATH) -> Dict:
    with open(path) as f:
        return json.load(f)


def load_or_create_splits(
    data_root: str,
    path: str = DEFAULT_SPLIT_PATH,
    n_test: int = 20,
    n_folds: int = 5,
    seed: int = 2024,
) -> Dict:
    """Load the frozen split file, creating it on first use."""
    p = Path(path)
    if p.exists():
        splits = load_splits(str(p))
        real_names = {img.name for img, _ in list_real_pairs(data_root)}
        referenced = set(splits["test"]) | {n for f in splits["folds"] for n in f}
        missing = referenced - real_names
        if missing:
            raise RuntimeError(
                f"Split file {p} references {len(missing)} images not present in {data_root}"
            )
        return splits
    splits = build_splits(data_root, n_test=n_test, n_folds=n_folds, seed=seed)
    save_splits(splits, str(p))
    print(f"Created frozen split file: {p}")
    return splits


def get_fold_paths(data_root: str, splits: Dict, fold: int) -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Real image/mask path lists for one cross-validation fold.

    Returns a dict with keys 'train', 'val', 'test', each mapping to
    (image_paths, mask_paths). Synthetic images are NOT included here.
    """
    if fold < 0 or fold >= splits["n_folds"]:
        raise ValueError(f"fold must be in [0, {splits['n_folds'] - 1}]")
    root = Path(data_root) / "training"

    def to_paths(names: List[str]) -> Tuple[List[str], List[str]]:
        imgs = [str(root / "images" / n) for n in names]
        masks = [str(root / "groundtruth" / n) for n in names]
        return imgs, masks

    val_names = splits["folds"][fold]
    train_names = [n for i, f in enumerate(splits["folds"]) if i != fold for n in f]
    return {
        "train": to_paths(sorted(train_names)),
        "val": to_paths(sorted(val_names)),
        "test": to_paths(sorted(splits["test"])),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create or inspect the frozen real-image splits")
    parser.add_argument("--data", default="data")
    parser.add_argument("--path", default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    s = load_or_create_splits(args.data, args.path, args.n_test, args.n_folds, args.seed)
    rf = s["road_fraction"]
    print(f"Test set: {len(s['test'])} images, road fraction "
          f"{np.mean([rf[n] for n in s['test']]):.3f} ± {np.std([rf[n] for n in s['test']]):.3f}")
    for i, f in enumerate(s["folds"]):
        print(f"Fold {i}: {len(f)} images, road fraction "
              f"{np.mean([rf[n] for n in f]):.3f} ± {np.std([rf[n] for n in f]):.3f}")
    print(f"Synthetic pairs available: {len(list_synthetic_pairs(args.data))}")
