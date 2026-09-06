"""
Duplicate and near-duplicate audit of the synthetic set
=======================================================

Counts exact duplicate masks and images among the Pix2Pix-generated pairs, and
exact matches between synthetic masks and the 100 real masks (including flips
and 90-degree rotations of real masks). Also reports how many distinct
committee-score tuples exist, which shows how much of the score scatter is
overplotted duplicates.

Usage:
    python -m experiments.check_duplicates --data data [--scores results/scores/attention_f0.csv]
"""

import argparse
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

from utils.splits import list_real_pairs, list_synthetic_pairs


def mask_array(p, size=None):
    m = Image.open(p).convert("L")
    if size:
        m = m.resize(size, Image.NEAREST)
    return (np.array(m) > 127)


def h(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--scores", default=None)
    args = parser.parse_args()

    real = list_real_pairs(args.data)
    synth = list_synthetic_pairs(args.data)
    print(f"real pairs: {len(real)} | synthetic pairs: {len(synth)}")

    mask_hashes = [h(mask_array(m)) for _, m in synth]
    img_hashes = [hashlib.md5(open(i, "rb").read()).hexdigest() for i, _ in synth]
    mc, ic = Counter(mask_hashes), Counter(img_hashes)
    print(f"distinct synthetic masks : {len(mc)} / {len(synth)}  "
          f"(largest group repeated {mc.most_common(1)[0][1]}x; "
          f"{sum(v for v in mc.values() if v > 1)} images share a mask with another)")
    print(f"distinct synthetic images: {len(ic)} / {len(synth)}")
    print("mask group sizes (top 10):", sorted(mc.values(), reverse=True)[:10])

    synth_size = Image.open(synth[0][1]).size
    real_variants = {}
    for img, m in real:
        a = mask_array(m, synth_size)
        for k in range(4):
            r = np.rot90(a, k)
            real_variants[h(r)] = img.name
            real_variants[h(np.fliplr(r))] = img.name
    hits = [(i.name, real_variants[hm]) for (i, _), hm in zip(synth, mask_hashes) if hm in real_variants]
    print(f"synthetic masks identical to a (flipped/rotated) real mask: {len(hits)}")
    if hits:
        print("  examples:", hits[:5])

    if args.scores:
        sc = pd.read_csv(args.scores)
        tup = sc[["seed_disagreement", "label_disagreement"]].round(4).drop_duplicates()
        print(f"distinct (seed, label) score pairs in {args.scores}: {len(tup)} / {len(sc)}")


if __name__ == "__main__":
    main()
