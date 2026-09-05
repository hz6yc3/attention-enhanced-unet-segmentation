"""
Progress report for a running experiment grid
=============================================

Reads every finished results/runs/*/result.json and prints per-run time,
validation and test Dice, plus an estimate of the remaining time.

Usage:
    python -m experiments.progress --results results --planned 60
"""

import argparse
import glob
import json
import os
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--planned", type=int, default=60, help="total runs in the launched plan")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, "runs", "*", "result.json")), key=os.path.getmtime)
    started = len(glob.glob(os.path.join(args.results, "runs", "*")))
    print(f"{'run':30s} {'min':>5s}  {'val Dice':>8s}  {'TEST Dice':>9s}  {'TEST IoU':>8s}  best step")
    times = []
    for p in files:
        r = json.load(open(p))
        times.append(r["elapsed_sec"] / 60)
        print(f"{r['run_name']:30s} {times[-1]:5.1f}  {r['val']['dice']:8.4f}  {r['test']['dice']:9.4f}  "
              f"{r['test']['iou']:8.4f}  {r['best_step']}/{r['max_steps']}")
    print(f"\nfinished: {len(files)} | in progress: {max(0, started - len(files))} | planned: {args.planned}")
    if times:
        per = sum(times) / len(times)
        remaining = max(0, args.planned - len(files))
        print(f"average {per:.1f} min per run -> about {remaining * per / 60:.1f} h remaining for {remaining} runs")
    latest = max(files, key=os.path.getmtime) if files else None
    if latest:
        print(f"last run finished {(time.time() - os.path.getmtime(latest)) / 60:.0f} min ago")


if __name__ == "__main__":
    main()
