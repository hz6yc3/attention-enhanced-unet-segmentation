#!/usr/bin/env bash
# One-shot, resumable execution of the budgeted experiment plan.
#
#   bash scripts/run_budget.sh <results_dir> [max_steps] [stages]
#
# stages is a comma-separated subset of: core,all,baseline,anti,dose (default: core,all)
#   core     attention U-Net, 5 folds x 3 seeds, real / random / filtered (k=250)  ~45 runs
#   all      + real + every synthetic image                                        ~15 runs
#   baseline + baseline U-Net real / random / filtered                             ~45 runs
#   anti     + antifiltered control                                                ~15 runs
#   dose     + random / filtered at k = 100 and 500                                ~60 runs
#   dedup    + deduplicated synthetic set (106 unique images): all / random /
#              filtered / antifiltered / middle at k = 50                           ~75 runs
# Every stage skips runs that already have a result.json, so the script can be
# re-launched after an interruption and it continues where it stopped.
set -euo pipefail

RESULTS_DIR="${1:-results}"
MAX_STEPS="${2:-1200}"
STAGES="${3:-core,all}"
K=250
COMMON="--max-steps $MAX_STEPS --eval-every 100 --out-dir $RESULTS_DIR --quiet"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/run_budget.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== run_budget start $(date) | results=$RESULTS_DIR steps=$MAX_STEPS stages=$STAGES ==="
python -m utils.splits --data data

has() { [[ ",$STAGES," == *",$1,"* ]]; }

if has core; then
  echo "=== stage core $(date) ==="
  python -m experiments.run_cv --archs attention --conditions random filtered --k $K $COMMON
fi
if has all; then
  echo "=== stage all $(date) ==="
  python -m experiments.run_cv --archs attention --conditions all $COMMON
fi
if has baseline; then
  echo "=== stage baseline $(date) ==="
  python -m experiments.run_cv --archs baseline --conditions random filtered --k $K $COMMON
fi
if has anti; then
  echo "=== stage anti $(date) ==="
  python -m experiments.run_cv --archs attention --conditions antifiltered --k $K $COMMON
fi
if has dedup; then
  echo "=== stage dedup $(date) ==="
  python -m experiments.run_cv --archs attention --dedup --conditions all random filtered antifiltered middle --k 50 $COMMON
fi
if has dose; then
  echo "=== stage dose $(date) ==="
  python -m experiments.run_cv --archs attention --conditions random filtered --k 100 500 $COMMON
fi

echo "=== aggregate $(date) ==="
python -m experiments.aggregate --results "$RESULTS_DIR"
echo "=== run_budget finished $(date) ==="
