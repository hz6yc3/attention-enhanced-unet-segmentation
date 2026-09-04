# U-Net Road Segmentation from Satellite Imagery

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

Deep learning project implementing U-Net and Attention U-Net for satellite road segmentation.

## 📄 Reference Paper

This project is based on methodologies discussed in:

> **"A Comprehensive Review of U-Net and Its Variants: Advances and Applications in Medical Image Segmentation"**  
> Wang Jiangtao, Nur Intan Raihana Ruhaiyem, Fu Panpan  
> IET Image Processing, 2025  
> [arXiv:2502.06895](https://arxiv.org/abs/2502.06895) | [IET Digital Library](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70019)

## 🎯 Project Overview

This project implements:
1. **Baseline U-Net**: Standard encoder-decoder architecture with skip connections
2. **Attention U-Net**: Enhanced U-Net with attention gates on skip connections

Both models are trained on satellite road segmentation dataset for binary segmentation (road vs. background).

## 📁 Project Structure

```
deep_learning/
├── config.py                 # Configuration and hyperparameters
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── inference.py              # Inference/prediction script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── models/
│   ├── __init__.py          # Model factory
│   ├── unet.py              # Baseline U-Net implementation
│   └── attention_unet.py    # Attention U-Net implementation
│
├── utils/
│   ├── __init__.py          # Utility imports
│   ├── dataset.py           # Dataset loading and augmentation
│   ├── metrics.py           # Evaluation metrics (Dice, IoU, etc.)
│   └── losses.py            # Loss functions (BCE, Dice, Focal)
│
├── data/                     # Dataset directory (download required)
│   ├── test_set_images/     # 50 test images (no ground truth)
│   │   ├── test_1/test_1.png
│   │   └── ...
│   └── training/
│       ├── images/          # 100 original satellite images
│       ├── groundtruth/     # 100 corresponding masks
│       ├── images_generated/    # 1000+ Pix2Pix generated
│       └── groundtruth_generated/
│
├── outputs/                  # Training outputs
│   ├── checkpoints/         # Model checkpoints
│   ├── logs/                # TensorBoard logs
│   └── visualizations/      # Generated visualizations
│
└── notebooks/               # Jupyter notebooks for analysis
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the **Satellite Images for Road Segmentation** dataset from:
- [Kaggle Dataset](https://www.kaggle.com/datasets/sanadalali/satellite-images-for-road-segmentation)

Extract and organize into the `data/` folder as shown in the project structure.

### 3. Train Models

```bash
# Train baseline U-Net (with all data including Pix2Pix generated images)
python train.py --model baseline --epochs 100

# Train Attention U-Net
python train.py --model attention --epochs 100

# Train with ONLY original 100 images (recommended for cleaner experiments)
python train.py --model baseline --epochs 100 --no-generated
python train.py --model attention --epochs 100 --no-generated
```

### 4. Evaluate Models

```bash
# Evaluate baseline model
python evaluate.py --model baseline --checkpoint outputs/checkpoints/baseline_best.pt

# Evaluate attention model
python evaluate.py --model attention --checkpoint outputs/checkpoints/attention_best.pt
```

### 5. Run Inference

```bash
# Single image
python inference.py --model baseline --checkpoint outputs/checkpoints/baseline_best.pt --input image.jpg

# Directory of images
python inference.py --model attention --checkpoint outputs/checkpoints/attention_best.pt --input images/
```

## 🧪 Real-Only Cross-Validation Protocol (journal revision)

The original `train.py` pipeline drew its validation split from the pooled real
+ synthetic images, so validation scores for the "expanded" condition were
dominated by synthetic images. The `experiments/` package replaces that with a
protocol where **synthetic images can only ever appear in training**:

| Piece | What it does |
|---|---|
| `utils/splits.py` | Freezes a stratified split of the 100 real images: 20 test images (never used for training, model selection or scoring) and 5 folds of 16 over the remaining 80. Written once to `splits/real_splits.json`. |
| `experiments/train_run.py` | Trains one (arch, fold, seed, condition) cell for a **fixed number of optimizer steps** (same budget for every condition) and reports per-image metrics on the real validation fold and the real test set. |
| `experiments/score_synthetic.py` | The proposed filter: runs the real-only seed models of a fold on every synthetic image and scores each by inter-seed disagreement (plus label disagreement, probability variance, road fraction). |
| `experiments/run_cv.py` | Driver for the full grid, resumable (skips cells with an existing `result.json`). |
| `experiments/aggregate.py` | Mean ± std, 95% CI, seed sensitivity and paired Wilcoxon / t-tests between conditions. |

Conditions per (arch, fold, seed): `real` (64 real only), `all` (real + 1,003 synthetic),
`random` (real + k random synthetic), `filtered` (real + k lowest-disagreement synthetic),
`antifiltered` (real + k highest-disagreement synthetic, a sanity control).

```bash
# 1. create the frozen splits and inspect them
python -m utils.splits --data data

# 2. run everything (2 archs x 5 folds x 3 seeds x 5 conditions = 150 runs), resumable
python -m experiments.run_cv --k 250

# pilot on one fold first
python -m experiments.run_cv --folds 0 --k 250

# dose-response over subset size
python -m experiments.run_cv --k 100 250 500

# 3. tables and paired tests
python -m experiments.aggregate --results results
```

Each run writes `results/runs/<name>/result.json`, `per_image_test.csv` and the
best-on-validation checkpoint. Scores are in `results/scores/<arch>_f<fold>.csv`.
Use `--max-steps` to change the shared optimizer budget (default 3000 steps of
batch 8) and `--pool-archs` to score synthetic images with both architectures'
seed models pooled. `notebooks/CV_Experiments_Colab.ipynb` runs the same
commands on Colab.

### Running unattended

`scripts/run_budget.sh <results_dir> [max_steps] [stages]` runs the whole budgeted plan
(stages `core,all` by default; add `baseline`, `anti`, `dose`) and aggregates at the end.
It is resumable, so re-launching after an interruption continues where it stopped.

* **Kaggle (free, true background):** open `notebooks/CV_Experiments_Kaggle.ipynb`, attach the
  Kaggle dataset, enable GPU + Internet, then *Save & Run All (Commit)*. Kaggle runs it on its own
  servers and stores `results/` as the notebook output.
* **Colab:** `notebooks/CV_Experiments_Colab.ipynb` has a `nohup` cell that launches the script and
  returns immediately; results go to Drive. The session must stay alive (Colab Pro+ offers
  background execution).
* **Any rented GPU box (RunPod, Lambda, Vast):** `tmux new -s exp` then
  `bash scripts/run_budget.sh results`, detach, come back later.

## 📊 Model Architecture

### Baseline U-Net

The standard U-Net consists of:
- **Encoder**: 4 downsampling blocks (64 → 128 → 256 → 512 channels)
- **Bottleneck**: 1024 channels
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: 1x1 convolution for binary segmentation

### Attention U-Net

Extends baseline with attention gates that:
- Learn to focus on relevant spatial regions
- Suppress irrelevant background features
- Improve segmentation accuracy for small objects

## 📈 Evaluation Metrics

As recommended in the reference paper, we use:
- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Jaccard Index)**: Intersection over Union
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Precision/Recall**: For detailed analysis

## ⚙️ Configuration

Key hyperparameters (in `config.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| Image Size | 256×256 | Input image dimensions |
| Batch Size | 8 | Training batch size |
| Learning Rate | 1e-4 | Initial learning rate |
| Epochs | 100 | Maximum training epochs |
| Loss | BCE+Dice | Combined loss function |
| Optimizer | Adam | With weight decay 1e-5 |

## 📝 Target Publication

This project is formatted for submission to:
- **IEEE Access** or **MDPI Applied Sciences**

Follow the specified journal's LaTeX template for the final paper.

## 🔗 Resources

- **Reference Paper**: [arXiv:2502.06895](https://arxiv.org/abs/2502.06895)
- **Dataset**: [Kaggle - Satellite Road Segmentation](https://www.kaggle.com/datasets/sanadalali/satellite-images-for-road-segmentation)
- **Original U-Net Paper**: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- **Attention U-Net Paper**: [Oktay et al., 2018](https://arxiv.org/abs/1804.03999)

## 📜 License

This project is for educational purposes.

## 🙏 Acknowledgments

- U-Net architecture by Ronneberger et al.
- Attention mechanism by Oktay et al.
- Dataset providers on Kaggle
