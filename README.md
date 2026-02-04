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
