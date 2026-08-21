# LightPestFormer: Lightweight Transformer-Inspired Pest Recognition

> **Paper:** *LightPestFormer: A Lightweight Multi-Scale Semantic Attention Network for Fine-Grained Insect Classification and Detection*
> **Journal:** *(Under Review — Q1 Journal)*
> **Authors:** Mageshkumar N S et al.

---

## Overview

LightPestFormer is a dual-branch framework for agricultural pest recognition:

| Branch | Backbone | Task | Dataset | Key Metric |
|---|---|---|---|---|
| **LightPestFormer** | EfficientNet-B2 + MSSA | Classification | INSECT204 | Top-1 Acc = **89.73%** |
| **YOLO11n** | CSPNet | Detection | AgroPest-12 | mAP@50 = **79.4%** |

![Architecture](results/figures/ARCH_architecture_diagram.png)

---

## Repository Structure

```
LightPestFormer/
├── configs/
│   ├── lightpestformer.yaml      # classification training config
│   └── yolo11n_agropest.yaml     # YOLO detection config
├── datasets/
│   ├── insect204_split.py        # train/val/test split script
│   ├── agropest12_prepare.py     # YOLO-format label prep
│   └── transforms.py             # augmentation pipeline
├── models/
│   ├── lightpestformer.py        # full model (backbone + MSSA + head)
│   ├── mssa.py                   # Multi-Scale Semantic Attention module
│   └── hybrid_loss.py            # HybridLoss (LSCE + SCL)
├── train/
│   ├── train_classification.py   # two-phase LightPestFormer training
│   └── train_detection.py        # YOLO11n training wrapper
├── evaluate/
│   ├── eval_classification.py    # Top-1/5, per-class metrics, ECE
│   └── eval_detection.py         # mAP@50, mAP@50-95, PR curve
├── xai/
│   ├── gradcam.py                # GradCAM on MSSA output
│   ├── gradcam_pp.py             # GradCAM++ implementation
│   ├── tsne_features.py          # t-SNE of 256-dim embeddings
│   └── attention_maps.py         # MSSA per-scale attention visualization
├── scripts/
│   ├── make_arch_diagram.py      # generate architecture figure
│   └── export_onnx.py            # export model to ONNX
├── results/
│   ├── figures/                  # all paper figures (G1–G20 + ARCH)
│   └── tables/                   # CSV exports of T1–T8
├── weights/
│   └── .gitkeep                  # model checkpoints go here (see below)
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/LightPestFormer.git
cd LightPestFormer
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate lightpestformer
```

### 2. Prepare datasets

**INSECT204** — download from [IP102 benchmark](https://github.com/xpwu95/IP102) and reorganise:

```bash
python datasets/insect204_split.py \
    --src /path/to/INSECT204_raw \
    --dst data/INSECT204 \
    --split 0.8 0.1 0.1
```

**AgroPest-12** — place in YOLO format under `data/AgroPest12/`:

```
data/AgroPest12/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Then run `python datasets/agropest12_prepare.py --root data/AgroPest12`.

### 3. Train LightPestFormer (classification)

```bash
python train/train_classification.py --config configs/lightpestformer.yaml
```

Training runs in two phases automatically:
- **Phase 1** (5 epochs): Backbone frozen, MSSA + head only, Adam lr=1e-3
- **Phase 2** (40 epochs): All params unfrozen, cosine LR, early-stop patience=10

Checkpoints saved to `weights/lightpestformer_best.pth`.

### 4. Train YOLO11n (detection)

```bash
python train/train_detection.py --config configs/yolo11n_agropest.yaml
```

Best epoch saved to `weights/yolo11n_best.pt` (runs 85 epochs, early-stop patience=30).

### 5. Evaluate

```bash
# Classification
python evaluate/eval_classification.py \
    --weights weights/lightpestformer_best.pth \
    --data data/INSECT204

# Detection
python evaluate/eval_detection.py \
    --weights weights/yolo11n_best.pt \
    --data data/AgroPest12
```

### 6. XAI visualizations

```bash
# GradCAM
python xai/gradcam.py --weights weights/lightpestformer_best.pth --image path/to/sample.jpg

# t-SNE of test-set embeddings
python xai/tsne_features.py --weights weights/lightpestformer_best.pth --data data/INSECT204/test

# MSSA attention maps
python xai/attention_maps.py --weights weights/lightpestformer_best.pth --image path/to/sample.jpg
```

---

## Pre-trained Weights

Due to file size, weights are hosted on Google Drive / Zenodo:

| Model | Dataset | Acc / mAP | Download |
|---|---|---|---|
| LightPestFormer | INSECT204 | Top-1 89.73% | [Download](#) |
| YOLO11n | AgroPest-12 | mAP@50 79.4% | [Download](#) |

Place downloaded files in the `weights/` folder.

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Backbone | EfficientNet-B2 (timm, ImageNet pre-trained) |
| MSSA projection dim | 256 |
| Dropout | 0.30 |
| Phase 1 lr | 1e-3 (Adam) |
| Phase 2 backbone lr | 1e-4 |
| Phase 2 MSSA/head lr | 5e-4 |
| HybridLoss α | 0.70 → 0.85 (annealed) |
| Label smoothing ε | 0.10 |
| Contrastive τ | 0.07 |
| Input resolution | 224 × 224 |
| Parameters | 8.21 M |
| FLOPs | 634.54 M |
| Inference latency | 12.62 ms (GPU) |

---

## Results

### Classification — INSECT204

| Method | Top-1 Acc | Top-5 Acc | Params |
|---|---|---|---|
| ResNet-50 | 79.14% | 93.82% | 25.6 M |
| EfficientNet-B2 | 84.31% | 96.15% | 9.1 M |
| ViT-B/16 | 85.67% | 97.03% | 86.0 M |
| **LightPestFormer (ours)** | **89.73%** | **98.21%** | **8.21 M** |

### Detection — AgroPest-12

| Method | mAP@50 | mAP@50-95 | FPS |
|---|---|---|---|
| YOLOv5s | 71.2% | 43.8% | 140 |
| YOLOv8n | 75.6% | 47.3% | 128 |
| **YOLO11n (ours)** | **79.4%** | **51.2%** | **89** |

---

## Citation

If you use this work, please cite:

```bibtex
@article{lightpestformer2025,
  title   = {LightPestFormer: A Lightweight Multi-Scale Semantic Attention Network
             for Fine-Grained Insect Classification and Detection},
  author  = {Mageshkumar, N S and others},
  journal = {(Journal Name)},
  year    = {2025},
  note    = {Under Review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
