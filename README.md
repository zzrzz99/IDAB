# IDAB: A frame-level heterogeneous forgery-oriented video deepfake detection

A **temporal misalignment attention**-based framework for video deepfake detection and multi-label source classification, supporting video-level and frame-level prediction, multi-source forgery recognition, and inter-frame inconsistency detection.

---

## Overview

This project implements **IDAB (Enhanced Temporal Misalignment Attention)** and its variants (e.g. **IDABTemporalSTTP**) for:

- **Video-level multi-label classification**: Identify forgery sources present in a video (e.g. DF, F2F, FS, FSh, NT, OR).
- **Frame-level prediction**: Per-frame forgery source prediction with visualization and fine-grained analysis.
- **Inter-frame inconsistency detection**: Detect whether the generation mechanism changes within the same video (dual-head task).

Pipeline: **Backbone → FrameDiff → TemporalAttn → TextureEnhance → BilinearPool → MAA → ML-FFormer**.

---

## Project structure

```
IDAB/
├── configs/
│   └── idab.yaml          # Model and training config (reproduction)
├── models/
│   ├── __init__.py        # Exports IDAB, IDABTemporalSTTP, etc.
│   ├── idab_v2.py         # IDAB main model (temporal misalignment + dual-head)
│   ├── idab_temporal_sttp.py  # IDABTemporalSTTP (frame attention + STTP pooling)
│   ├── backbone/          # EfficientNet and other backbones
│   ├── temporal/         # FrameDiff, TemporalAttention
│   ├── enhancement/      # Texture enhancement, bilinear pooling
│   ├── tokenization/     # MAA multi-attention aggregation
│   ├── transformer/     # ML-FFormer
│   ├── sttp_pooling.py   # Spatio-temporal Transformer pooling
│   ├── maa_module.py     # Token attention aggregator
│   └── ...
├── utils/
│   ├── dataset_loader.py  # Video dataset (label-dir / Phase1 format)
│   ├── transforms.py      # Train/val data augmentation
│   ├── trainer_core.py   # train_epoch, validate
│   ├── trainer.py        # Compatibility entry
│   ├── metrics.py         # Evaluation (Accuracy, F1, AUC, threshold tuning, etc.)
│   ├── dataset_pair_splitter.py   # Train/val/test split for mixed video pairs
│   ├── copy_single_forgery_split.py  # Copy single-forgery videos by split
│   └── ...
├── train_test_results.json  # Train/val history and test results (example)
├── best_model.pth         # Best checkpoint (if present)
└── README.md
```

---

## Requirements

- **Python** 3.8+
- **PyTorch** (with CUDA if using GPU)
- **torchvision**
- **scikit-learn** (metrics and threshold optimization)
- **scipy** (optional, for threshold optimization)
- **PIL / Pillow**

Edit `configs/idab.yaml` for data path, batch size, learning rate, epochs, etc.

---

## Dataset and data format

- The default config uses an **FHDeepfake**-style dataset; the data root is set in `data.root` in `configs/idab.yaml`.
- Two directory layouts are supported:
  1. **Label-based folders**: `root/split/LABEL/video_folder/*.jpg`
  2. **Phase1 (FH-FF++) format**: `root/split/video_xxxx/frames/*.jpg` with `video_label.json` in each video directory (and optional `frame_label.json` for frame-level / inconsistency tasks).

Videos are sampled to a fixed number of frames (e.g. 16); frame size is set to 256×256 in the config.

---

## Obtaining the FHF-DF dataset

**To use the FHF-DF dataset, please contact the authors:**

- **Email**: 20233001404@hainanu.edu.cn

---

## Training and evaluation

- **Training**: Use `train_epoch` and `validate` from `utils/trainer_core`. You need to write a script or notebook that loads `configs/idab.yaml`, builds a DataLoader with `utils.dataset_loader.load_data`, instantiates the model (e.g. `models.IDAB` or `models.IDABTemporalSTTP`), and calls these functions.
- **Evaluation**: `utils.metrics` provides:
  - `evaluate_with_frame_predictions`: Video-level and frame-level predictions, per-class threshold optimization (F1 / EACC).
  - `evaluate_dual_task`: Dual-head (frame-level source classification + inter-frame inconsistency + video-level source recall).
  - `evaluate_inconsistency`: Inconsistency binary classification only.

Enable `optimize_threshold` to optimize classification thresholds per class or jointly.

---

## Models

- **IDAB** (`models/idab_v2.py`): Temporal misalignment attention + texture enhancement + AGBP + MAA + ML-FFormer; optional dual-head (frame-level source + inter-frame inconsistency).
- **IDABTemporalSTTP** (`models/idab_temporal_sttp.py`): Frame-wise attention reweighting based on L2 features + STTP pooling + MAA + ML-FFormer, for spatio-temporal modeling.

Hyperparameters (number of classes, backbone, frames per video, learning rate, etc.) are in `configs/idab.yaml`.

---

## Citation and license

If you use this code or the FHF-DF dataset, please cite the work and comply with the relevant license. For dataset terms, please confirm with the authors (20233001404@hainanu.edu.cn).
