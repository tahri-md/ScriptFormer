# ScriptFormer: Arabic Handwritten Text Recognition

A deep learning-based Optical Character Recognition (OCR) system for recognizing Arabic handwritten text in manuscript images. ScriptFormer uses a CNN Encoder + Transformer Decoder architecture to accurately transcribe historical and handwritten Arabic documents.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Inference & Prediction](#inference--prediction)
  - [Evaluation](#evaluation)
- [Data Format](#data-format)
- [Known Issues & Limitations](#known-issues--limitations)
- [Project Details](#project-details)

## 📖 Overview

**ScriptFormer** is an end-to-end OCR pipeline designed specifically for Arabic manuscripts. It combines:

- **Preprocessing**: Image binarization, denoising, and augmentation for manuscript data
- **Encoding**: CNN-based visual feature extraction from manuscript images
- **Decoding**: Transformer-based autoregressive text generation
- **Postprocessing**: Arabic text normalization and diacritic handling

The system is trained on the KHATT dataset (Arabic handwritten text corpus) and uses character-level tokenization for Arabic script recognition.

## 🏗️ Architecture

### Model Components

```
Input Image (1 × 64 × 384)
    ↓
[CNN Encoder]
  - 4 Convolutional blocks with batch normalization
  - Progressive feature extraction: 1→32→64→128→256 channels
  - MaxPooling for dimensionality reduction
  - Output: Sequence of visual feature vectors (256-dim)
    ↓
[Projection & Normalization]
  - Linear projection to hidden_size (256-dim)
  - LayerNorm for training stability
    ↓
[Transformer Decoder]
  - 6 transformer decoder layers with 8 attention heads
  - Causal masking for autoregressive generation
  - Token embeddings + positional encoding
  - Outputs logits for vocabulary prediction
    ↓
Output Text (arbitrary length, up to 128 characters)
```

### Key Design Decisions

- **Image Dimensions**: 384×64 pixels captures full text lines while keeping computation efficient
- **Grayscale**: Single channel (1-channel) reduces model size without losing important information
- **CNN Encoder**: Custom CNN (not Vision Transformer) for efficiency on smaller manuscript images
- **Transformer Decoder**: Standard PyTorch TransformerDecoder for text generation
- **Character-level Tokenization**: Each Arabic character treated as a token (simpler than subword tokenization)

## 📁 Project Structure

```
scriptformer/
├── README.md                  # This file
├── configs/
│   └── default.yml           # All hyperparameters and configuration
├── data/
│   ├── __init__.py
│   ├── dataset.py            # PyTorch Dataset classes
│   ├── labelparser.py        # Parse KHATT dataset annotations
│   └── tokenizer.py          # Arabic character tokenizer
├── preprocessing/
│   ├── __init__.py
│   └── transforms.py         # Image preprocessing & augmentation
├── model/
│   ├── __init__.py
│   └── trocr.py              # ScriptFormer model definitions
├── inference/
│   ├── __init__.py
│   └── pipeline.py           # End-to-end prediction pipeline
├── postprocessing/
│   ├── __init__.py
│   └── arabic_text.py        # Arabic text normalization
├── training/
│   ├── __init__.py
│   └── trainer.py            # Training loop and checkpointing
├── evaluation/
│   ├── __init__.py
│   └── metrics.py            # CER, WER, and other metrics
└── scripts/
    ├── train.py              # Training entry point
    ├── predict.py            # Inference entry point
    └── evaluate.py           # Evaluation entry point
```

## 🚀 Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- torchvision
- OpenCV (cv2)
- PyYAML
- NumPy

### Setup

```bash
# Clone or navigate to the project
cd scriptformer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install opencv-python pyyaml numpy
```

## ⚡ Quick Start

### 1. Prepare Your Data

Place your KHATT dataset in the following structure:
```
data/raw/KHATT/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── val/
├── test/
└── annotations.csv  # Format: image_path,text
```

### 2. Train the Model

```bash
python scripts/train.py --config configs/default.yml --epochs 100
```

### 3. Make Predictions

```bash
# Single image
python scripts/predict.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pt

# Batch directory
python scripts/predict.py --dir path/to/images/ --checkpoint checkpoints/best_model.pt
```

### 4. Evaluate Performance

```bash
python scripts/evaluate.py --config configs/default.yml --checkpoint checkpoints/best_model.pt
```

## ⚙️ Configuration

All hyperparameters are defined in `configs/default.yml`. Key sections:

### Project Settings
```yaml
project:
  name: "scriptformer"
  seed: 42                # Reproducibility
  device: "cpu"           # "cpu" or "cuda"
```

### Data Configuration
```yaml
data:
  raw_dir: "data/raw"
  train_ratio: 0.8
  val_ratio: 0.1
  image:
    height: 64
    width: 384
    channels: 1           # Grayscale
```

### Preprocessing Options
```yaml
preprocessing:
  binarization:
    method: "sauvola"     # Adaptive thresholding (better for manuscripts)
    window_size: 25
  augmentation:
    rotation_range: 3
    elastic_distortion: true
    brightness_range: 0.2
```

### Model Architecture
```yaml
model:
  encoder:
    hidden_size: 256
  decoder:
    hidden_size: 256
    num_layers: 6
    num_heads: 8
    max_length: 128
```

### Training Hyperparameters
```yaml
training:
  epochs: 50              # Change via CLI: --epochs 100
  batch_size: 16          # Reduce if out of memory
  learning_rate: 0.0001
  warmup_steps: 500       # LR warmup for stability
  early_stopping:
    enabled: true
    patience: 5           # Stop if no improvement for 5 epochs
```

## 📚 Usage

### Training

**Basic training** with default config:
```bash
python scripts/train.py
```

**Override specific parameters**:
```bash
python scripts/train.py \
  --config configs/default.yml \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.0001
```

**Resume from checkpoint**:
```bash
python scripts/train.py --resume checkpoints/epoch_25.pt
```

**Output**: Checkpoints saved to `checkpoints/` directory with losses logged to `logs/`

### Inference & Prediction

**Single image prediction**:
```bash
python scripts/predict.py \
  --image path/to/text_line.jpg \
  --checkpoint checkpoints/best_model.pt
```

**Batch prediction** (entire directory):
```bash
python scripts/predict.py \
  --dir path/to/test_images/ \
  --checkpoint checkpoints/best_model.pt
```

Output: `predictions.csv` with filename and predicted text

**Advanced options**:
```bash
python scripts/predict.py \
  --image image.jpg \
  --checkpoint checkpoints/best_model.pt \
  --max-length 128 \
  --normalize-alef \
  --remove-diacritics
```

### Evaluation

**Calculate metrics** on test set:
```bash
python scripts/evaluate.py \
  --config configs/default.yml \
  --checkpoint checkpoints/best_model.pt
```

**Metrics computed**:
- Character Error Rate (CER): % of characters incorrectly recognized
- Word Error Rate (WER): % of words incorrectly recognized
- Sequence Error Rate (SER): % of completely wrong sequences

## 📦 Data Format

### KHATT Dataset

Expected structure:
```
data/raw/KHATT/
├── train/          # Training images
├── val/            # Validation images  
├── test/           # Test images
└── annotations.csv
```

**Annotations CSV format**:
```csv
image_path,text
train/page001_line01.jpg,السلام عليكم
train/page001_line02.jpg,ورحمة الله
...
```

### Image Requirements
- **Format**: JPG, PNG, or grayscale images
- **Size**: Flexible (preprocessor resizes to 384×64)
- **Content**: Handwritten Arabic text lines
- **Quality**: Clear, high-contrast for binarization to work well


## 📊 Project Details

### What Was Implemented

✅ **Data Pipeline**
- KHATT dataset parser
- Image preprocessing (binarization, denoising, augmentation)
- Arabic character tokenizer with special tokens
- PyTorch Dataset and DataLoader integration

✅ **Model**
- CNN Encoder (4-layer convolutional feature extractor)
- Transformer Decoder (6-layer, 8-head, autoregressive)
- Positional encoding and causal masking
- Full forward/backward pass pipeline

✅ **Training**
- AdamW optimizer with weight decay
- Learning rate warmup and cosine annealing scheduler
- Early stopping with patience
- Gradient clipping for stability
- Checkpoint saving and resuming

✅ **Inference**
- End-to-end OCR pipeline
- Batch and single-image prediction
- Arabic text postprocessing and normalization

✅ **Evaluation**
- Character Error Rate (CER)
- Word Error Rate (WER)
- Sequence Error Rate (SER)

### Architecture Rationale

- **CNN instead of ViT**: Handwritten documents need local feature extraction; CNNs are more data-efficient than Vision Transformers
- **Character-level tokenization**: Arabic morphology is complex; character-level avoids OOV issues
- **Transformer Decoder**: Proven effective for sequence generation; standard PyTorch implementation for maintainability
- **Adaptive binarization (Sauvola)**: Handles varying ink intensity and page aging in historical manuscripts
