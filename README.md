# CREMA-D Speech Emotion Recognition

A deep learning system for **speech emotion recognition** using the **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**. The final system uses a pretrained **WavLM** speech representation model with partial fine-tuning, a **Bidirectional GRU (BiGRU)** sequence layer, **multi-head attention**, attentive pooling, and a classification head to predict six emotions from speech audio.

## Overview

The system takes a `.wav` speech recording as input and predicts one of six emotions:

- Angry
- Disgust
- Fearful
- Happy
- Neutral
- Sad

This is an **audio-only** emotion classification project. Emotion intensity/level encoded in CREMA-D filenames is **not used as the prediction target**.

## Dataset

**Dataset:** CREMA-D — Crowd-sourced Emotional Multimodal Actors Dataset

The dataset used contains:

- **7,442 WAV audio clips**
- **91 actors**
- **12 sentence IDs**
- **6 emotion classes**

### Class Distribution

| Emotion | Samples |
|---|---:|
| Angry | 1,271 |
| Disgust | 1,271 |
| Fearful | 1,271 |
| Happy | 1,271 |
| Neutral | 1,087 |
| Sad | 1,271 |
| **Total** | **7,442** |

Expected dataset location:

```text
archive/
└── AudioWAV/
    └── *.wav
```

Metadata:

```text
data/cremad_metadata.csv
```

## Preprocessing

1. Load WAV audio with `librosa`.
2. Resample to **16 kHz**.
3. Convert to **mono**.
4. Convert the waveform to `float32`.
5. Pass the waveform to WavLM.
6. Process the learned speech representation with the BiGRU and attention layers.

`librosa` is used for audio loading instead of `torchaudio.load()`.

## Train / Validation / Test Split

| Split | Samples |
|---|---:|
| Training | 4,750 |
| Validation | 1,223 |
| Test | 1,469 |
| **Total** | **7,442** |

The held-out evaluation split was prepared to avoid speaker leakage.

## Final Model

**WavLM + Partial Fine-Tuning + BiGRU + Multi-Head Attention + Attentive Pooling**

```text
Input Speech Waveform
        │
        ▼
   WavLM Backbone
 (Pretrained Speech Model)
        │
        ▼
 Partial Fine-Tuning
 (4 WavLM layers unfrozen)
        │
        ▼
     BiGRU Layer
        │
        ▼
 Multi-Head Attention
        │
        ▼
 Attentive Pooling
        │
        ▼
 Classification Head
        │
        ▼
  6 Emotion Classes
```

### Components

**WavLM:** pretrained speech representations provide contextual acoustic features.

**Partial Fine-Tuning:** four WavLM layers are unfrozen so the pretrained representation can adapt to emotion recognition.

**BiGRU:** models temporal dependencies in both forward and backward directions.

**Multi-Head Attention:** learns to focus on informative parts of the speech sequence.

**Attentive Pooling:** converts the temporal sequence into a compact representation for classification.

**Classification Head:** maps the pooled representation to the six emotion classes.

## Algorithms / Techniques Used

- Transfer learning
- Partial fine-tuning
- Bidirectional GRU (BiGRU)
- Multi-head attention
- Attentive pooling
- Gradient accumulation
- CUDA mixed-precision training
- Validation-based checkpoint selection using Macro F1

## Training Configuration

| Parameter | Value |
|---|---:|
| Batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Maximum epochs | 12 |
| Initial WavLM learning rate | `1e-5` |
| Initial head learning rate | `2e-4` |
| Best checkpoint metric | Validation Macro F1 |
| Best epoch | 10 |
| Best validation Macro F1 | 0.6606 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM | 6 GB |

Effective batch size:

```text
2 × 4 = 8
```

## Results

Final performance on the held-out test set:

| Metric | Score |
|---|---:|
| Test Accuracy | **64.13%** |
| Test Macro F1 | **0.6290** |

### Per-Class Performance

| Emotion | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| Angry | 0.6356 | 0.9243 | **0.7532** | 251 |
| Disgust | 0.7812 | 0.3984 | 0.5277 | 251 |
| Fearful | 0.7222 | 0.5179 | 0.6032 | 251 |
| Happy | 0.6345 | 0.4980 | 0.5580 | 251 |
| Neutral | 0.5460 | 0.9159 | 0.6841 | 214 |
| Sad | 0.6625 | 0.6335 | 0.6477 | 251 |

### Confusion Matrix

Rows are actual classes and columns are predicted classes.

Class order:

```text
Angry, Disgust, Fearful, Happy, Neutral, Sad
```

```text
              Predicted
              A    D    F    H    N    S
Actual  A    232    1    0    9    7    2
        D     41  100   20   23   33   34
        F     22    6  130   32   27   34
        H     54    5   14  125   49    4
        N      7    1    1    2  196    7
        S      9   15   15    6   47  159
```

## Model Parameters

| Parameter group | Count |
|---|---:|
| Total parameters | 99,607,287 |
| Trainable parameters | 33,580,503 |
| Trainable WavLM parameters | 28,355,152 |
| Trainable head parameters | 5,225,351 |

## Project Structure

```text
MLv2/
├── archive/
│   └── AudioWAV/
│       └── *.wav
├── data/
│   └── cremad_metadata.csv
├── models/
│   └── cremad_wavlm_bigru_attention.pt
├── results/
│   └── cremad_wavlm_bigru_attention_results.txt
├── trained_finetune_lm.py
└── README.md
```

## Requirements

Recommended environment:

- Python 3.10+
- PyTorch with CUDA support
- Transformers
- librosa
- NumPy
- pandas
- scikit-learn
- soundfile

Install packages:

```powershell
pip install torch torchvision torchaudio
pip install transformers librosa numpy pandas scikit-learn soundfile
```

> The project uses `librosa` for WAV loading. If `torchaudio` causes TorchCodec/FFmpeg loading issues on Windows, use the librosa-based loader used by this project.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Place CREMA-D audio files in:

```text
archive/AudioWAV/
```

Make sure metadata exists at:

```text
data/cremad_metadata.csv
```

## Training

Run the final model training script:

```powershell
python trained_finetune_lm.py
```

The best checkpoint is selected using validation Macro F1.

The trained model is saved as:

```text
models/cremad_wavlm_bigru_attention.pt
```

## Results Output

Evaluation results are saved to:

```text
results/cremad_wavlm_bigru_attention_results.txt
```

## Hardware

Training and evaluation were performed using:

```text
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
VRAM: 6 GB
```

CUDA-enabled PyTorch was used for training and evaluation.

## Reproducibility

Keep the following consistent when reproducing the experiment:

- CREMA-D dataset content
- train/validation/test split
- 16 kHz sampling rate
- WavLM configuration
- four unfrozen WavLM layers
- BiGRU and attention architecture
- optimizer and learning rates
- gradient accumulation
- validation Macro F1 checkpoint selection

## Limitations

The reported performance is for the held-out CREMA-D test set. Results may differ on other datasets, speakers, microphones, recording environments, or real-world audio.

## Dataset Reference

CREMA-D project repository:

https://github.com/CheyneyComputerScience/CREMA-D

## Author

Developed as a speech emotion recognition / deep learning project using CREMA-D and pretrained WavLM speech representations.
