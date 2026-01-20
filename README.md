# Birdcall Pipeline  
**Event-centered bird call classification with modern audio ML best practices**

This repository contains a **research-grade, end-to-end pipeline** for bird call classification built around:

- human-annotated acoustic events
- event-centered fixed-length audio clips
- rectangular (time-rich) log-mel spectrograms
- leakage-safe grouped evaluation
- two complementary model paths:
  - pretrained audio embeddings (baseline)
  - CNNs with SpecAugment (advanced)

The pipeline is designed for **scientific validation first**, not premature deployment.

---

## Why this pipeline exists

Many audio ML examples:
- chop audio arbitrarily
- leak data across train/validation
- force square spectrograms
- overfit small datasets
- optimize demos instead of signal quality

This project was explicitly designed to avoid those pitfalls.

---

## Core design principles

- **Event-centered clips**  
  Audio clips are centered on annotated bird calls (not random windows).

- **Reflect padding (not silence)**  
  Prevents non-physical edge artifacts in short clips.

- **Rectangular spectrograms**  
  Time is preserved; frequency is not artificially emphasized.

- **Grouped evaluation**  
  Clips from the same recording never appear in both train and validation.

- **Embeddings first, CNN second**  
  Validate the dataset before tuning architectures.

---

## Pipeline overview

Spectrolipi annotations (JSON)
↓
Event-centered clip generation (WAV)
↓
metadata.csv (single source of truth)
↓
Feature extraction
├── Pretrained embeddings (baseline)
└── Log-mel spectrograms (CNN)
↓
Grouped training & evaluation
↓
File-level performance metrics

yaml
Copy code

---

## Repository structure

birdcall_pipeline/
├── data/
│   ├── raw_mp3/
│   ├── annotations_json/
│   ├── wav_full/
│   └── clips/
├── preprocessing/
│   ├── audio_io.py
│   ├── windowing.py
│   └── padding.py
├── features/
│   ├── spectrograms.py
│   └── augmentations.py
├── models/
│   ├── cnn.py
│   └── embeddings.py
├── training/
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
└── notebook.ipynb

yaml
Copy code

> **Note:** Audio data is intentionally *not* included in the repository.

---

## Data expectations

### Audio
- Input format: MP3 or WAV
- Sample rate: resampled to 22,050 Hz
- Mono

### Annotations
- Produced externally (e.g. **Spectrolipi**)
- One JSON file per audio file
- Each annotation must include:
  - `start_time` (seconds)
  - `end_time` (seconds)
  - `label` (species or call type)

---

## Canonical metadata

All downstream processing is driven by `metadata.csv`:

| column        | description |
|--------------|-------------|
| clip_id      | unique clip identifier |
| source_file  | original recording |
| label        | species / call type |
| start_sec    | annotation start |
| end_sec      | annotation end |
| clip_start   | generated clip start |
| clip_end     | generated clip end |
| sr           | sample rate |
| clip_path    | path to WAV clip |

This table is the **single source of truth**.

---

## Recommended workflow (Colab)

### 1. Clone the repo
```python
!git clone https://github.com/AJAX6255/birdcall-pipeline.git
%cd birdcall-pipeline
!pip -q install -r requirements.txt
2. Mount Google Drive
python
Copy code
from google.colab import drive
drive.mount('/content/drive')
Expected Drive layout:

bash
Copy code
MyDrive/birdcall_pipeline/data/
├── raw_mp3/
├── annotations_json/
└── clips/          # generated
3. Build dataset
python
Copy code
from preprocessing.windowing import build_dataset
from pathlib import Path

build_dataset(
    raw_audio_dir=Path('/content/drive/MyDrive/birdcall_pipeline/data/raw_mp3'),
    annotation_dir=Path('/content/drive/MyDrive/birdcall_pipeline/data/annotations_json'),
    clips_dir=Path('/content/drive/MyDrive/birdcall_pipeline/data/clips'),
    output_metadata_csv=Path('/content/drive/MyDrive/birdcall_pipeline/data/metadata.csv'),
    clip_duration=3.0,
    target_sr=22050
)
Model paths
1️⃣ Embeddings baseline (recommended first)
Uses pretrained audio embeddings

Trains a lightweight classifier

Fast, interpretable, low-variance

Purpose:

validate labels

detect annotation issues

establish a sanity baseline

2️⃣ CNN on spectrograms
Rectangular log-mel inputs

Reflect padding

Frequency-first pooling

SpecAugment regularization

Purpose:

exploit temporal and harmonic structure

improve performance once data quality is confirmed

Evaluation
Two levels of evaluation are reported:

Window-level (individual clips)

File-level (aggregated predictions per recording)

File-level metrics are emphasized, as they reflect real-world usage.

What this repo is not
❌ A turnkey production service

❌ A cloud-first demo

❌ A UI-centric project

❌ Optimized for huge datasets

This repository is about doing small-to-medium audio ML correctly.

When to extend this pipeline
Only after validation should you consider:

attention mechanisms

transformers

longer clips

deployment (Cloud Run, AI Studio, etc.)

License
Specify as appropriate (e.g. MIT, Apache-2.0).

Acknowledgements
Bioacoustics annotation workflows inspired by Spectrolipi

Audio embedding concepts from YAMNet-style models

Lessons learned from UrbanSound-style benchmarks (and their limitations)

Status
🟢 Actively validated in Google Colab
🟡 CNN performance under evaluation
🔵 Deployment intentionally deferred

If you are interested in contributing, discussing methodology, or adapting this pipeline to other bioacoustic tasks, please open an issue.
