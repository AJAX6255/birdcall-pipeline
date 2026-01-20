# birdcall-pipeline
Quick experiment to verify the use of non-symetric spectrograms in classification of bird songs.

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
