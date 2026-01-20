"""
A minimal PyTorch Dataset wrapper that loads clips and computes spectrograms on the fly.
"""
from typing import Optional, List
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa

class BirdcallDataset(Dataset):
    def __init__(self, clips_dir: str, annotations: Optional[dict] = None, sr: int = 32000, duration: float = 5.0, transform=None):
        self.clips_dir = clips_dir
        self.files = sorted([os.path.join(clips_dir, f) for f in os.listdir(clips_dir) if f.lower().endswith('.wav')])
        self.annotations = annotations or {}
        self.sr = sr
        self.duration = duration
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        audio, _ = librosa.load(path, sr=self.sr, mono=True)
        # pad/trim
        target_len = int(self.sr * self.duration)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        sample = {'audio': audio, 'path': path}
        label = None
        basename = os.path.basename(path)
        if basename in self.annotations:
            label = self.annotations[basename]
        if self.transform:
            sample = self.transform(sample)
        if label is None:
            label = -1
        return sample, int(label)
