"""
Audio I/O utilities: load, save, convert and resample.
"""
import os
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa

def load_audio(path: str, sr: int = 32000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load an audio file and return waveform (float32) and sample rate.

    Tries soundfile first (preserves sample rate), falls back to librosa.
    """
    try:
        data, orig_sr = sf.read(path, always_2d=False)
        if data.ndim > 1 and mono:
            data = np.mean(data, axis=1)
        if orig_sr != sr:
            data = librosa.resample(data.astype(np.float32), orig_sr, sr)
        return data.astype(np.float32), sr
    except Exception:
        data, _ = librosa.load(path, sr=sr, mono=mono)
        return data.astype(np.float32), sr

def save_audio(path: str, audio: np.ndarray, sr: int = 32000, subtype: str = 'PCM_16') -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, sr, subtype=subtype)

def convert_mp3_to_wav(src: str, dst: str, sr: int = 32000) -> None:
    """Load src at sr and write a WAV file to dst."""
    audio, _ = load_audio(src, sr=sr, mono=True)
    save_audio(dst, audio, sr)

def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio.astype(np.float32), orig_sr, target_sr)