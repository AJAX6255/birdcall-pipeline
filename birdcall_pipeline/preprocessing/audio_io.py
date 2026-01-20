"""
Lightweight audio I/O utilities.
"""
import os
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa

def load_audio(path: str, sr: int = 32000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load an audio file and return waveform and sample rate.

    Uses soundfile for reading where possible, falls back to librosa.
    """
    try:
        audio, orig_sr = sf.read(path, always_2d=False)
        if audio.ndim > 1 and mono:
            audio = np.mean(audio, axis=1)
        if orig_sr != sr:
            audio = librosa.resample(audio.astype(float), orig_sr, sr)
        return audio.astype(float), sr
    except Exception:
        audio, orig_sr = librosa.load(path, sr=sr, mono=mono)
        return audio.astype(float), sr

def save_audio(path: str, audio: np.ndarray, sr: int = 32000) -> None:
    """Save waveform to file (WAV).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, sr)

def convert_mp3_to_wav(src: str, dst: str, sr: int = 32000) -> None:
    """Convert mp3 (or other readable formats) to WAV at target sample rate.

    This implementation uses librosa to load and soundfile to write.
    """
    audio, _ = load_audio(src, sr=sr, mono=True)
    save_audio(dst, audio, sr)

def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio.astype(float), orig_sr, target_sr)
