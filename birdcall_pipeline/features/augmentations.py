"""
Basic waveform and spectrogram augmentations.
"""
import numpy as np
import random
import librosa
from typing import Tuple

def add_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add Gaussian noise to reach a target SNR (dB)."""
    audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    snr = 10.0 ** (snr_db / 20.0)
    noise_rms = rms / (snr + 1e-12)
    noise = np.random.normal(0.0, noise_rms, size=audio.shape).astype(np.float32)
    return audio + noise

def random_gain(audio: np.ndarray, min_db: float = -6.0, max_db: float = 6.0) -> np.ndarray:
    db = float(random.uniform(min_db, max_db))
    factor = 10.0 ** (db / 20.0)
    return (audio * factor).astype(np.float32)

def time_shift(audio: np.ndarray, sr: int, shift_max_s: float = 0.5) -> np.ndarray:
    """Random circular time shift within +/- shift_max_s seconds."""
    max_shift = int(abs(shift_max_s) * sr)
    if max_shift == 0:
        return audio
    shift = random.randint(-max_shift, max_shift)
    if shift == 0:
        return audio
    return np.roll(audio, shift)

def pitch_shift(audio: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    """Pitch shift using librosa. n_steps can be positive or negative.

    Falls back to original audio if librosa fails.
    """
    try:
        return librosa.effects.pitch_shift(audio.astype(np.float32), sr, n_steps)
    except Exception:
        return audio

def spec_augment(spec: np.ndarray, freq_mask_param: int = 10, time_mask_param: int = 10, num_masks: int = 1) -> np.ndarray:
    """Apply SpecAugment-style frequency and time masking to a spectrogram.

    spec is expected shape (n_mels, n_frames).
    """
    spec = spec.copy()
    n_mels, n_frames = spec.shape
    for _ in range(num_masks):
        # Frequency mask
        f = random.randint(0, freq_mask_param)
        if f > 0:
            f0 = random.randint(0, max(0, n_mels - f))
            spec[f0:f0 + f, :] = 0
        # Time mask
        t = random.randint(0, time_mask_param)
        if t > 0:
            t0 = random.randint(0, max(0, n_frames - t))
            spec[:, t0:t0 + t] = 0
    return spec

def apply_random_augmentations(audio: np.ndarray, sr: int, p: float = 0.5) -> np.ndarray:
    """Apply a random chain of waveform augmentations with probability p for each."""
    out = audio.copy()
    if random.random() < p:
        out = random_gain(out)
    if random.random() < p:
        out = add_noise(out, snr_db=random.uniform(5, 30))
    if random.random() < p:
        out = time_shift(out, sr, shift_max_s=0.5)
    return out