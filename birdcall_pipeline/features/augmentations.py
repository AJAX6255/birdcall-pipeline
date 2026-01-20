"""
Basic waveform and spectrogram augmentations.
"""
import numpy as np
import random

def add_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    rms = (audio ** 2).mean() ** 0.5
    snr = 10 ** (snr_db / 20.0)
    noise_rms = rms / (snr + 1e-9)
    noise = np.random.normal(0, noise_rms, size=audio.shape)
    return audio + noise

def random_gain(audio: np.ndarray, min_db: float = -6.0, max_db: float = 6.0) -> np.ndarray:
    db = random.uniform(min_db, max_db)
    factor = 10 ** (db / 20.0)
    return audio * factor

def time_shift(audio: np.ndarray, sr: int, shift_max_s: float = 0.5) -> np.ndarray:
    shift = int(random.uniform(-shift_max_s, shift_max_s) * sr)
    if shift == 0:
        return audio
    if shift > 0:
        return np.concatenate([audio[shift:], np.zeros(shift)])
    return np.concatenate([np.zeros(-shift), audio[:shift]])

def spec_augment(spec: np.ndarray, freq_mask_param: int = 10, time_mask_param: int = 10) -> np.ndarray:
    """Apply simple SpecAugment-style masks in-place (returns a copy)."""
    spec = spec.copy()
    num_mels, num_frames = spec.shape
    # freq mask
    f = np.random.randint(0, freq_mask_param + 1)
    f0 = np.random.randint(0, max(1, num_mels - f)) if num_mels - f > 0 else 0
    spec[f0:f0 + f, :] = 0
    # time mask
    t = np.random.randint(0, time_mask_param + 1)
    t0 = np.random.randint(0, max(1, num_frames - t)) if num_frames - t > 0 else 0
    spec[:, t0:t0 + t] = 0
    return spec
