"""
Spectrogram computation utilities (mel / log / normalization).
"""
import numpy as np
import librosa
from typing import Optional

def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    asymmetric: bool = True,
    power: float = 2.0,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """Compute a mel spectrogram from a waveform.

    Parameters
    - audio: 1D waveform (float32)
    - sr: sample rate
    - n_fft / hop_length: STFT params. asymmetric flag is informational but caller can set non-square params.
    - n_mels: number of mel bands
    - power: power for spectrogram (1.0 for amplitude, 2.0 for power)
    - fmin / fmax: mel band limits

    Returns
    - S: np.ndarray shape (n_mels, n_frames) (float32)
    """
    if audio is None or len(audio) == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
        fmin=fmin,
        fmax=fmax,
    )
    return S.astype(np.float32)

def compute_log_spectrogram(audio: np.ndarray, sr: int = 32000, amin: float = 1e-10, **kwargs) -> np.ndarray:
    """Return log-scaled (dB) mel spectrogram.

    Additional STFT/mel parameters are forwarded to compute_mel_spectrogram.
    """
    S = compute_mel_spectrogram(audio, sr=sr, **kwargs)
    if S.size == 0:
        return S
    S_db = librosa.power_to_db(S, ref=np.max, amin=amin)
    return S_db.astype(np.float32)

def normalize_spectrogram(spec: np.ndarray, method: str = 'per_sample', eps: float = 1e-6) -> np.ndarray:
    """Normalize spectrogram.

    Methods:
      - per_sample: zero-mean unit-variance per sample
      - global: scale by global max (placeholder)
    """
    if spec.size == 0:
        return spec
    if method == 'per_sample':
        mean = float(spec.mean())
        std = float(spec.std())
        return ((spec - mean) / (std + eps)).astype(np.float32)
    elif method == 'global':
        return (spec / (np.max(np.abs(spec)) + eps)).astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def to_tensor(spec: np.ndarray):
    """Convert numpy spectrogram (n_mels, n_frames) to a torch-friendly shape (1, n_mels, n_frames).

    This function returns the numpy array cast to float32 with a leading channel dim.
    """
    if spec is None:
        return None
    spec = np.asarray(spec, dtype=np.float32)
    if spec.ndim == 2:
        return spec[np.newaxis, :, :]
    if spec.ndim == 3:
        return spec
    raise ValueError('spec must be 2D or 3D')
