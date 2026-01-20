"""
Spectrogram computation utilities (mel / log / normalization).
"""
import numpy as np
import librosa

def compute_mel_spectrogram(audio: np.ndarray, sr: int = 32000, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 128, asymmetric: bool = True, power: float = 2.0) -> np.ndarray:
    """Compute a mel spectrogram. If asymmetric=True, use non-square STFT parameters as an experiment flag."""
    # asymmetric parameter can be used by the caller to choose different n_fft/hop combos
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    return S

def compute_log_spectrogram(audio: np.ndarray, sr: int = 32000, **kwargs) -> np.ndarray:
    S = compute_mel_spectrogram(audio, sr=sr, **kwargs)
    return librosa.power_to_db(S, ref=np.max)

def normalize_spectrogram(spec: np.ndarray, method: str = 'per_sample') -> np.ndarray:
    """Simple normalization options.

    method: 'per_sample', 'global' (placeholder)
    """
    if method == 'per_sample':
        mean = spec.mean()
        std = spec.std() + 1e-6
        return (spec - mean) / std
    return spec
