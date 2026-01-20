"""
Padding and trimming utilities.
"""
import numpy as np

def pad_or_trim(audio: np.ndarray, target_length: int, mode: str = 'constant') -> np.ndarray:
    """Pad or trim a 1D numpy array to target length.

    mode: one of 'constant', 'reflect'
    """
    if len(audio) == target_length:
        return audio
    if len(audio) > target_length:
        return audio[:target_length]
    pad_total = target_length - len(audio)
    if mode == 'reflect':
        left = pad_total // 2
        right = pad_total - left
        left_pad = np.flip(audio[:left]) if left > 0 and len(audio) > 0 else np.zeros(left)
        right_pad = np.flip(audio[-right:]) if right > 0 and len(audio) > 0 else np.zeros(right)
        return np.concatenate([left_pad, audio, right_pad])
    else:
        return np.concatenate([audio, np.zeros(pad_total)])

def ensure_length(audio: np.ndarray, sr: int, target_seconds: float, mode: str = 'constant') -> np.ndarray:
    target_len = int(target_seconds * sr)
    return pad_or_trim(audio, target_len, mode=mode)