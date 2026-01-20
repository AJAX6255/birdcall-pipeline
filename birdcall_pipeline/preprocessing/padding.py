"""
Padding and trimming utilities.
"""
import numpy as np

def pad_or_trim(audio: np.ndarray, target_length: int, mode: str = 'constant') -> np.ndarray:
    """Pad or trim a 1D numpy array to target length.

    mode: 'constant' (zero pad) or 'reflect'
    """
    if target_length <= 0:
        raise ValueError('target_length must be > 0')
    audio = np.asarray(audio)
    cur = audio.shape[0]
    if cur == target_length:
        return audio.astype(np.float32)
    if cur > target_length:
        return audio[:target_length].astype(np.float32)
    pad_total = target_length - cur
    if mode == 'reflect' and cur > 0:
        left = pad_total // 2
        right = pad_total - left
        left_pad = np.flip(audio[:left]) if left > 0 else np.zeros(0)
        right_pad = np.flip(audio[-right:]) if right > 0 else np.zeros(0)
        out = np.concatenate([left_pad, audio, right_pad])
        return out.astype(np.float32)
    # default constant (zeros)
    return np.concatenate([audio, np.zeros(pad_total, dtype=audio.dtype)]).astype(np.float32)

def ensure_length(audio: np.ndarray, sr: int, target_seconds: float, mode: str = 'constant') -> np.ndarray:
    """Ensure audio has length of target_seconds (in seconds) at sample rate sr."""
    target_len = int(round(target_seconds * int(sr)))
    return pad_or_trim(audio, target_len, mode=mode)