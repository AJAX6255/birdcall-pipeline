"""
Windowing and simple activity detection utilities.
"""
from typing import List, Tuple
import numpy as np

def sliding_windows(audio: np.ndarray, sr: int, window_s: float, hop_s: float) -> List[np.ndarray]:
    """Yield overlapping windows from a waveform.

    Returns list of 1D numpy arrays each of length int(window_s * sr).
    """
    window_len = int(window_s * sr)
    hop_len = int(hop_s * sr)
    if window_len <= 0 or hop_len <= 0:
        return []
    windows = []
    for start in range(0, max(1, len(audio) - window_len + 1), hop_len):
        windows.append(audio[start:start + window_len])
    return windows

def detect_activity(audio: np.ndarray, sr: int, frame_s: float = 0.02, threshold: float = 1e-4) -> List[Tuple[float, float]]:
    """A naive energy-based activity detector.

    Returns a list of (start_s, end_s) segments where frames exceed threshold.
    """
    frame_len = int(frame_s * sr)
    if frame_len <= 0:
        return []
    hop = frame_len
    energy = []
    for i in range(0, len(audio), hop):
        frame = audio[i:i + frame_len]
        if frame.size == 0:
            energy.append(0.0)
        else:
            energy.append(float((frame ** 2).mean()))
    active = [e > threshold for e in energy]
    segments = []
    start_idx = None
    for i, a in enumerate(active):
        if a and start_idx is None:
            start_idx = i
        if not a and start_idx is not None:
            segments.append((start_idx * frame_s, i * frame_s))
            start_idx = None
    if start_idx is not None:
        segments.append((start_idx * frame_s, len(audio) / sr))
    return segments
