"""Audio I/O, SNR mixing, and noisy-clean pair construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from backend.config import SAMPLE_RATE
from backend.synthesizer import mix_at_snr, resample_to


def load_wav(path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    if hasattr(path, "read"):
        data, file_sr = sf.read(path, always_2d=False)
    else:
        data, file_sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float64)
    if file_sr != sr:
        data = resample_to(data, file_sr, sr)
    peak = np.max(np.abs(data)) + 1e-12
    if peak > 1.0:
        data = data / peak
    return data, sr


def save_wav(path: str | Path, data: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(data, dtype=np.float64), -1.0, 1.0)
    sf.write(str(path), x, sr, subtype="PCM_16")


def ensure_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    reps = int(np.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n]


def mix_files(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = ensure_length(noise, len(clean))
    return mix_at_snr(clean, noise, snr_db)


SNR_GRID = (-5, 0, 5, 10, 15, 20)
