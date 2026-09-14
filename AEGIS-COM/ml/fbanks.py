"""32-band triangular filterbank on a 512-point rFFT (257 bins)."""

from __future__ import annotations

import numpy as np

from backend.config import N_FFT, SAMPLE_RATE

N_BANDS = 32
N_BINS = N_FFT // 2 + 1  # 257


def band_centers_hz(sr: int = SAMPLE_RATE, n_bands: int = N_BANDS) -> np.ndarray:
    return np.linspace(80.0, sr / 2 - 80.0, n_bands)


def filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_bands: int = N_BANDS) -> np.ndarray:
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    centers = band_centers_hz(sr, n_bands)
    edges = np.concatenate([[0.0], centers, [sr / 2]])
    fb = np.zeros((n_bands, len(freqs)), dtype=np.float32)
    for i in range(n_bands):
        left, c, right = edges[i], centers[i], edges[i + 2]
        up = (freqs >= left) & (freqs <= c)
        down = (freqs > c) & (freqs <= right)
        if c > left:
            fb[i, up] = (freqs[up] - left) / (c - left)
        if right > c:
            fb[i, down] = (right - freqs[down]) / (right - c)
        s = float(fb[i].sum())
        if s > 0:
            fb[i] /= s
    return fb


def to_bands(mag: np.ndarray, fb: np.ndarray) -> np.ndarray:
    mag = np.nan_to_num(np.asarray(mag, dtype=np.float64), nan=0.0, posinf=1e6, neginf=0.0)
    mag = np.clip(mag, 0.0, 1e6)
    # Input is already sanitised; some BLAS builds still raise stale FP flags here.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        bands = mag @ np.asarray(fb, dtype=np.float64).T
    return np.nan_to_num(bands, nan=0.0, posinf=1e6, neginf=0.0).astype(np.float32)


def upsample_gain(gain_bands: np.ndarray, n_bins: int = N_BINS, sr: int = SAMPLE_RATE) -> np.ndarray:
    centers = band_centers_hz(sr, len(gain_bands))
    freqs = np.fft.rfftfreq((n_bins - 1) * 2, 1.0 / sr)
    g = np.interp(freqs, centers, gain_bands)
    return np.clip(g, 0.05, 1.0).astype(np.float64)
