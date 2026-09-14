"""Per-frame acoustic intelligence features.

REAL: all features are computed from the audio buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

from backend.config import HIGH_BAND, LOW_BAND, SAMPLE_RATE, SPEECH_BAND


@dataclass
class AcousticFeatures:
    rms: float
    rms_db: float
    zcr: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_flux: float
    kurtosis: float
    low_ratio: float
    speech_ratio: float
    high_ratio: float
    harmonicity: float
    flatness: float
    snr_est_db: float
    speech_prob: float
    vector: np.ndarray


def _band_energy(freqs: np.ndarray, mag: np.ndarray, band: tuple[float, float]) -> float:
    mask = (freqs >= band[0]) & (freqs < band[1])
    return float(np.sum(mag[mask] ** 2) + 1e-12)


def frame_features(
    frame: np.ndarray,
    prev_mag: np.ndarray | None,
    noise_rms: float,
    sr: int = SAMPLE_RATE,
) -> AcousticFeatures:
    x = np.asarray(frame, dtype=np.float64)
    if len(x) < 32:
        x = np.pad(x, (0, 32 - len(x)))
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    rms_db = 20.0 * np.log10(rms + 1e-12)
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))))

    n_fft = int(2 ** np.ceil(np.log2(len(x))))
    spec = np.fft.rfft(x * np.hanning(len(x)), n=n_fft)
    mag = np.abs(spec) + 1e-12
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mag_sum = float(np.sum(mag) + 1e-12)
    centroid = float(np.sum(freqs * mag) / mag_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / mag_sum))
    if prev_mag is None or len(prev_mag) != len(mag):
        flux = 0.0
    else:
        d = mag - prev_mag
        flux = float(np.sqrt(np.sum(np.clip(d, 0, None) ** 2)) / mag_sum)
    mu = np.mean(x)
    m4 = np.mean((x - mu) ** 4)
    m2 = np.mean((x - mu) ** 2) + 1e-12
    kurtosis = float(m4 / (m2 ** 2))

    low = _band_energy(freqs, mag, LOW_BAND)
    speech = _band_energy(freqs, mag, SPEECH_BAND)
    high = _band_energy(freqs, mag, HIGH_BAND)
    tot = low + speech + high + 1e-12
    low_r, speech_r, high_r = low / tot, speech / tot, high / tot

    # Harmonicity: peak of normalized autocorrelation in 70–300 Hz lag
    min_lag = max(1, int(sr / 300))
    max_lag = min(len(x) - 1, int(sr / 70))
    if max_lag > min_lag + 2:
        xc = np.correlate(x, x, mode="full")
        mid = len(x) - 1
        ac = xc[mid + min_lag : mid + max_lag]
        harmonicity = float(np.max(ac) / (xc[mid] + 1e-12))
        harmonicity = float(np.clip(harmonicity, 0.0, 1.0))
    else:
        harmonicity = 0.0

    geo = float(np.exp(np.mean(np.log(mag))))
    arith = float(np.mean(mag))
    flatness = float(geo / (arith + 1e-12))

    snr_est = 10.0 * np.log10((rms ** 2) / (noise_rms ** 2 + 1e-12))

    # Speech probability: energy in speech band + harmonicity + moderate ZCR
    speech_prob = (
        0.40 * float(np.clip(speech_r, 0, 1))
        + 0.30 * harmonicity
        + 0.15 * float(np.clip(1.0 - abs(zcr - 0.12) / 0.3, 0, 1))
        + 0.15 * float(np.clip((rms_db + 40) / 30.0, 0, 1))
    )
    speech_prob = float(np.clip(speech_prob, 0.0, 1.0))
    # suppress speech_prob during obvious transients
    if kurtosis > 12 and flux > 0.4:
        speech_prob *= 0.25

    vec = np.array(
        [
            rms_db / 40.0,
            zcr,
            centroid / 4000.0,
            bandwidth / 4000.0,
            flux,
            min(kurtosis / 20.0, 3.0),
            low_r,
            speech_r,
            high_r,
            harmonicity,
            flatness,
            np.clip(snr_est / 30.0, -1, 2),
        ],
        dtype=np.float64,
    )
    return AcousticFeatures(
        rms=rms,
        rms_db=rms_db,
        zcr=zcr,
        spectral_centroid=centroid,
        spectral_bandwidth=bandwidth,
        spectral_flux=flux,
        kurtosis=kurtosis,
        low_ratio=low_r,
        speech_ratio=speech_r,
        high_ratio=high_r,
        harmonicity=harmonicity,
        flatness=flatness,
        snr_est_db=float(snr_est),
        speech_prob=speech_prob,
        vector=vec,
    )


def spectrogram(x: np.ndarray, sr: int = SAMPLE_RATE, n_fft: int = 512, hop: int = 256):
    f, t, z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window="hann", boundary=None)
    return f, t, z
