"""SIMULATED talkers for training the tiny speech-isolation net.

Richer than `backend.synthesizer.synth_speech`, because the net has to learn
*where speech is*, not just how to flatten noise. That needs:

  - varied speakers        f0 85-210 Hz, vocal-tract scaling, spectral tilt
  - phoneme structure      voiced / fricative / plosive segments
  - real pauses            silence between words, so VAD has negatives
  - varied levels          the operator is not always at the same distance

This is still SIMULATED speech, not recorded talkers. A model trained only on
this will not match one trained on LibriSpeech-class data; it is sized for a
Raspberry Pi and for a POC we can regenerate offline in seconds.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter

from backend.config import SAMPLE_RATE
from backend.synthesizer import _butter_filter, normalize

# Rough phoneme inventory: (F1, F2, F3) in Hz for a neutral vocal tract.
VOWELS = (
    (700.0, 1220.0, 2450.0),   # /a/
    (530.0, 1840.0, 2480.0),   # /e/
    (390.0, 2300.0, 3010.0),   # /i/
    (570.0, 840.0, 2410.0),    # /o/
    (440.0, 1020.0, 2240.0),   # /u/
    (620.0, 1660.0, 2430.0),   # schwa-ish
)
FRICATIVES = ((4200.0, 0.55), (2800.0, 0.45), (6000.0, 0.35))  # (centre, level)


def _pulse_train(f0: np.ndarray, sr: int) -> np.ndarray:
    phase = np.cumsum(f0 / sr)
    marks = np.diff(np.floor(phase), prepend=0.0) > 0
    src = np.zeros(len(f0))
    src[marks] = 1.0
    return src


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def _formant_filter(x: np.ndarray, f: np.ndarray, bw: float, sr: int, block: int = 256) -> np.ndarray:
    y = np.zeros_like(x)
    r = np.exp(-np.pi * bw / sr)
    for start in range(0, len(x), block):
        sl = slice(start, min(start + block, len(x)))
        fc = float(np.mean(f[sl]))
        w = 2 * np.pi * fc / sr
        y[sl] = lfilter([1.0], [1.0, -2 * r * np.cos(w), r * r], x[sl])
    return y


def synth_talker(
    duration_s: float,
    sr: int = SAMPLE_RATE,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (audio, speech_mask). `speech_mask` is 1 where a talker is active."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    t = np.arange(n) / sr

    # ---- speaker identity
    f0_base = float(rng.uniform(85.0, 210.0))
    vt_scale = float(rng.uniform(0.85, 1.20))     # vocal tract length
    tilt = float(rng.uniform(-0.35, 0.15))        # spectral tilt
    top_hz = float(rng.uniform(3400.0, 6500.0))   # channel bandwidth

    # ---- segment timeline: word (phonemes) then pause
    seg_type = np.zeros(n, dtype=np.int8)  # 0 silence, 1 voiced, 2 fricative, 3 plosive
    f1 = np.full(n, 600.0)
    f2 = np.full(n, 1400.0)
    f3 = np.full(n, 2500.0)
    fric_c = np.full(n, 4000.0)
    fric_l = np.zeros(n)
    amp = np.zeros(n)

    i = 0
    # start mid-pause sometimes, so clips do not all begin with speech
    if rng.random() < 0.35:
        i += int(rng.uniform(0.05, 0.45) * sr)
    while i < n:
        word_len = int(rng.uniform(0.25, 0.85) * sr)
        word_end = min(n, i + word_len)
        word_gain = float(rng.uniform(0.6, 1.0))
        j = i
        while j < word_end:
            ph = int(rng.uniform(0.05, 0.14) * sr)
            k = min(word_end, j + ph)
            roll = rng.random()
            if roll < 0.68:
                v = VOWELS[int(rng.integers(0, len(VOWELS)))]
                seg_type[j:k] = 1
                f1[j:k], f2[j:k], f3[j:k] = (v[0] * vt_scale, v[1] * vt_scale, v[2] * vt_scale)
                amp[j:k] = word_gain * float(rng.uniform(0.7, 1.0))
            elif roll < 0.9:
                c, lev = FRICATIVES[int(rng.integers(0, len(FRICATIVES)))]
                seg_type[j:k] = 2
                fric_c[j:k] = c
                fric_l[j:k] = lev
                amp[j:k] = word_gain * float(rng.uniform(0.25, 0.55))
            else:
                # plosive: brief closure then burst
                stop = min(k, j + int(0.02 * sr))
                seg_type[j:stop] = 0
                amp[j:stop] = 0.0
                seg_type[stop:k] = 3
                amp[stop:k] = word_gain * float(rng.uniform(0.5, 0.9))
            j = k
        i = word_end + int(rng.uniform(0.12, 0.65) * sr)  # pause between words

    # ---- excitation
    jitter = 1.0 + 0.02 * rng.standard_normal(n)
    contour = 1.0 + 0.10 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 6.28))
    f0 = np.clip(f0_base * contour * jitter, 60.0, 320.0)
    voiced = (seg_type == 1).astype(np.float64)
    source = _pulse_train(f0, sr) * _smooth(voiced, 128)

    noise = rng.standard_normal(n)
    fric = np.zeros(n)
    for c, lev in FRICATIVES:
        sel = np.isclose(fric_c, c) & (fric_l > 0)
        if not np.any(sel):
            continue
        band = _butter_filter(noise, [max(c - 1200.0, 300.0), min(c + 1500.0, 0.48 * sr)], sr, "bandpass", order=2)
        fric[sel] = lev * band[sel]
    burst = np.zeros(n)
    plos = seg_type == 3
    if np.any(plos):
        hp = _butter_filter(noise, 900.0, sr, "highpass", order=2)
        burst[plos] = 0.7 * hp[plos]

    src = source + fric + burst + 0.004 * rng.standard_normal(n)

    # ---- vocal tract
    out = src.copy()
    for f, bw, g in ((f1, 90.0, 1.0), (f2, 110.0, 0.7), (f3, 160.0, 0.4)):
        out = out + g * _formant_filter(src, _smooth(f, 256), bw, sr)
    if tilt < 0:
        out = out + tilt * _butter_filter(out, 2000.0, sr, "highpass", order=1)

    env = _smooth(amp, 96)
    out = out * env
    out = _butter_filter(out, [90.0, min(top_hz, 0.47 * sr)], sr, "bandpass", order=3)
    peak = float(np.max(np.abs(out)) + 1e-12)
    if peak > 1e-6:
        out = normalize(out, float(rng.uniform(0.18, 0.62)))
    mask = (env > 0.02 * (float(np.max(env)) + 1e-12)).astype(np.float64)
    return out, mask
