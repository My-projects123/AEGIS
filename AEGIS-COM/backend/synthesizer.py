"""Synthetic demonstration signals.

SIMULATED: these are laboratory stand-ins for defence recordings.
They are labelled as synthetic in the UI and must never be presented
as field gunshot / helicopter captures.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, resample_poly

from backend.config import SAMPLE_RATE


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _butter_filter(data: np.ndarray, cutoff: float, sr: int, btype: str, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sr
    normal = np.clip(np.atleast_1d(cutoff) / nyq, 1e-4, 0.999)
    b, a = butter(order, normal, btype=btype)
    return lfilter(b, a, data)


def fade(x: np.ndarray, sr: int = SAMPLE_RATE, ms: float = 15.0) -> np.ndarray:
    n = int(sr * ms / 1000.0)
    n = min(n, len(x) // 4)
    if n <= 0:
        return x
    env = np.ones(len(x), dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, n)
    env[:n] *= ramp
    env[-n:] *= ramp[::-1]
    return x * env


def normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.max(np.abs(x)) + 1e-12
    return x * (peak / m)


def synth_speech(duration_s: float, sr: int = SAMPLE_RATE, seed: int = 7) -> np.ndarray:
    """Klatt-lite radio voice: glottal pulses through moving formants.

    SIMULATED speech — not a recorded talker.
    """
    n = int(duration_s * sr)
    rng = _rng(seed)
    t = np.arange(n) / sr

    f0 = 118.0 + 8.0 * np.sin(2 * np.pi * 2.2 * t)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    # band-limited pulse train
    source = np.zeros(n)
    last = 0.0
    acc = 0.0
    for i, f in enumerate(f0):
        acc += f / sr
        if acc >= 1.0:
            acc -= 1.0
            source[i] = 1.0
            last = i
        elif i - last < 4:
            source[i] = source[i]  # keep impulse
    source = source + 0.015 * rng.normal(size=n)

    # syllable amplitude: ~4 Hz speech envelope
    syllables = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.4 * t + 0.3))
    syllables *= 0.25 + 0.75 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.45 * t))
    # unvoiced bursts (consonant-like)
    uv = (np.sin(2 * np.pi * 7.1 * t) > 0.92).astype(np.float64)
    noise = rng.normal(size=n)
    noise = _butter_filter(noise, 3500.0, sr, "highpass", order=2)
    source = source * (1.0 - 0.35 * uv) + 0.12 * noise * uv
    source *= syllables

    # time-varying formants (rough /a/ /e/ /i/ /o/)
    f1 = 700 + 180 * np.sin(2 * np.pi * 0.9 * t)
    f2 = 1220 + 380 * np.sin(2 * np.pi * 1.1 * t + 0.7)
    f3 = 2450 + 220 * np.sin(2 * np.pi * 0.6 * t + 1.4)

    out = source.copy()
    for formant, bw, gain in ((f1, 90.0, 1.0), (f2, 110.0, 0.7), (f3, 160.0, 0.4)):
        # resonant IIR approximated by sliding biquad in blocks
        block = 256
        y = np.zeros_like(out)
        for start in range(0, n, block):
            sl = slice(start, min(start + block, n))
            fc = float(np.mean(formant[sl]))
            r = np.exp(-np.pi * bw / sr)
            w = 2 * np.pi * fc / sr
            b = [1.0]
            a = [1.0, -2 * r * np.cos(w), r * r]
            y[sl] = lfilter(b, a, out[sl])
        out = out + gain * y

    out = _butter_filter(out, [80.0, 4500.0], sr, "bandpass", order=3)
    return fade(normalize(out, 0.55), sr)


def synth_engine(duration_s: float, sr: int = SAMPLE_RATE, seed: int = 11) -> np.ndarray:
    """Low-frequency harmonic engine rumble. SIMULATED."""
    n = int(duration_s * sr)
    rng = _rng(seed)
    t = np.arange(n) / sr
    rpm = 28.0 + 3.0 * np.sin(2 * np.pi * 0.15 * t)
    x = np.zeros(n)
    for k in range(1, 9):
        x += (1.0 / k) * np.sin(2 * np.pi * rpm * k * t + rng.uniform(0, 2 * np.pi))
    x += 0.25 * _butter_filter(rng.normal(size=n), 180.0, sr, "lowpass")
    x = _butter_filter(x, 40.0, sr, "highpass", order=2)
    return fade(normalize(x, 0.85), sr)


def synth_rotor(duration_s: float, sr: int = SAMPLE_RATE, seed: int = 13) -> np.ndarray:
    """Blade-passing amplitude modulation. SIMULATED helicopter-like."""
    n = int(duration_s * sr)
    rng = _rng(seed)
    t = np.arange(n) / sr
    bpf = 36.0 + 2.5 * np.sin(2 * np.pi * 0.2 * t)
    carrier = _butter_filter(rng.normal(size=n), 1800.0, sr, "lowpass")
    carrier += 0.35 * np.sin(2 * np.pi * 85.0 * t)
    carrier += 0.25 * _butter_filter(rng.normal(size=n), [1200.0, 4000.0], sr, "bandpass", order=2)
    mod = 0.55 + 0.45 * np.sin(2 * np.pi * bpf * t)
    x = carrier * mod
    x += 0.15 * np.sin(2 * np.pi * 4.0 * bpf * t) * _butter_filter(rng.normal(size=n), 2000.0, sr, "lowpass")
    return fade(normalize(x, 0.85), sr)


def synth_wind(duration_s: float, sr: int = SAMPLE_RATE, seed: int = 17) -> np.ndarray:
    """1/f-ish buffeting. SIMULATED."""
    n = int(duration_s * sr)
    rng = _rng(seed)
    spec = rng.normal(size=n // 2 + 1) + 1j * rng.normal(size=n // 2 + 1)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    spec /= np.maximum(freqs, 8.0) ** 0.85
    x = np.fft.irfft(spec, n=n).real
    gust = 0.7 + 0.3 * np.sin(2 * np.pi * 0.35 * np.arange(n) / sr)
    x = _butter_filter(x * gust, 80.0, sr, "highpass", order=2)
    return fade(normalize(x, 0.8), sr)


def synth_siren(duration_s: float, sr: int = SAMPLE_RATE, seed: int = 19) -> np.ndarray:
    """Two-tone emergency siren. SIMULATED."""
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    f = 650.0 + 400.0 * np.sin(2 * np.pi * 0.85 * t)
    x = 0.7 * np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 1.5 * f * t)
    return fade(normalize(x, 0.75), sr)


def synth_drone(duration_s: float, sr: int = SAMPLE_RATE, seed: int = 23) -> np.ndarray:
    """High-pitch motor whine. SIMULATED UAV-like."""
    n = int(duration_s * sr)
    rng = _rng(seed)
    t = np.arange(n) / sr
    f = 1850.0 + 80.0 * np.sin(2 * np.pi * 6.0 * t)
    x = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t)
    x += 0.2 * _butter_filter(rng.normal(size=n), [1200.0, 5000.0], sr, "bandpass")
    return fade(normalize(x, 0.7), sr)


def synth_impulse(
    duration_s: float = 0.35,
    sr: int = SAMPLE_RATE,
    seed: int = 29,
    kind: str = "gunshot",
) -> np.ndarray:
    """Transient burst. SIMULATED gunshot / blast-like — not a real recording."""
    n = int(duration_s * sr)
    rng = _rng(seed)
    t = np.arange(n) / sr
    if kind == "artillery":
        thump = np.exp(-t / 0.045) * np.sin(2 * np.pi * 55.0 * t)
        crack = np.exp(-t / 0.012) * rng.normal(size=n)
        x = 0.7 * thump + 0.9 * crack
    elif kind == "explosion":
        thump = np.exp(-t / 0.08) * np.sin(2 * np.pi * 40.0 * t)
        crack = np.exp(-t / 0.02) * rng.normal(size=n)
        x = 0.85 * thump + 0.7 * crack
    else:  # gunshot-like: sharp crack + short thump
        crack = np.exp(-t / 0.006) * rng.normal(size=n)
        crack = _butter_filter(crack, 400.0, sr, "highpass", order=1)
        thump = np.exp(-t / 0.018) * np.sin(2 * np.pi * 90.0 * t)
        x = 1.0 * crack + 0.45 * thump
    x[: max(1, int(0.0004 * sr))] *= np.linspace(0, 1, max(1, int(0.0004 * sr)))
    return normalize(x, 0.99)


def overlay_impulse(x: np.ndarray, at_s: float, sr: int = SAMPLE_RATE, kind: str = "gunshot", seed: int = 29) -> np.ndarray:
    y = x.copy()
    imp = synth_impulse(0.28, sr=sr, seed=seed, kind=kind)
    start = int(at_s * sr)
    end = min(len(y), start + len(imp))
    y[start:end] += imp[: end - start]
    return np.clip(y, -1.0, 1.0)


def resample_to(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x.astype(np.float64)
    g = np.gcd(src_sr, dst_sr)
    return resample_poly(x, dst_sr // g, src_sr // g)


def make_demo_scene(sr: int = SAMPLE_RATE, seed: int = 3) -> dict:
    """Scripted 12.5 s scene used by START DEMO.

    Timeline (seconds):
      0.0–2.0   clean speech
      2.0–5.0   engine (stationary)
      5.0–8.0   engine + rotor (dynamic / mixed)
      8.00      impulsive event
      8.0–10.2  impulse tail + residual engine (recovery)
      10.2–12.5 cleaner speech
    """
    duration = 12.5
    speech = synth_speech(duration, sr=sr, seed=seed)
    engine = synth_engine(duration, sr=sr, seed=seed + 1)
    rotor = synth_rotor(duration, sr=sr, seed=seed + 2)

    n = len(speech)
    t = np.arange(n) / sr
    speech = normalize(speech, 0.38)
    engine = normalize(engine, 0.9)
    rotor = normalize(rotor, 0.9)

    def _gate(t0, t1, fade_s=0.18):
        g = np.zeros(n)
        on = (t >= t0) & (t < t1)
        g[on] = 1.0
        fade_in = (t >= t0) & (t < t0 + fade_s)
        fade_out = (t >= t1 - fade_s) & (t < t1)
        if np.any(fade_in):
            g[fade_in] = np.linspace(0.0, 1.0, np.count_nonzero(fade_in))
        if np.any(fade_out):
            g[fade_out] = np.minimum(g[fade_out], np.linspace(1.0, 0.0, np.count_nonzero(fade_out)))
        return g

    eng_env = _gate(2.0, 10.2, fade_s=0.22)
    rot_env = _gate(5.0, 8.15, fade_s=0.20)

    noise = 0.50 * engine * eng_env + 0.62 * rotor * rot_env
    # Keep headroom so the impulse is not destroyed by clipping.
    noisy = mix_at_snr(speech, noise, snr_db=6.0)
    peak = np.max(np.abs(noisy)) + 1e-12
    noisy = noisy * (0.42 / peak)
    noisy = overlay_impulse(noisy, at_s=8.0, sr=sr, kind="gunshot", seed=seed + 9)
    # Soft clip only samples that still exceed full scale
    noisy = np.tanh(noisy * 1.15) / np.tanh(1.15)

    labels = [
        {"t0": 0.0, "t1": 2.0, "scene": "clean speech", "expected_mode": "SPEECH_PRESERVATION"},
        {"t0": 2.0, "t1": 5.0, "scene": "engine (stationary)", "expected_mode": "STATIONARY"},
        {"t0": 5.0, "t1": 8.0, "scene": "engine + rotor (dynamic)", "expected_mode": "DYNAMIC"},
        {"t0": 8.0, "t1": 8.15, "scene": "impulsive event", "expected_mode": "IMPULSE_PROTECTION"},
        {"t0": 8.15, "t1": 10.2, "scene": "post-impulse recovery", "expected_mode": "RECOVERY"},
        {"t0": 10.2, "t1": 12.5, "scene": "speech restoration", "expected_mode": "SPEECH_PRESERVATION"},
    ]
    return {
        "clean": speech,
        "noisy": np.clip(noisy, -1.0, 1.0),
        "noise": noise,
        "sr": sr,
        "labels": labels,
        "impulse_time_s": 8.0,
        "kind": "SIMULATED demo scene (not field recordings)",
    }


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    n = min(len(clean), len(noise))
    s = clean[:n].astype(np.float64)
    d = noise[:n].astype(np.float64)
    ps = np.mean(s ** 2) + 1e-12
    pn = np.mean(d ** 2) + 1e-12
    scale = np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))
    mixed = s + scale * d
    peak = np.max(np.abs(mixed)) + 1e-12
    if peak > 0.99:
        mixed *= 0.99 / peak
        s *= 0.99 / peak
    return mixed
