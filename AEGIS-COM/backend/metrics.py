"""Objective metrics. Never fabricate unavailable scores."""

from __future__ import annotations

import time

import numpy as np


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return np.asarray(a[:n], dtype=np.float64), np.asarray(b[:n], dtype=np.float64)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    s, y = _align(clean, estimate)
    noise = y - s
    return float(10.0 * np.log10((np.mean(s ** 2) + 1e-12) / (np.mean(noise ** 2) + 1e-12)))


def si_snr_db(clean: np.ndarray, estimate: np.ndarray) -> float:
    s, y = _align(clean, estimate)
    s -= np.mean(s)
    y -= np.mean(y)
    dot = np.dot(y, s)
    s_energy = np.dot(s, s) + 1e-12
    s_target = (dot / s_energy) * s
    e = y - s_target
    return float(10.0 * np.log10((np.dot(s_target, s_target) + 1e-12) / (np.dot(e, e) + 1e-12)))


def peak_reduction_db(before: np.ndarray, after: np.ndarray, center_s: float, sr: int, window_s: float = 0.12) -> float:
    i0 = max(0, int((center_s - window_s / 2) * sr))
    i1 = min(len(before), int((center_s + window_s / 2) * sr))
    pb = np.max(np.abs(before[i0:i1])) + 1e-12
    pa = np.max(np.abs(after[i0:i1])) + 1e-12
    return float(20.0 * np.log10(pb / pa))


def try_stoi(clean: np.ndarray, estimate: np.ndarray, sr: int) -> dict:
    try:
        import warnings
        from pystoi import stoi as stoi_fn

        s, y = _align(clean, estimate)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = float(stoi_fn(s, y, sr, extended=False))
        if not np.isfinite(val):
            return {"available": False, "value": None, "label": "UNAVAILABLE", "reason": "non-finite STOI"}
        return {"available": True, "value": val, "label": "REAL / MEASURED"}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "value": None, "label": "UNAVAILABLE", "reason": str(exc)}


def try_pesq(clean: np.ndarray, estimate: np.ndarray, sr: int) -> dict:
    try:
        from pesq import pesq as pesq_fn

        s, y = _align(clean, estimate)
        mode = "wb" if sr >= 16000 else "nb"
        val = float(pesq_fn(sr, s, y, mode))
        return {"available": True, "value": val, "label": "REAL / MEASURED"}
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "value": None,
            "label": "UNAVAILABLE / TARGET only",
            "reason": str(exc),
            "target": 2.5,
        }


def summarize_metrics(
    clean: np.ndarray | None,
    noisy: np.ndarray,
    enhanced: np.ndarray,
    sr: int,
    process_seconds: float,
    extra: dict | None = None,
) -> dict:
    out = {
        "latency_ms_per_s": float(1000.0 * process_seconds / (len(noisy) / sr + 1e-12)),
        "rtf": float(process_seconds / (len(noisy) / sr + 1e-12)),
        "audio_duration_s": float(len(noisy) / sr),
        "process_seconds": float(process_seconds),
        "stoi": try_stoi(clean, enhanced, sr) if clean is not None else {"available": False, "reason": "no clean reference"},
        "pesq": try_pesq(clean, enhanced, sr) if clean is not None else {"available": False, "reason": "no clean reference"},
        "honesty": {
            "snr": "REAL / MEASURED" if clean is not None else "UNAVAILABLE (need clean reference)",
            "si_snr": "REAL / MEASURED" if clean is not None else "UNAVAILABLE",
            "latency": "REAL / MEASURED (laptop wall-clock, not embedded)",
            "stoi": "measured if pystoi installed",
            "pesq": "measured if pesq installed, else TARGET only",
        },
    }
    if clean is not None:
        out["input_snr_db"] = snr_db(clean, noisy)
        out["output_snr_db"] = snr_db(clean, enhanced)
        out["snr_improvement_db"] = out["output_snr_db"] - out["input_snr_db"]
        out["input_si_snr_db"] = si_snr_db(clean, noisy)
        out["output_si_snr_db"] = si_snr_db(clean, enhanced)
        out["stoi_noisy"] = try_stoi(clean, noisy, sr)
    else:
        out["input_snr_db"] = None
        out["output_snr_db"] = None
        out["snr_improvement_db"] = None
    if extra:
        out.update(extra)
    return out


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
