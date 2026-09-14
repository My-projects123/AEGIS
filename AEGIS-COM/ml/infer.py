"""ONNX Runtime hop enhancer. Raspberry Pi uses the same INT8 (or FP32) file."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from backend.config import N_FFT, SAMPLE_RATE, WIN_LENGTH
from ml.fbanks import filterbank, to_bands, upsample_gain
from ml.paths import INT8_ONNX, FP32_ONNX, META_JSON

_SESSION = None
_META: dict = {}
_ERR: str | None = None


def _pick_model() -> Path | None:
    if INT8_ONNX.exists():
        return INT8_ONNX
    if FP32_ONNX.exists():
        return FP32_ONNX
    return None


def tiny_status() -> dict:
    load_session()
    path = _pick_model()
    return {
        "available": _SESSION is not None,
        "backend": "onnxruntime" if _SESSION is not None else "none",
        "model_path": str(path) if path else None,
        "quantized": bool(path and "int8" in path.name),
        "params": _META.get("params"),
        "size_kb": _META.get("size_kb"),
        "n_bands": _META.get("n_bands", 32),
        "hidden": _META.get("hidden", 32),
        "heads": _META.get("heads"),
        "vad_accuracy": _META.get("vad_accuracy"),
        "vad_recall": _META.get("vad_recall"),
        "vad_false_alarm": _META.get("vad_false_alarm"),
        "train_data": _META.get("train_data", "SIMULATED speech + defence-like noise"),
        "target_device": "Raspberry Pi 4/5 (ONNX Runtime CPU). Pi 1 is too slow for real-time.",
        "pipeline": ["PyTorch", "ONNX", "INT8 quantize", "Raspberry Pi", "ONNX Runtime"],
        "error": _ERR,
        "meta": _META,
    }


def load_session():
    global _SESSION, _META, _ERR
    if _SESSION is not None or _ERR:
        return _SESSION
    path = _pick_model()
    if path is None:
        _ERR = "No ONNX file. Run: python ml/train_tiny.py"
        return None
    try:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        _SESSION = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        if META_JSON.exists():
            _META = json.loads(META_JSON.read_text())
        _META["size_kb"] = round(path.stat().st_size / 1024.0, 2)
        _META["model_file"] = path.name
        return _SESSION
    except Exception as exc:  # noqa: BLE001
        _ERR = str(exc)
        _SESSION = None
        return None


class TinyOnnxMasker:
    """Causal hop: 512-sample window in, windowed time frame out (for OLA)."""

    def __init__(self) -> None:
        self.sess = load_session()
        self.ok = self.sess is not None
        self.fb = filterbank()
        self.win = np.hanning(WIN_LENGTH)
        self.hidden = int(_META.get("hidden") or 32)
        self.h = np.zeros((1, 1, self.hidden), dtype=np.float32)
        self._infer_s = 0.0
        self._infer_n = 0
        self._has_vad = bool(self.sess) and any(o.name == "vad" for o in self.sess.get_outputs())
        self.last_vad: float | None = None

    def reset(self) -> None:
        self.h = np.zeros((1, 1, self.hidden), dtype=np.float32)
        self.last_vad = None

    def mask_frame(self, frame: np.ndarray) -> tuple[np.ndarray | None, float | None]:
        """Return (per-bin gain, speech probability) for this frame.

        The gain cascades onto the DSP gain; the probability drives the speech
        gate. Returns (None, None) if ONNX is unavailable, so callers fall back
        to DSP only. Models exported before the VAD head return a gain and None.
        """
        if not self.ok:
            return None, None
        x = np.asarray(frame, dtype=np.float64)
        if len(x) != WIN_LENGTH:
            x = np.pad(x, (0, max(0, WIN_LENGTH - len(x))))[:WIN_LENGTH]
        spec = np.fft.rfft(x * self.win, n=N_FFT)
        mag = np.clip(np.abs(spec), 1e-12, 1e6)
        bands = to_bands(mag.astype(np.float32), self.fb)
        logb = np.log(bands + 1e-8).reshape(1, 1, -1).astype(np.float32)
        t0 = time.perf_counter()
        vad = None
        if self._has_vad:
            try:
                gain, vad_out, h_out = self.sess.run(["gain", "vad", "h_out"], {"bands": logb, "h": self.h})
                vad = float(np.asarray(vad_out).reshape(-1)[0])
            except Exception:
                self._has_vad = False
                gain, h_out = self.sess.run(["gain", "h_out"], {"bands": logb, "h": self.h})
        else:
            gain, h_out = self.sess.run(["gain", "h_out"], {"bands": logb, "h": self.h})
        self._infer_s += time.perf_counter() - t0
        self._infer_n += 1
        self.h = np.asarray(h_out, dtype=np.float32)
        self.last_vad = vad
        g = upsample_gain(np.asarray(gain).reshape(-1))
        if len(g) != len(mag):
            g = np.interp(np.linspace(0, 1, len(mag)), np.linspace(0, 1, len(g)), g)
        return np.clip(g, 0.0, 1.0), vad

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Standalone masked frame (windowed, ready for overlap-add)."""
        x = np.asarray(frame, dtype=np.float64)
        if len(x) != WIN_LENGTH:
            x = np.pad(x, (0, max(0, WIN_LENGTH - len(x))))[:WIN_LENGTH]
        spec = np.fft.rfft(x * self.win, n=N_FFT)
        mag = np.clip(np.abs(spec), 1e-12, 1e6)
        phase = np.angle(spec)
        g, _vad = self.mask_frame(x)
        if g is None:
            rec = np.fft.irfft(mag * np.exp(1j * phase), n=N_FFT).real[:WIN_LENGTH]
            return rec * self.win
        rec = np.fft.irfft((mag * g) * np.exp(1j * phase), n=N_FFT).real[:WIN_LENGTH]
        return rec * self.win

    def mean_infer_ms(self) -> float | None:
        if self._infer_n < 8:
            return None
        return 1000.0 * self._infer_s / self._infer_n
