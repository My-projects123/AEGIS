"""Mode-aware speech enhancer.

Primary path: class-conditioned / mode-conditioned Wiener DSP (always available).
Optional path: RNNoise (pyrnnoise) mixed in for STATIONARY / DYNAMIC if installed.

Honesty:
  DSP Wiener           REAL, always on
  RNNoise              REAL if import succeeds, else unused
  DeepFilterNet        not used in tonight's POC (heavier stack)
"""

from __future__ import annotations

import numpy as np

from backend.config import HOP, MODE_PARAMS, SAMPLE_RATE, WIN_LENGTH
from enhancement.dsp_fallback import WienerEnhancer, highpass, overlap_add, peak_limit
from enhancement.speech_gate import SpeechGate

_RNNOISE = None
_RNNOISE_ERR = None
_TINY_ERR = None


def try_load_tiny():
    """New masker instance (own GRU state). ONNX session is shared."""
    global _TINY_ERR
    try:
        from ml.infer import TinyOnnxMasker, load_session, tiny_status

        if load_session() is None:
            _TINY_ERR = tiny_status().get("error") or "onnx not loaded"
            return None
        return TinyOnnxMasker()
    except Exception as exc:  # noqa: BLE001
        _TINY_ERR = str(exc)
        return None


def tiny_ml_status() -> dict:
    from ml.infer import tiny_status

    st = tiny_status()
    m = try_load_tiny()
    st["available"] = bool(m)
    if m and hasattr(m, "mean_infer_ms"):
        st["infer_ms_per_hop"] = m.mean_infer_ms()
    if _TINY_ERR and not st.get("error"):
        st["error"] = _TINY_ERR
    return st

_RNNOISE = None
_RNNOISE_ERR = None


def try_load_rnnoise():
    global _RNNOISE, _RNNOISE_ERR
    if _RNNOISE is not None or _RNNOISE_ERR:
        return _RNNOISE
    try:
        from pyrnnoise import RNNoise  # type: ignore

        _RNNOISE = RNNoise(sample_rate=48000)
        return _RNNOISE
    except Exception as exc:  # noqa: BLE001
        _RNNOISE_ERR = str(exc)
        _RNNOISE = False
        return None


def rnnoise_status() -> dict:
    model = try_load_rnnoise()
    if model:
        return {"available": True, "name": "RNNoise (pyrnnoise)", "note": "REAL optional AI enhancer"}
    return {
        "available": False,
        "name": "DSP Wiener (POC primary)",
        "note": f"RNNoise not loaded ({_RNNOISE_ERR}). POC uses real DSP enhancement.",
    }


def _rnnoise_apply(x: np.ndarray, sr: int) -> np.ndarray | None:
    model = try_load_rnnoise()
    if not model:
        return None
    try:
        from backend.synthesizer import resample_to

        x48 = resample_to(x, sr, 48000).astype(np.float32)
        # pyrnnoise expects int16-ish range in some versions; try float then int16
        pcm = np.clip(x48 * 32767.0, -32768, 32767).astype(np.int16)
        out_chunks = []
        if hasattr(model, "denoise_chunk"):
            arr = pcm.reshape(1, -1)
            for _prob, frame in model.denoise_chunk(arr, partial=True):
                out_chunks.append(np.asarray(frame).reshape(-1))
            y48 = np.concatenate(out_chunks) if out_chunks else pcm.astype(np.float32)
        else:
            y48 = pcm.astype(np.float32)
        y48 = np.asarray(y48, dtype=np.float64).reshape(-1)
        if np.max(np.abs(y48)) > 2.0:
            y48 = y48 / 32768.0
        y = resample_to(y48, 48000, sr)
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        return y[: len(x)]
    except Exception:
        return None


class EventAwareEnhancer:
    def __init__(self, sr: int = SAMPLE_RATE, use_ai: bool = True) -> None:
        self.sr = sr
        self.use_ai = use_ai
        self.dsp = WienerEnhancer(sr)
        self.ai_audio = None  # full-file RNNoise cache
        self.ai_ok = False
        self.tiny = try_load_tiny() if use_ai else None
        self.last_vad: float | None = None
        self.gate = SpeechGate()
        if self.tiny:
            self.tiny.reset()

    def reset(self) -> None:
        self.dsp.reset()
        self.ai_audio = None
        self.ai_ok = False
        self.last_vad = None
        self.gate.reset()
        if self.tiny:
            self.tiny.reset()

    def prepare_ai(self, x: np.ndarray) -> None:
        if not self.use_ai:
            return
        y = _rnnoise_apply(x, self.sr)
        if y is not None:
            self.ai_audio = y
            self.ai_ok = True

    def process_frame(
        self,
        frame: np.ndarray,
        mode: str,
        ref: np.ndarray | None = None,
        aggression: float = 1.0,
    ) -> np.ndarray:
        ai_gain = None
        mix = 0.0
        self.last_vad = None
        if self.tiny is not None:
            mix = MODE_PARAMS[mode]["ai_mix"] * aggression
            ai_gain, self.last_vad = self.tiny.mask_frame(frame)
            if ai_gain is None:  # ONNX unavailable → DSP only
                mix = 0.0
        return self.dsp.process_frame(
            frame, mode, ref=ref, ai_gain=ai_gain, ai_mix=mix, aggression=aggression
        )

    def process_file(self, x: np.ndarray, modes: list[str]) -> np.ndarray:
        """Overlap-add using per-frame modes. `modes` length = number of hops."""
        x = np.asarray(x, dtype=np.float64)
        pad = np.pad(x, (WIN_LENGTH, WIN_LENGTH))
        self.dsp.warmup_noise(pad[: max(self.sr // 5, WIN_LENGTH)])
        frames = []
        n_hops = 1 + (len(pad) - WIN_LENGTH) // HOP
        if len(modes) < n_hops:
            modes = modes + [modes[-1] if modes else "STATIONARY"] * (n_hops - len(modes))
        self.gate.reset()
        for i, start in enumerate(range(0, len(pad) - WIN_LENGTH + 1, HOP)):
            mode = modes[min(i, len(modes) - 1)]
            rec = self.process_frame(pad[start : start + WIN_LENGTH], mode)
            if self.ai_ok and self.ai_audio is not None:
                mix = MODE_PARAMS[mode]["ai_mix"]
                # align to original (unpadded) timeline
                orig_idx = start - WIN_LENGTH
                if 0 <= orig_idx < len(self.ai_audio) - WIN_LENGTH:
                    ai_fr = self.ai_audio[orig_idx : orig_idx + WIN_LENGTH] * np.hanning(WIN_LENGTH)
                    rec = (1 - mix) * rec + mix * ai_fr
            if self.last_vad is not None:
                # Same speech-only gate as the live path. Offline there is no DSP
                # vote to fall back on, so this runs on the GRU VAD alone.
                lvl = 20.0 * np.log10(float(np.sqrt(np.mean(rec ** 2))) + 1e-12)
                rec = rec * self.gate.step(0.0, self.last_vad, level_db=lvl)
            frames.append(rec)
        y = overlap_add(frames, HOP, n_samples=len(pad))[WIN_LENGTH : WIN_LENGTH + len(x)]
        return peak_limit(highpass(y, self.sr))
