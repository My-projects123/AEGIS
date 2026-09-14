"""Looping synthetic battlefield overlays for the live microphone.

SIMULATED laboratory stand-ins. Not field recordings. Mixed onto REAL mic audio.
"""

from __future__ import annotations

import json

import numpy as np

from backend.config import SAMPLE_RATE
from backend.synthesizer import (
    normalize,
    synth_drone,
    synth_engine,
    synth_impulse,
    synth_rotor,
    synth_siren,
    synth_wind,
)

LOOP_S = 8.0

DISTURBANCES = (
    ("none", "Mic only"),
    ("engine", "Vehicle engine"),
    ("rotor", "Helicopter rotor"),
    ("drone", "UAV drone"),
    ("wind", "Field wind"),
    ("siren", "Siren"),
    ("engine_rotor", "Engine + rotor"),
)

IMPULSES = (
    ("gunshot", "Gunshot-like"),
    ("artillery", "Artillery-like"),
    ("explosion", "Blast-like"),
)

_LOOPS: dict[str, np.ndarray] = {}

# Acoustic path from the disturbance to the PRIMARY (boom) mic, expressed
# relative to what the REFERENCE mic hears: a few samples of propagation delay
# plus reflections and a level difference. SIMULATED — a real headset would
# measure its own path. The NLMS filter has to learn this; it is not told.
PRIMARY_PATH = np.array(
    [0.0, 0.0, 0.0, 0.62, 0.28, -0.14, 0.09, -0.05, 0.03, -0.02],
    dtype=np.float64,
)
REF_SENSOR_NOISE = 0.01  # reference mic self-noise, so cancellation is not exact
# Fixed headroom on both channels. Must be constant: per-buffer peak
# normalisation would make the path time-varying and the NLMS filter would
# spend its life re-converging instead of cancelling.
MIX_HEADROOM = 0.7


def _build_loops() -> dict[str, np.ndarray]:
    if _LOOPS:
        return _LOOPS
    eng = synth_engine(LOOP_S, sr=SAMPLE_RATE, seed=41)
    rot = synth_rotor(LOOP_S, sr=SAMPLE_RATE, seed=43)
    _LOOPS["engine"] = normalize(eng, 0.9)
    _LOOPS["rotor"] = normalize(rot, 0.9)
    _LOOPS["drone"] = normalize(synth_drone(LOOP_S, sr=SAMPLE_RATE, seed=47), 0.85)
    _LOOPS["wind"] = normalize(synth_wind(LOOP_S, sr=SAMPLE_RATE, seed=53), 0.85)
    _LOOPS["siren"] = normalize(synth_siren(LOOP_S, sr=SAMPLE_RATE, seed=59), 0.8)
    mix = 0.62 * eng + 0.72 * rot
    _LOOPS["engine_rotor"] = normalize(mix, 0.9)
    return _LOOPS


def disturbance_catalog() -> list[dict]:
    return [{"id": k, "label": lab} for k, lab in DISTURBANCES]


def impulse_catalog() -> list[dict]:
    return [{"id": k, "label": lab} for k, lab in IMPULSES]


class LiveDisturbance:
    """Adds a looping SIMULATED defence noise (and optional impulse) onto mic hops."""

    def __init__(self) -> None:
        _build_loops()
        self.kind = "none"
        self.gain = 0.42
        self.pos = 0
        self._impulse: np.ndarray | None = None
        self.last_impulse = ""
        self._path_tail = np.zeros(len(PRIMARY_PATH) - 1)
        self._rng = np.random.default_rng(1234)

    def status(self) -> dict:
        label = dict(DISTURBANCES).get(self.kind, self.kind)
        return {
            "kind": self.kind,
            "label": label,
            "gain": float(self.gain),
            "simulated": self.kind != "none" or bool(self.last_impulse),
            "last_impulse": self.last_impulse,
        }

    def handle_text(self, text: str) -> dict:
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            if text == "reset":
                self.kind = "none"
                self.pos = 0
                self._impulse = None
                self.last_impulse = ""
            return self.status()
        typ = msg.get("type")
        if typ == "config":
            kind = str(msg.get("kind") or "none")
            if kind in dict(DISTURBANCES):
                if kind != self.kind:
                    self.pos = 0
                self.kind = kind
            if "gain" in msg:
                self.gain = float(np.clip(float(msg["gain"]), 0.0, 1.0))
        elif typ == "impulse":
            ik = str(msg.get("kind") or "gunshot")
            if ik not in dict(IMPULSES):
                ik = "gunshot"
            burst = synth_impulse(0.32, sr=SAMPLE_RATE, seed=int(np.random.randint(1, 10_000)), kind=ik)
            # Pre-compensate the primary-path loss and the mix headroom, so the
            # burst still arrives well above the noise bed at the primary mic.
            # A real gunshot is tens of dB louder than an engine; digital full
            # scale caps how much of that we can represent, and the primary may
            # clip — which is also what a real mic does.
            self._impulse = burst * 2.0
            self.last_impulse = ik
        elif typ == "reset":
            self.kind = "none"
            self.pos = 0
            self._impulse = None
            self.last_impulse = ""
        return self.status()

    def _take(self, n: int) -> np.ndarray:
        loop = _LOOPS[self.kind]
        out = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            remain = len(loop) - self.pos
            take = min(remain, n - i)
            out[i : i + take] = loop[self.pos : self.pos + take]
            self.pos += take
            i += take
            if self.pos >= len(loop):
                self.pos = 0
        return out

    def _through_path(self, ref: np.ndarray) -> np.ndarray:
        """Filter the reference through the SIMULATED primary-mic path."""
        xp = np.concatenate([self._path_tail, ref])
        y = np.convolve(xp, PRIMARY_PATH, mode="full")[len(self._path_tail) : len(self._path_tail) + len(ref)]
        self._path_tail = xp[-(len(PRIMARY_PATH) - 1) :]
        return y

    def mix(self, mic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (primary, reference).

        primary   = REAL mic + disturbance through the primary-mic path
        reference = the disturbance as the reference mic hears it, plus sensor noise

        The two channels are deliberately *not* identical: the NLMS canceller
        has to learn the path between them, exactly as it would on a real
        two-mic headset. Both channels are SIMULATED disturbance on REAL voice.
        """
        mic = np.asarray(mic, dtype=np.float64).reshape(-1)
        src = np.zeros(len(mic), dtype=np.float64)
        if self.kind != "none" and self.kind in _LOOPS and self.gain > 0:
            src += self.gain * self._take(len(mic))
        if self._impulse is not None and len(self._impulse):
            k = min(len(mic), len(self._impulse))
            src[:k] += self._impulse[:k]
            rest = self._impulse[k:]
            self._impulse = rest if len(rest) else None
        ref = src + REF_SENSOR_NOISE * self._rng.standard_normal(len(src)) * (np.max(np.abs(src)) + 1e-6)
        y = MIX_HEADROOM * (mic + self._through_path(src))
        ref = MIX_HEADROOM * ref
        return np.clip(y, -1.0, 1.0).astype(np.float32), ref.astype(np.float32)
