"""Causal decision-directed Wiener enhancer.

REAL DSP. This is the always-on baseline. Mode parameters change
oversubtraction, noise-update rate, speech protection, and impulse gating.

Two input paths:
  primary only          blind noise PSD tracking (offline files, no reference)
  primary + reference   the reference channel PSD is subtracted directly

The reference path is the "primary + reference microphone" arrangement the
problem statement asks for. In this POC the disturbance is synthesised by us,
so the reference is sample-aligned and exact — that is SIMULATED, and a real
reference mic would be noisier and partially correlated with the voice.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

from backend.config import (
    AGC_MAX_GAIN_DB,
    AGC_TARGET_DBFS,
    DD_BETA,
    HPF_CUTOFF_HZ,
    HOP,
    LIMITER_DB,
    MODE_PARAMS,
    N_FFT,
    SAMPLE_RATE,
    WIN_LENGTH,
)


def highpass(x: np.ndarray, sr: int = SAMPLE_RATE, cutoff: float = HPF_CUTOFF_HZ) -> np.ndarray:
    b, a = butter(2, cutoff / (0.5 * sr), btype="highpass")
    return lfilter(b, a, x)


def peak_limit(x: np.ndarray, limit_db: float = LIMITER_DB) -> np.ndarray:
    lim = 10 ** (limit_db / 20.0)
    peak = np.max(np.abs(x)) + 1e-12
    if peak > lim:
        x = x * (lim / peak)
    return np.clip(x, -1.0, 1.0)


def smooth_across_bins(g: np.ndarray) -> np.ndarray:
    """3-tap frequency smoothing. Isolated surviving bins are what musical noise
    (the warbling/metallic residue of spectral subtraction) is made of."""
    padded = np.concatenate([g[:1], g, g[-1:]])
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]


class WienerEnhancer:
    """Overlap-add decision-directed Wiener with an optional reference channel.

    Blind noise PSD is updated only when the controller allows it.
    During IMPULSE_PROTECTION the PSD is frozen and a transient peak limiter
    is applied in the time domain on the residual.
    """

    def __init__(self, sr: int = SAMPLE_RATE) -> None:
        self.sr = sr
        self.n_fft = N_FFT
        self.hop = HOP
        self.win = np.hanning(WIN_LENGTH)
        bins = self.n_fft // 2 + 1
        self.noise_psd = np.ones(bins) * 1e-6
        self.ref_psd = np.zeros(bins)
        self.prev_gain = np.ones(bins)
        self.prev_pow = np.zeros(bins)
        self.prev_frame = np.zeros(WIN_LENGTH)
        freqs = np.fft.rfftfreq(self.n_fft, 1 / self.sr)
        self.speech_bins = (freqs >= 300) & (freqs <= 3400)
        self.ola = None
        self.inited = False

    def reset(self) -> None:
        self.__init__(self.sr)

    def warmup_noise(self, x: np.ndarray) -> None:
        if len(x) < WIN_LENGTH:
            return
        spec = np.fft.rfft(x[:WIN_LENGTH] * self.win, n=self.n_fft)
        self.noise_psd = np.maximum(np.abs(spec) ** 2, 1e-8)

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        if len(frame) != WIN_LENGTH:
            frame = np.pad(frame, (0, max(0, WIN_LENGTH - len(frame))))[:WIN_LENGTH]
        return frame

    def process_frame(
        self,
        frame: np.ndarray,
        mode: str,
        ref: np.ndarray | None = None,
        ai_gain: np.ndarray | None = None,
        ai_mix: float = 0.0,
        aggression: float = 1.0,
    ) -> np.ndarray:
        """`aggression` in (0, 1] scales suppression down when an upstream stage
        (the NLMS canceller) has already removed the disturbance. Suppressing
        noise that is no longer there only damages the voice."""
        p = MODE_PARAMS[mode]
        aggression = float(np.clip(aggression, 0.05, 1.0))
        frame = self._fit(np.asarray(frame, dtype=np.float64))
        spec = np.fft.rfft(frame * self.win, n=self.n_fft)
        pow_y = np.abs(spec) ** 2
        phase = np.angle(spec)

        if ref is not None:
            ref_spec = np.fft.rfft(self._fit(np.asarray(ref, dtype=np.float64)) * self.win, n=self.n_fft)
            # light smoothing only: this reference is sample-aligned
            self.ref_psd = 0.5 * self.ref_psd + 0.5 * np.abs(ref_spec) ** 2
        else:
            self.ref_psd = np.zeros_like(self.ref_psd)

        alpha = p["noise_alpha"]
        if alpha < 0.999:
            # update noise in likely non-speech bins
            speech_like = pow_y > 2.5 * self.noise_psd
            update = (~speech_like) | (mode in ("STATIONARY", "DYNAMIC") and np.mean(pow_y) < 1.8 * np.mean(self.noise_psd))
            self.noise_psd = np.where(
                update,
                alpha * self.noise_psd + (1 - alpha) * pow_y,
                self.noise_psd,
            )
        # freeze when alpha == 1.0 (impulse)

        over = 1.0 + (p["oversubtraction"] - 1.0) * aggression
        ref_over = p["ref_over"] * aggression
        noise = over * self.noise_psd + ref_over * self.ref_psd
        noise = np.maximum(noise, 1e-10)

        # decision-directed a-priori SNR → Wiener gain
        gamma = pow_y / noise
        xi_prev = (self.prev_gain ** 2) * self.prev_pow / noise
        xi = DD_BETA * xi_prev + (1.0 - DD_BETA) * np.maximum(gamma - 1.0, 0.0)
        gain = xi / (1.0 + xi)

        # Speech protection lifts 300–3400 Hz, but only where the reference says
        # there is no known disturbance. Lifting bins the reference has flagged
        # is exactly what used to leave engine/rotor energy sitting on the voice.
        ref_frac = np.clip(ref_over * self.ref_psd / noise, 0.0, 1.0)
        protect = p["speech_protect"] * (1.0 - ref_frac)
        sb = self.speech_bins
        gain[sb] = protect[sb] * np.maximum(gain[sb], 0.35) + (1.0 - protect[sb]) * gain[sb]

        if ai_gain is not None and ai_mix > 0.0:
            # cascade: the tiny GRU mask refines the DSP gain instead of being
            # averaged with its output (one STFT, no double windowing)
            g_ai = np.clip(np.asarray(ai_gain, dtype=np.float64), 0.0, 1.0)
            if len(g_ai) == len(gain):
                gain = gain * ((1.0 - ai_mix) + ai_mix * g_ai)

        floor = p["gain_floor"] + (1.0 - p["gain_floor"]) * 0.25 * (1.0 - aggression)
        gain = np.clip(gain, floor, 1.0)
        gain = smooth_across_bins(gain)
        gain = 0.6 * gain + 0.4 * self.prev_gain
        self.prev_gain = gain
        self.prev_pow = pow_y

        mag = np.sqrt(pow_y) * gain
        rec = np.fft.irfft(mag * np.exp(1j * phase), n=self.n_fft).real[:WIN_LENGTH]
        rec = rec * self.win

        gate = p["impulse_gate"]
        if gate > 0:
            # Peak-limit the residual transient. The old version cross-faded to
            # 0.12 * previous frame, which also muted the voice for ~100 ms.
            prev_peak = np.max(np.abs(self.prev_frame)) + 1e-12
            rec_peak = np.max(np.abs(rec)) + 1e-12
            cap = max(prev_peak * (1.4 - 0.6 * gate), 0.06)
            if rec_peak > cap:
                rec *= cap / rec_peak
        self.prev_frame = rec
        return rec


class OutputAGC:
    """Slow, capped make-up gain for the live path.

    Suppression removes energy, so the cleaned voice ends up quieter than the
    noisy input. Without this the demo sounds "clean but far away".
    Offline evaluation does not use it — it would change absolute SNR.
    """

    def __init__(
        self,
        target_dbfs: float = AGC_TARGET_DBFS,
        max_gain_db: float = AGC_MAX_GAIN_DB,
        alpha: float = 0.92,
    ) -> None:
        self.target = 10.0 ** (target_dbfs / 20.0)
        self.max_gain = 10.0 ** (max_gain_db / 20.0)
        self.alpha = alpha
        self.gain = 1.0

    def reset(self) -> None:
        self.gain = 1.0

    def process(self, hop: np.ndarray, active: bool = True) -> np.ndarray:
        rms = float(np.sqrt(np.mean(hop ** 2) + 1e-12))
        if active and rms > 1e-4:
            desired = float(np.clip(self.target / rms, 0.5, self.max_gain))
        else:
            desired = self.gain  # hold through pauses, do not pump up the noise
        self.gain = self.alpha * self.gain + (1.0 - self.alpha) * desired
        y = hop * self.gain
        peak = float(np.max(np.abs(y)) + 1e-12)
        if peak > 0.98:
            y = y * (0.98 / peak)
        return np.clip(y, -1.0, 1.0)

    def gain_db(self) -> float:
        return float(20.0 * np.log10(self.gain + 1e-12))


def overlap_add(frames: list[np.ndarray], hop: int = HOP, n_samples: int | None = None) -> np.ndarray:
    if not frames:
        return np.zeros(0)
    win_len = len(frames[0])
    n_out = hop * (len(frames) - 1) + win_len
    acc = np.zeros(n_out)
    wsum = np.zeros(n_out)
    w = np.hanning(win_len)
    for i, fr in enumerate(frames):
        s = i * hop
        acc[s : s + win_len] += fr
        wsum[s : s + win_len] += w ** 2
    acc = acc / np.maximum(wsum, 1e-6)
    if n_samples is not None:
        acc = acc[:n_samples]
        if len(acc) < n_samples:
            acc = np.pad(acc, (0, n_samples - len(acc)))
    return acc


def classical_dsp(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Fixed Wiener baseline (no adaptive modes) for comparison."""
    enh = WienerEnhancer(sr)
    n = len(x)
    pad = np.pad(x, (WIN_LENGTH, WIN_LENGTH))
    frames = []
    enh.warmup_noise(pad[: max(sr // 4, WIN_LENGTH)])
    for start in range(0, len(pad) - WIN_LENGTH + 1, HOP):
        frames.append(enh.process_frame(pad[start : start + WIN_LENGTH], "STATIONARY"))
    y = overlap_add(frames, HOP, n_samples=len(pad))[WIN_LENGTH : WIN_LENGTH + n]
    return peak_limit(highpass(y, sr))
