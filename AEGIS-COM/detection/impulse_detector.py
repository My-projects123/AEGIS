"""Practical impulse / transient detector.

REAL implementation:
  - onset vs a slow envelope (not a frozen quiet floor)
  - spectral flux, kurtosis, high-band energy
  - warmup so file-start is not treated as a gunshot
  - hard maximum duration so recovery always occurs
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.config import (
    IMPULSE_DB_JUMP,
    IMPULSE_ENERGY_RATIO,
    IMPULSE_FLUX,
    IMPULSE_HOLD_FRAMES,
    IMPULSE_KURTOSIS,
    IMPULSE_REFRACTORY_FRAMES,
)
from detection.acoustic_analyzer import AcousticFeatures

WARMUP_FRAMES = 24
MAX_IMPULSE_FRAMES = 10  # ~160 ms — laboratory bangs are short


@dataclass
class ImpulseEvent:
    active: bool
    probability: float
    just_started: bool
    just_ended: bool
    hold_frames_left: int
    energy_ratio: float
    threshold: float


class ImpulseDetector:
    def __init__(self) -> None:
        self.slow_rms = 1e-3
        self.fast_rms = 1e-3
        self.db_hist: list[float] = []
        self.active = False
        self.hold = 0
        self.refract = 0
        self.age = 0
        self.active_for = 0
        self.onset_slow = 1e-3
        self.detection_latency_hops: list[int] = []
        self.false_candidates = 0
        self.frame_i = 0
        self.noise_rms = 1e-3  # alias used by feature SNR heuristic

    def reset(self) -> None:
        self.__init__()

    def update(self, feat: AcousticFeatures) -> ImpulseEvent:
        just_started = False
        just_ended = False
        self.frame_i += 1

        self.fast_rms = 0.55 * self.fast_rms + 0.45 * feat.rms
        self.noise_rms = self.slow_rms
        if not self.active:
            # slow envelope: rise slowly, fall faster so we track engines
            if feat.rms > self.slow_rms:
                self.slow_rms = 0.98 * self.slow_rms + 0.02 * feat.rms
            else:
                self.slow_rms = 0.90 * self.slow_rms + 0.10 * feat.rms
            self.db_hist.append(feat.rms_db)
            if len(self.db_hist) > 80:
                self.db_hist = self.db_hist[-80:]
        self.noise_rms = self.slow_rms

        energy_ratio = self.fast_rms / (self.slow_rms + 1e-12)
        med_db = float(np.median(self.db_hist[-20:])) if len(self.db_hist) >= 8 else feat.rms_db
        db_jump = float(feat.rms_db - med_db)

        k_score = np.clip((feat.kurtosis - 3.5) / 4.0, 0, 1)
        e_score = np.clip((energy_ratio - 1.4) / 3.0, 0, 1)
        f_score = np.clip(feat.spectral_flux / max(IMPULSE_FLUX, 1e-6), 0, 1)
        h_score = np.clip(feat.high_ratio / 0.28, 0, 1)
        j_score = np.clip(db_jump / max(IMPULSE_DB_JUMP, 1e-6), 0, 1)
        speech_penalty = 0.45 * feat.harmonicity * feat.speech_ratio
        raw = 0.26 * e_score + 0.18 * k_score + 0.18 * f_score + 0.14 * h_score + 0.24 * j_score
        probability = float(np.clip(raw - speech_penalty, 0.0, 1.0))

        warmed = self.frame_i > WARMUP_FRAMES
        loud_onset = energy_ratio >= IMPULSE_ENERGY_RATIO or db_jump >= IMPULSE_DB_JUMP
        transient_shape = (
            feat.kurtosis >= IMPULSE_KURTOSIS
            or feat.high_ratio > 0.24
            or (feat.spectral_flux >= IMPULSE_FLUX + 0.06 and db_jump >= IMPULSE_DB_JUMP)
        )
        broadband_crack = feat.high_ratio > 0.07 or feat.spectral_centroid > 2200.0
        harmonic_block = feat.harmonicity > 0.42 and feat.high_ratio < 0.08
        trigger = (
            warmed
            and (not self.active)
            and self.refract == 0
            and loud_onset
            and transient_shape
            and broadband_crack
            and not harmonic_block
            and probability >= 0.40
        )

        if trigger:
            self.active = True
            self.hold = IMPULSE_HOLD_FRAMES
            self.active_for = 0
            self.onset_slow = max(self.slow_rms, 1e-4)
            just_started = True
            self.detection_latency_hops.append(0)
            probability = max(probability, 0.85)
        elif self.active:
            self.active_for += 1
            self.hold -= 1
            still_hot = (feat.rms > 2.4 * self.onset_slow) and (
                feat.kurtosis > 5.0 or feat.spectral_flux > 0.14 or db_jump > 4.0
            )
            if still_hot:
                self.hold = max(self.hold, 1)
            timed_out = self.active_for >= MAX_IMPULSE_FRAMES
            released = self.hold <= 0 and not still_hot
            if timed_out or released:
                self.active = False
                self.refract = IMPULSE_REFRACTORY_FRAMES
                just_ended = True
            probability = max(probability, 0.70 if self.active else probability)
        else:
            if self.refract > 0:
                self.refract -= 1
            if energy_ratio >= IMPULSE_ENERGY_RATIO * 0.8 and feat.harmonicity > 0.5:
                self.false_candidates += 1

        return ImpulseEvent(
            active=self.active,
            probability=probability,
            just_started=just_started,
            just_ended=just_ended,
            hold_frames_left=self.hold,
            energy_ratio=float(energy_ratio),
            threshold=float(IMPULSE_ENERGY_RATIO),
        )
