"""Confidence-gated finite-state adaptive controller.

REAL: this is the novelty mechanism implemented in the POC.

States:
  SPEECH_PRESERVATION
  STATIONARY
  DYNAMIC
  IMPULSE_PROTECTION
  RECOVERY

Impulse has priority. Recovery is mandatory after an impulse so the
enhancer does not smear speech with a contaminated noise estimate.
Low classifier confidence holds the current non-impulse state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.config import CLASS_CONFIDENCE_GATE, MIN_DWELL, RECOVERY_FRAMES
from detection.classifier import ClassResult
from detection.impulse_detector import ImpulseEvent
from detection.acoustic_analyzer import AcousticFeatures

STATES = (
    "SPEECH_PRESERVATION",
    "STATIONARY",
    "DYNAMIC",
    "IMPULSE_PROTECTION",
    "RECOVERY",
)


@dataclass
class ControllerState:
    mode: str
    previous_mode: str
    dwell: int
    recovery_left: int
    reason: str
    noise_class: str
    confidence: float
    severity: float
    speech_prob: float
    impulse_prob: float
    snr_est_db: float
    switched: bool
    history: list[str] = field(default_factory=list)


class AdaptiveController:
    def __init__(self) -> None:
        self.mode = "SPEECH_PRESERVATION"
        self.dwell = 0
        self.recovery_left = 0
        self.pre_impulse_mode = "SPEECH_PRESERVATION"
        self.switches = 0
        self.illegal_hops = 0
        self.history: list[str] = []
        self.rms_buf: list[float] = []
        self.flux_ema = 0.0

    def reset(self) -> None:
        self.__init__()

    def _can_leave(self) -> bool:
        return self.dwell >= MIN_DWELL.get(self.mode, 4)

    def _enter(self, new_mode: str, reason: str) -> str:
        if new_mode != self.mode:
            self.switches += 1
            self.mode = new_mode
            self.dwell = 0
            self.history.append(new_mode)
            if len(self.history) > 4000:
                self.history = self.history[-4000:]
        return reason

    def _scene_mode(self, cls: ClassResult, feat: AcousticFeatures) -> str:
        flux = self.flux_ema
        self.rms_buf.append(feat.rms)
        if len(self.rms_buf) > 16:
            self.rms_buf = self.rms_buf[-16:]
        mod = 0.0
        if len(self.rms_buf) >= 8:
            mu = float(np.mean(self.rms_buf)) + 1e-12
            mod = float(np.std(self.rms_buf) / mu)

        if feat.low_ratio > 0.42 and cls.severity > 0.22:
            speech_ok = False
        else:
            speech_ok = True

        if speech_ok and (cls.label == "quiet" or (feat.speech_prob > 0.58 and cls.severity < 0.28 and flux < 0.10 and feat.low_ratio < 0.40)):
            return "SPEECH_PRESERVATION"
        # Low-frequency rumble without blade-rate modulation → stationary engine
        rumble = (
            feat.low_ratio > 0.34
            and flux < 0.16
            and cls.label not in ("rotor", "mixed", "siren")
        )
        if rumble and cls.label in ("engine", "speech", "wind", "quiet"):
            return "STATIONARY"
        if self.mode == "STATIONARY":
            if cls.label in ("rotor", "mixed", "siren") or (flux > 0.12 and mod > 0.16):
                return "DYNAMIC"
            if speech_ok and feat.speech_prob > 0.62 and cls.severity < 0.24 and feat.low_ratio < 0.35:
                return "SPEECH_PRESERVATION"
            return "STATIONARY"
        if self.mode == "DYNAMIC":
            if flux < 0.07 and mod < 0.08 and cls.label in ("engine", "wind") and cls.severity >= 0.22:
                return "STATIONARY"
            if speech_ok and feat.speech_prob > 0.62 and cls.severity < 0.22 and flux < 0.08 and feat.low_ratio < 0.32:
                return "SPEECH_PRESERVATION"
            return "DYNAMIC"
        if cls.label in ("engine", "wind") and flux < 0.11 and mod < 0.12:
            return "STATIONARY"
        if cls.label in ("rotor", "siren", "mixed") or flux >= 0.11 or mod > 0.12:
            return "DYNAMIC"
        if speech_ok and cls.label == "speech" and cls.severity < 0.40:
            return "SPEECH_PRESERVATION"
        if speech_ok and cls.severity < 0.28 and feat.low_ratio < 0.38:
            return "SPEECH_PRESERVATION"
        return "DYNAMIC" if (flux >= 0.10 or mod > 0.12) else "STATIONARY"

    def step(
        self,
        feat: AcousticFeatures,
        impulse: ImpulseEvent,
        cls: ClassResult,
    ) -> ControllerState:
        prev = self.mode
        reason = "hold"
        switched = False
        self.flux_ema = 0.8 * self.flux_ema + 0.2 * feat.spectral_flux

        # 1. Impulse has absolute priority
        if impulse.just_started or (impulse.active and self.mode != "IMPULSE_PROTECTION"):
            if self.mode not in ("IMPULSE_PROTECTION", "RECOVERY"):
                self.pre_impulse_mode = self.mode
            reason = self._enter("IMPULSE_PROTECTION", "impulse onset")
        elif self.mode == "IMPULSE_PROTECTION":
            if impulse.just_ended or not impulse.active:
                self.recovery_left = RECOVERY_FRAMES
                reason = self._enter("RECOVERY", "impulse ended → recovery")
            else:
                reason = "impulse hold"
        elif self.mode == "RECOVERY":
            self.recovery_left -= 1
            if self.recovery_left <= 0 and self._can_leave():
                target = self._scene_mode(cls, feat)
                # after a blast, prefer speech preservation briefly if scene is quieting
                if cls.severity < 0.32:
                    target = "SPEECH_PRESERVATION"
                reason = self._enter(target, "recovery complete")
            else:
                reason = "recovery hold"
        else:
            target = self._scene_mode(cls, feat)
            gated = cls.confidence < CLASS_CONFIDENCE_GATE and target != self.mode
            if gated:
                reason = f"confidence gate ({cls.confidence:.2f}) — hold {self.mode}"
            elif target != self.mode and self._can_leave():
                reason = self._enter(target, f"scene → {target} ({cls.label})")
            elif target != self.mode:
                reason = "min-dwell hold"
            else:
                reason = "stable"

        self.dwell += 1
        switched = self.mode != prev
        return ControllerState(
            mode=self.mode,
            previous_mode=prev,
            dwell=self.dwell,
            recovery_left=self.recovery_left,
            reason=reason,
            noise_class=cls.label,
            confidence=cls.confidence,
            severity=cls.severity,
            speech_prob=feat.speech_prob,
            impulse_prob=impulse.probability,
            snr_est_db=feat.snr_est_db,
            switched=switched,
            history=list(self.history[-40:]),
        )
