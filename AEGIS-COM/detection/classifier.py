"""Feature-based noise-class estimator.

REAL, explainable rules with a confidence score.
A fragile softmax trainer was removed after it overflowed on synthetic
features — that would have been dishonest ML. Production can replace
this with a model trained on field / DRDO audio without changing the FSM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.config import NOISE_CLASSES
from detection.acoustic_analyzer import AcousticFeatures


@dataclass
class ClassResult:
    label: str
    confidence: float
    probs: dict[str, float]
    severity: float  # 0 quiet … 1 severe


class AcousticClassifier:
    def __init__(self, seed: int = 0) -> None:
        self.classes = list(NOISE_CLASSES)
        self.ready = True
        self.seed = seed

    def predict_features(self, feat: AcousticFeatures, impulse_prob: float) -> ClassResult:
        scores = {c: 0.02 for c in self.classes}

        if feat.rms_db < -42:
            scores["quiet"] += 1.4
        if feat.harmonicity > 0.35 and feat.speech_ratio > 0.35 and feat.kurtosis < 8:
            scores["speech"] += 0.9 + 0.6 * feat.harmonicity
        if feat.low_ratio > 0.45 and feat.spectral_centroid < 900 and feat.spectral_flux < 0.16:
            scores["engine"] += 1.1 + 0.4 * feat.low_ratio
        if feat.low_ratio > 0.35 and feat.spectral_flux >= 0.06:
            scores["rotor"] += 0.9 + 0.8 * feat.spectral_flux
            scores["mixed"] += 0.8
        if feat.flatness > 0.35 and feat.spectral_centroid > 800 and feat.harmonicity < 0.25:
            scores["wind"] += 0.8
        if 0.12 < feat.spectral_flux < 0.5 and 600 < feat.spectral_centroid < 3500 and feat.harmonicity > 0.2:
            scores["siren"] += 0.45
        if impulse_prob > 0.45 or feat.kurtosis > 8 or feat.high_ratio > 0.3:
            scores["impulse"] += 1.2 * max(impulse_prob, 0.4)
        if feat.spectral_flux > 0.12 and feat.low_ratio > 0.25 and feat.speech_ratio > 0.2:
            scores["mixed"] += 0.9

        # impulse detector override — controller still owns the mode
        scores["impulse"] = max(scores["impulse"], 2.2 * impulse_prob)

        arr = np.array([scores[c] for c in self.classes], dtype=np.float64)
        arr = np.exp(arr - np.max(arr))
        probs = arr / (arr.sum() + 1e-12)
        k = int(np.argmax(probs))
        label = self.classes[k]
        conf = float(probs[k])
        # Severity = how hostile the noise is, not how loud the talker is.
        snr = feat.snr_est_db
        severity = float(np.clip((8.0 - snr) / 20.0, 0.0, 1.0))
        severity = 0.55 * severity + 0.45 * float(np.clip(feat.low_ratio + 0.5 * feat.spectral_flux, 0, 1))
        if feat.speech_prob > 0.55 and feat.low_ratio < 0.35:
            severity *= 0.45
        if label == "quiet" or feat.rms_db < -40:
            severity = min(severity, 0.15)
        dist = {c: float(p) for c, p in zip(self.classes, probs)}
        return ClassResult(label=label, confidence=conf, probs=dist, severity=severity)

    def predict(self, vec: np.ndarray, rms: float, impulse_prob: float) -> ClassResult:
        """Back-compat for callers that only have the feature vector."""
        dummy = AcousticFeatures(
            rms=rms,
            rms_db=20 * np.log10(rms + 1e-12),
            zcr=float(vec[1]) if len(vec) > 1 else 0.1,
            spectral_centroid=float(vec[2] * 4000) if len(vec) > 2 else 1000,
            spectral_bandwidth=float(vec[3] * 4000) if len(vec) > 3 else 1000,
            spectral_flux=float(vec[4]) if len(vec) > 4 else 0.1,
            kurtosis=float(vec[5] * 20) if len(vec) > 5 else 3.0,
            low_ratio=float(vec[6]) if len(vec) > 6 else 0.3,
            speech_ratio=float(vec[7]) if len(vec) > 7 else 0.4,
            high_ratio=float(vec[8]) if len(vec) > 8 else 0.2,
            harmonicity=float(vec[9]) if len(vec) > 9 else 0.2,
            flatness=float(vec[10]) if len(vec) > 10 else 0.2,
            snr_est_db=float(vec[11] * 30) if len(vec) > 11 else 0.0,
            speech_prob=0.5,
            vector=np.asarray(vec, dtype=np.float64),
        )
        return self.predict_features(dummy, impulse_prob)
