"""Speech-only output gate.

The mask cleans the spectrum; this closes the channel when nobody is talking,
so the output carries the operator's voice and (almost) nothing else.

Two detectors vote:

  tiny GRU VAD   learned, trained on SIMULATED talkers (ml/train_tiny)
  DSP speech_prob  harmonicity / speech-band ratio / ZCR (detection/)

The learned head leads and can open the gate on its own. The DSP vote is
de-weighted, so it only overrides when it is *strongly* confident. That
asymmetry is the point: the DSP features sit near 0.5 on noise that happens to
be harmonic, which is most vehicle noise, so an equal vote would hold the gate
open permanently — but the learned head has never heard a real voice, so
removing its safety net entirely would cut real talkers the net does not
recognise. Strong DSP evidence still opens the gate.

Hysteresis plus a hangover keeps word endings and breaths: closing hard on
every inter-syllable dip is what makes gated audio sound chopped.
"""

from __future__ import annotations

import numpy as np

from backend.config import (
    DSP_VOTE_WEIGHT,
    FLOOR_WINDOW_HOPS,
    GATE_FLOOR_MARGIN_DB,
    GATE_FLOOR_SPAN_DB,
    GATE_FLOOR_DB,
    GATE_HANGOVER_HOPS,
    GATE_OFF_THRESHOLD,
    GATE_ON_THRESHOLD,
)


class SpeechGate:
    def __init__(
        self,
        floor_db: float = GATE_FLOOR_DB,
        on_threshold: float = GATE_ON_THRESHOLD,
        off_threshold: float = GATE_OFF_THRESHOLD,
        hangover_hops: int = GATE_HANGOVER_HOPS,
        attack: float = 0.5,
        release: float = 0.15,
        floor_margin_db: float = GATE_FLOOR_MARGIN_DB,
        floor_span_db: float = GATE_FLOOR_SPAN_DB,
        dsp_weight: float = DSP_VOTE_WEIGHT,
    ) -> None:
        self.floor_margin_db = floor_margin_db
        self.floor_span_db = floor_span_db
        self.dsp_weight = dsp_weight
        self.floor = 10.0 ** (floor_db / 20.0)
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.hangover_hops = int(hangover_hops)
        self.attack = attack        # fast open, so onsets are not clipped
        # Close over ~150 ms. The hangover already protects word endings, so a
        # slower ramp only means short pauses never reach the floor.
        self.release = release
        self.gain = 1.0
        self.open = False
        self.hold = 0
        self.confidence = 0.0
        self.floor_db = -70.0
        self._levels: list[float] = []

    def reset(self) -> None:
        self.gain = 1.0
        self.open = False
        self.hold = 0
        self.confidence = 0.0
        self.floor_db = -70.0
        self._levels = []

    def step(
        self,
        dsp_speech_prob: float,
        tiny_vad: float | None,
        level_db: float | None = None,
        impulse_prob: float = 0.0,
    ) -> float:
        """Advance the gate one hop and return its linear gain.

        `level_db` is the hop level after cancellation. Speech-likeness alone is
        not enough to open the gate: the "is it speech-shaped?" score of near
        silence sits around 0.35, which is above the release threshold, so the
        gate would latch open forever. Requiring the frame to also sit above the
        tracked residual floor is what makes pauses actually go quiet.
        """
        if tiny_vad is None:
            conf = float(dsp_speech_prob)
        else:
            conf = max(float(tiny_vad), self.dsp_weight * float(dsp_speech_prob))
        # A gunshot is loud, broadband and speech-shaped enough to fool both
        # votes, so an impulse may not *open* the gate. It cannot close an
        # already-open one either: that would mute the operator mid-word, which
        # is what IMPULSE_PROTECTION and the output ceiling are for.
        conf *= float(np.clip(1.0 - impulse_prob, 0.0, 1.0))

        if level_db is not None:
            # Minimum statistics over a sliding window, not an all-time minimum:
            # a single quiet moment at start-up would otherwise pin the floor
            # forever and every later frame would look "above floor", which
            # silently disables this whole test. The window is longer than a
            # spoken phrase, so it still finds an inter-word dip to sit on.
            self._levels.append(float(level_db))
            if len(self._levels) > FLOOR_WINDOW_HOPS:
                self._levels.pop(0)
            self.floor_db = min(self._levels)
            above = level_db - self.floor_db
            conf *= float(np.clip((above - self.floor_margin_db) / self.floor_span_db, 0.0, 1.0))

        self.confidence = conf

        if self.open:
            if conf < self.off_threshold:
                self.hold -= 1
                if self.hold <= 0:
                    self.open = False
            else:
                self.hold = self.hangover_hops
        elif conf >= self.on_threshold:
            self.open = True
            self.hold = self.hangover_hops

        target = 1.0 if self.open else self.floor
        rate = self.attack if target > self.gain else self.release
        self.gain += rate * (target - self.gain)
        return self.gain

    def apply(self, hop: np.ndarray) -> np.ndarray:
        return hop * self.gain

    def gain_db(self) -> float:
        return float(20.0 * np.log10(self.gain + 1e-12))
