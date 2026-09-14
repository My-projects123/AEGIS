"""AEGIS-COM configuration.

Honesty labels used throughout the POC:
  REAL / MEASURED  — implemented and computed from audio
  SIMULATED        — synthetic signals standing in for field recordings
  TARGET           — SIH/DRDO goals, not claimed as achieved
  FUTURE           — hardware / field deployment not present today
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "audio_samples"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"

SAMPLE_RATE = 16000
N_FFT = 512
HOP = 256
WIN_LENGTH = 512
FRAME_MS = 1000.0 * HOP / SAMPLE_RATE  # 16 ms analysis hop

# Frequency bands (Hz)
LOW_BAND = (20.0, 400.0)
SPEECH_BAND = (300.0, 3400.0)
HIGH_BAND = (4000.0, 7600.0)

# Impulse detector — tuned so a headroom-preserving synthetic bang fires,
# while harmonic speech plosives are still penalised.
IMPULSE_KURTOSIS = 5.5
IMPULSE_ENERGY_RATIO = 2.6
IMPULSE_FLUX = 0.16
IMPULSE_DB_JUMP = 6.0            # onset in dB vs recent median
IMPULSE_HOLD_FRAMES = 8          # ~128 ms hold after trigger
IMPULSE_REFRACTORY_FRAMES = 5
IMPULSE_END_RATIO = 1.8
ADAPTIVE_NOISE_ALPHA = 0.96

# FSM dwell (frames) — prevents chaotic switching
MIN_DWELL = {
    "SPEECH_PRESERVATION": 12,
    "STATIONARY": 14,
    "DYNAMIC": 12,
    "IMPULSE_PROTECTION": 6,
    "RECOVERY": 16,
}
RECOVERY_FRAMES = 18             # ~288 ms recovery window
CLASS_CONFIDENCE_GATE = 0.38

# Enhancement
HPF_CUTOFF_HZ = 70.0
LIMITER_DB = -1.0
WIENER_FLOOR = 0.05

# Decision-directed a-priori SNR smoothing (Ephraim-Malah style).
# Higher = smoother gains = less musical noise, slightly slower transients.
DD_BETA = 0.92

# Live make-up gain so the cleaned voice stays at a usable listening level.
AGC_TARGET_DBFS = -22.0
AGC_MAX_GAIN_DB = 14.0
# Output ceiling while an impulse is being handled, so the make-up gain cannot
# re-amplify a transient the IMPULSE_PROTECTION mode just knocked down.
IMPULSE_CEILING = 0.45

# Speech-only output gate. The floor is not 0 dB of silence on purpose: a hard
# mute sounds broken on a radio and hides whether the link is still alive.
GATE_FLOOR_DB = -32.0
GATE_ON_THRESHOLD = 0.45
GATE_OFF_THRESHOLD = 0.32      # hysteresis
GATE_HANGOVER_HOPS = 10        # ~160 ms, keeps word endings
FLOOR_WINDOW_HOPS = 156        # ~2.5 s, longer than a spoken phrase
# The DSP speech features can still open the gate alone, but only when very
# confident (0.45 / 0.75 = 0.6), since they read ~0.5 on harmonic noise.
# Swept against "fraction of real speech frames cut": below ~0.75 the blind
# (no reference mic) path starts chopping syllables, which is worse than
# letting some noise through.
DSP_VOTE_WEIGHT = 0.75
# How far above the tracked floor a frame must sit to count as speech. Kept
# small for the same reason: with no reference mic the residual noise sits
# only ~8 dB under the voice, so a wide margin eats weak fricatives.
GATE_FLOOR_MARGIN_DB = 1.5
GATE_FLOOR_SPAN_DB = 5.0

# ref_over: how hard the reference-channel PSD is subtracted (two-mic path).
# gain_floor: minimum spectral gain; keeps a low noise bed instead of gating
#             to silence, which is what causes "underwater" artifacts.
MODE_PARAMS = {
    "SPEECH_PRESERVATION": {
        "oversubtraction": 1.0,
        "noise_alpha": 0.98,
        "speech_protect": 0.85,
        "impulse_gate": 0.0,
        "ai_mix": 0.35,
        "ref_over": 1.6,
        "gain_floor": 0.10,
    },
    "STATIONARY": {
        "oversubtraction": 1.7,
        "noise_alpha": 0.97,
        "speech_protect": 0.55,
        "impulse_gate": 0.0,
        "ai_mix": 0.85,
        "ref_over": 2.4,
        "gain_floor": 0.04,
    },
    "DYNAMIC": {
        "oversubtraction": 1.6,
        "noise_alpha": 0.88,
        "speech_protect": 0.60,
        "impulse_gate": 0.0,
        "ai_mix": 0.75,
        "ref_over": 2.2,
        "gain_floor": 0.05,
    },
    "IMPULSE_PROTECTION": {
        "oversubtraction": 1.1,
        "noise_alpha": 1.0,       # freeze noise PSD
        "speech_protect": 0.90,
        "impulse_gate": 0.97,
        "ai_mix": 0.15,           # do not let AI smear the transient
        "ref_over": 3.0,
        "gain_floor": 0.03,
    },
    "RECOVERY": {
        "oversubtraction": 1.3,
        "noise_alpha": 0.995,     # slow unfreeze
        "speech_protect": 0.80,
        "impulse_gate": 0.35,
        "ai_mix": 0.45,
        "ref_over": 2.0,
        "gain_floor": 0.06,
    },
}

NOISE_CLASSES = (
    "quiet",
    "speech",
    "engine",
    "rotor",
    "wind",
    "siren",
    "impulse",
    "mixed",
)

# SIH target metrics — TARGET, not claimed
TARGETS = {
    "snr_db": 15.0,
    "stoi": 0.85,
    "pesq": 2.5,
}

PROJECT_NAME = "AEGIS-COM"
PROJECT_TITLE = "ADAPTIVE DEFENCE COMMUNICATION SYSTEM"
PROJECT_SUBTITLE = "Event-Aware AI + DSP Speech Enhancement"
PS_ID = "SIH26052"
ORG = "DRDO"
