# AEGIS-COM architecture

**Frozen solution for SIH26052 (DRDO).** Do not change this story tomorrow.

## One-line concept

AEGIS-COM is an **event-aware adaptive speech-protection controller** for the **communication path**. It estimates the acoustic condition every 16 ms and **changes processing strategy** so radio speech stays intelligible through stationary noise, changing noise, and impulsive events — then **recovers** without smearing the next syllables.

## What this POC is / is not

| Claim | Label |
|---|---|
| Digital speech enhancement on a laptop | **REAL / Implemented** |
| Impulse detector + FSM mode switching | **REAL / Implemented** |
| NLMS two-mic adaptive canceller | **REAL / Implemented** (reference channel is SIMULATED) |
| TinyML GRU: 32-band mask + VAD head, INT8 ONNX | **REAL / Implemented** (trained on SIMULATED talkers) |
| Speech-only output gate (only the human voice is passed) | **REAL / Implemented** |
| Measured SNR, SI-SNR, laptop RTF | **REAL / Measured** |
| Physical analog ANC / destructive interference at the ear | **NOT this POC** |
| Jetson / Raspberry Pi / tactical headset | **FUTURE** |
| STOI > 0.85, PESQ > 2.5, SNR > 15 dB | **TARGET** (use measured values if `results/eval.json` has them) |

## Why not physical ANC tomorrow

ANC (FxLMS / analog ANR) needs a reference microphone, error microphone, loudspeaker, and secondary-path model. We do not have that hardware. Claiming ANC would be dishonest.

**Final product mapping:** analog/digital ANR remains on the earcup for continuous low-frequency vehicle/rotor noise. AEGIS-COM sits on the **boom-mic / radio / intercom** path (and can later also drive hear-through gain). The laptop POC proves the **controller + enhancement policy**, which is the software novelty.

## Pipeline (POC = this exact chain)

```
Primary mic (voice + h·disturbance)        Reference mic (disturbance)
    → high-pass / framing (32 ms window, 16 ms hop)
    → Acoustic intelligence            ← runs on the PRIMARY, so impulses are
         features, noise class, severity,   seen before anything is cancelled
         speech P, impulse P, confidence, SNR heuristic
    → Impulse detector (energy ratio + kurtosis + flux + debounce)
    → Confidence-gated FSM controller
    → NLMS adaptive canceller  e = primary − w·reference
         adaptation slowed while speech is present (double-talk),
         frozen during IMPULSE_PROTECTION
    → Residual estimate ρ (minimum statistics over ~1 s)
    → Mode-specific enhancement on e (decision-directed Wiener
         × TinyML GRU mask, both scaled by how much residual ρ reports)
    → Speech-only gate: GRU VAD head leads, DSP speech features override
         only when strongly confident; both read the *enhanced* signal and
         must clear the tracked noise floor
    → Residual DSP (HPF + limiter) + live make-up AGC + transient ceiling
    → Enhanced speech + metrics
```

**Why a gate at all.** The mask suppresses noise *under* the voice, but between
words there is no voice to protect and no reason to pass anything at all. The
gate is what turns "quieter noise" into "only the talker": with no talker
present the output drops ~86 dB, and in pauses between words it sits 16–30 dB
below the speech with a reference mic, about 8 dB without one — the blind path
has a louder residual for the gate to work against, so the contrast is smaller.
About 3–4% of true speech frames are cut, mostly weak fricatives; the tuning
deliberately favours that over chopping syllables.

Four details make it usable rather than annoying:

- Both votes read the **enhanced** signal, never the raw mic, where the DSP
  features read "speech" almost permanently because engine and rotor noise are
  harmonic too. The window is pre-gate, or a closed gate would keep itself shut.
- The learned VAD **leads** and the DSP vote is de-weighted (×0.75), so it only
  overrides when strongly confident. An equal vote holds the gate open on any
  harmonic noise; no vote at all would cut real talkers the net cannot
  recognise, since it has only ever heard simulated speech.
- Speech-likeness alone cannot open the gate. Near silence still scores ~0.35,
  which is above the release threshold, so the gate would latch open forever;
  the frame must also sit above the noise floor, tracked as a minimum over a
  sliding ~2.5 s window. An all-time minimum silently disables the test — one
  quiet moment at start-up pins the floor and everything later looks loud.
- An impulse may not **open** the gate (a gunshot is loud, broadband and
  speech-shaped enough to fool both votes) but also cannot **close** an open one
  — muting the operator mid-word is worse than a loud bang, which
  IMPULSE_PROTECTION and the output ceiling already handle.

**Why two stages.** A spectral mask can only scale magnitudes, so noise sitting
on top of the voice band cannot be removed without removing voice with it. The
NLMS stage subtracts the disturbance *waveform* using the reference channel,
which leaves the voice untouched; the spectral/AI stage then only has to clean a
small residual — and it is told to work proportionally less hard when ρ says the
residual is already small.

In this POC the reference channel is **SIMULATED**: the same synthetic
disturbance reaches the primary mic through a different synthetic path
(`backend/live_mix.PRIMARY_PATH`) plus reference sensor noise, so the filter has
to learn that path. On a real headset this is the environment mic.

## Modes (locked)

| Mode | When | What processing does |
|---|---|---|
| SPEECH_PRESERVATION | Quiet / speech-dominated | Light Wiener, protect consonants |
| STATIONARY | Engine-like, low flux | Stronger, slow noise estimate |
| DYNAMIC | Rotor / mixed / high flux | Faster noise update, moderate subtraction |
| IMPULSE_PROTECTION | Detector trigger | Freeze noise PSD, gate transient, **do not** let AI smear the event |
| RECOVERY | After impulse ends | Mandatory hold, slow unfreeze, then return to scene mode |

## Controller rules (locked)

1. Impulse has priority over every other state.
2. Every impulse is followed by RECOVERY (cannot skip).
3. Minimum dwell times stop chatter.
4. If class confidence < 0.42, do not change non-impulse mode.

## Model choice (locked)

| Candidate | Why not for tonight | Role in story |
|---|---|---|
| RNNoise (~60k params, RTF ~0.001) | Optional; C wrapper may fail to install | **Target edge AI** if it loads |
| DeepFilterNet3 (~2.3M, better PESQ) | PyTorch + weights too heavy/fragile tonight | **Future** quality upgrade |
| DCCRN / Conv-TasNet | Too much compute for radio-class edge | Not used |
| **POC primary** | Causal Wiener + speech-band protect + impulse gate | **REAL tonight** |

Judges: “We selected RNNoise-class models for the edge target because they are causal and tiny. Tonight the POC proves the **controller**. The enhancer is swappable.”

## Frame parameters (POC)

- Sample rate: **16 kHz**
- Window: **512 samples (32 ms)**
- Hop: **256 samples (16 ms)**
- Algorithmic latency (future streaming): ~32 ms hop+window overlap ≈ **32–48 ms** (ESTIMATE)
- Laptop RTF: **measured** in `eval.json` (wall-clock, not a DSP chip)

## Files

- `detection/` acoustic features, impulse detector, synthetic-trained softmax classifier
- `controller/adaptive_controller.py` FSM
- `enhancement/` Wiener + optional RNNoise mix
- `backend/processor.py` full pipeline
- `app.py` Streamlit dashboard
