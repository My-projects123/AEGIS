# Novelty — what we will and will not say

## Existing systems we studied (official / vendor sources)

### 3M PELTOR ComTac VI
- Hearing protection for **steady-state and impulse** noise (ANSI S12.42, impulse tested to high peak SPL).
- Level-dependent environmental listening (talk-through) with compression of loud sounds.
- **Mission Audio Profiles (MAP)**: Comfort, Conversation, Patrol/Patrolling, Observation/Overwatch — **operator-selected** ambient profiles with frequency shaping.
- NIB short-range face-to-face radio-less comms.
- Excellent **hearing protection + situational awareness** product.

### INVISIO RA108 (and successor RA5100)
- Combined **passive + ANR** for continuous high noise (vehicles), up to ~36 dB attenuation (vendor figure) on RA108.
- Hear-through limited to safe levels (<85 dBA on RA108).
- ANR / hear-through **enabled or disabled by the user** (earcup control).
- RA5100: digital ANR for constant noise, hear-through with impulse compression, dual-protection mode. Again **mode selection by the operator / mission**.

### Literature / adjacent tech
- Impulse-robust ANC (M-estimators, DMPFxNLMS) — adapts **filter weights**, not a speech-comms policy with recovery.
- Patents on adaptive gunshot suppression time for **hearing protection**.
- Generic AI denoisers (RNNoise, DeepFilterNet) — **one policy** for all conditions; transients often smear.
- EARMOR Mark4 Voice/Ambient/Recon — **user-selected pickup modes**.

## What is NOT novel (do not claim)

- “AI + DSP”
- “We handle impulse noise and they cannot” (they can, for hearing)
- “We have situational awareness and they do not”
- CNN / LSTM / Transformer / STFT / LMS / dual-mic / Jetson / ONNX as such
- Physical ANC on a laptop

## Exact technical gap (frozen)

Existing tactical headsets **protect the ear** and offer **manual listening profiles**.

They do **not** automatically reconfigure the **radio/intercom speech-enhancement strategy** from simultaneous estimates of:

noise class · severity · speech probability · impulse probability · confidence

including a **mandatory recovery state** so a gunshot-like transient does not poison the noise estimate and destroy the next spoken command.

3M MAP is the closest idea — and it is **not automatic**, and it is for **ambient listening**, not for event-driven comms-path enhancement.

Generic AI noise suppression is the other neighbour — and it is **not event-aware**: one model, one behaviour, poor impulse recovery.

## Exact novelty (frozen)

**Event-Aware Adaptive Speech Protection (EAASP)** implemented as a **confidence-gated finite-state controller** that **routes the same audio through different processing policies**, with impulse freeze + recovery.

The invention is the **mechanism** (sense → decide → change strategy → recover), not the brand of neural net.

## How we keep this honest if a judge knows ComTac

Say:

> “ComTac already protects hearing and already has MAP. We are not replacing the headset. We are adding a software controller on the communication path that switches enhancement policy automatically when the scene changes, including after an impulse. MAP is a user menu. AEGIS-COM is a closed-loop policy switch with recovery.”
