# Dataset and training plan

## What tomorrow uses (POC)

- **SIMULATED** formant speech and defence-*like* noises generated in `backend/synthesizer.py`
- Optional user WAV uploads
- Tiny softmax classifier trained at startup on **synthetic feature vectors** (not spectrogram deep learning)

Do not tell judges you trained DeepFilterNet on DRDO data.

## What the final system should use

**Clean speech (public):**
- LibriSpeech (train-clean-100 / 360)
- VoiceBank-DEMAND speech subset
- Indian languages: MUCS / Kathbath / Shrutilipi if license allows (do not claim until downloaded)

**Noise:**
- Vehicle / rotor / wind: FSD50K, AudioSet tags, DEMAND, MS-SNSD
- Impulses: only with license; never scrape classified recordings
- If DRDO provides a defence library: **that becomes the primary set**

**Pairing:**  
`y = s + α n(+ impulse)` at SNR ∈ {−5, 0, 5, 10, 15, 20} dB.

**Scene types:** stationary, non-stationary, impulsive, mixed (engine+impulse+speech).

**Augmentation:** random gain ±6 dB, mild clipping, convolution with short RIRs, mic frequency tilt, impulse time jitter, two-noise mixes.

## Splits (when you train for real)

- **Train:** 80% speakers, 80% noise files
- **Val:** 10% speakers, unseen noise clips of *seen classes*
- **Test:** 10% speakers + **unseen noise types** + unseen SNR
- Never put the same impulse file in train and test
- Keep a “field-like” test that is mixed scenes only

## Target model (future training)

**Input:** causal STFT magnitude / ERB bands (RNNoise-style) or complex STFT (DeepFilterNet-style), 16 kHz.  
**Output:** gain per band or enhanced waveform.  
**Loss (standard, not invented):**

- SI-SNR:  
  \( s_{target} = \frac{\langle \hat{s}, s\rangle}{\|s\|^2}s,\quad \mathcal{L}_{SI} = -10\log_{10}\frac{\|s_{target}\|^2}{\|\hat{s}-s_{target}\|^2} \)
- Multi-resolution STFT (L1/L2 on mag):  
  \( \mathcal{L}_{MR} = \sum_k \| |STFT_k(\hat{s})| - |STFT_k(s)| \|_1 \)
- Optional PESQ proxy / multi-task VAD

**Event-aware extra (justified, not magic):**  
During labeled impulse frames, add

\( \mathcal{L}_{event} = \lambda_{imp}\|\hat{s}-s\|_1^{imp} + \lambda_{freeze}\| \hat{N}-N_{pre} \|_2 \)

so the net neither reconstructs the bang as speech nor updates noise memory. This is a **training recipe**, not a published law of physics. Do not claim a paper result you did not run.

**Optimizer:** Adam, lr 1e-3 → 1e-4, batch 16–32, early stop on val SI-SNR, 50–80 epochs typical for RNNoise-scale, more for DeepFilterNet.

**POC classifier:** 250 epochs of softmax GD on ~320 synthetic frames — **toy**, labelled as such.
