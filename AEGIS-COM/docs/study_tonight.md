# STUDY TONIGHT — AEGIS-COM from zero

Read in order. Each item: simple → technical → example → what to tell a judge.

## 1. What is ANC?
- **Simple:** Headphones make an opposite sound to cancel a constant drone (engine, fan).
- **Technical:** Feedforward/feedback adaptive filters (often FxLMS) drive a loudspeaker so that at the eardrum the residual is small. Needs reference + error mics and a secondary-path model. Works best on **low-frequency, slowly changing** noise.
- **Example:** Vehicle hum reduced in an ANR earcup.
- **Judge:** “ANC is the earcup loop for continuous rumble. Our laptop POC is not that loop.”

## 2. What is speech enhancement?
- **Simple:** Clean the voice already captured by a microphone.
- **Technical:** Estimate a mask or filter in STFT/time domain to suppress noise in a speech recording or radio chain.
- **Example:** Noisy boom-mic → clearer radio TX.
- **Judge:** “Enhancement is the comms-path problem SIH also cares about: intelligibility.”

## 3. ANC vs speech enhancement
- **Simple:** ANC cancels sound in the air at the ear. Enhancement cleans a signal in software/DSP.
- **Technical:** ANC is electro-acoustic closed loop. SE is a signal estimator. Metrics differ (insertion loss vs STOI/PESQ/SI-SNR).
- **Example:** You can have great ANR and still transmit muddy radio speech.
- **Judge:** “We target the radio path today; ANR stays on the headset tomorrow.”

## 4. Stationary noise
- **Simple:** Sounds that stay similar (engine idle).
- **Technical:** Statistics (PSD) change slowly; Wiener/spectral subtraction work.
- **Example:** Our STATIONARY mode.
- **Judge:** “Slow noise estimate, stronger suppression.”

## 5. Non-stationary noise
- **Simple:** Sounds that keep changing (rotor, siren, passing vehicle).
- **Technical:** PSD tracking must be faster; long averages lag.
- **Example:** DYNAMIC mode.
- **Judge:** “Faster update, less aggressive than a slow engine filter.”

## 6. Impulsive noise
- **Simple:** A sudden bang shorter than a spoken syllable.
- **Technical:** High kurtosis, high peak, broadband, milliseconds. Adaptive filters diverge; AI denoisers smear.
- **Example:** Synthetic gunshot-like burst at t = 8 s in the demo.
- **Judge:** “We detect, freeze, gate, then recover.”

## 7. Why gunshots are difficult
- **Simple:** Too fast for normal filters; too much like a speech ‘p’ if you are careless.
- **Technical:** Rise time << ANC group delay; energy leaks into all bands; aftershock corrupts noise PSD.
- **Example:** Without freeze, Wiener thinks the bang is the new noise floor and eats the next word.
- **Judge:** “Hearing clippers exist in ComTac. Our problem is the *next radio word* after the bang.”

## 8. SNR
- **Simple:** How much louder speech is than noise, in dB.
- **Technical:** 10 log10(P_speech / P_noise). We also report **SI-SNR** (scale-invariant).
- **Example:** Mix at 5 dB, measure output SNR vs the clean reference.
- **Judge:** Quote **measured** dashboard numbers only. “>15 dB” is the SIH **target**.

## 9. STOI
- **Simple:** A score of “can you understand the words,” 0 to 1.
- **Technical:** Short-Time Objective Intelligibility vs a clean reference.
- **Example:** 0.85 is the SIH target.
- **Judge:** If dashboard shows n/a, say unavailable tonight, target 0.85.

## 10. PESQ
- **Simple:** A score of how pleasant/clear the call sounds (~1–4.5).
- **Technical:** ITU-T P.862, needs reference. Installation is picky.
- **Example:** Target 2.5.
- **Judge:** Same honesty as STOI.

## 11. STFT
- **Simple:** Slice sound into short windows and look at frequencies.
- **Technical:** Hann window 512, hop 256, FFT → magnitude/phase.
- **Example:** Spectrograms on the dashboard.
- **Judge:** “All our DSP runs in STFT frames of 16 ms hops.”

## 12. Spectral masking
- **Simple:** Turn down frequency bins that look like noise.
- **Technical:** Gain G(f) ∈ [floor, 1] applied to |Y(f)|.
- **Example:** Wiener gain in the enhancer.
- **Judge:** “The mask is *stronger or weaker* depending on mode.”

## 13. Adaptive filtering
- **Simple:** A filter that keeps updating its coefficients.
- **Technical:** LMS/NLMS/FxLMS minimize error online.
- **Example:** Headset ANR — future, not the POC loop.
- **Judge:** “Team architecture still includes LMS ANR on the ear. POC shows the other half: policy switching.”

## 14. LMS
- **Simple:** Tiny rule: change weights a little toward less error.
- **Technical:** w ← w + μ x e. FxLMS filters x by the secondary path.
- **Example:** Cancelling a 100 Hz rumble.
- **Judge:** “LMS fails on impulses because e is huge and μ is wrong. That is textbook; we do not pretend we ran it today.”

## 15. AI speech enhancement
- **Simple:** A small neural net that learned “this bin is speech / noise.”
- **Technical:** RNNoise GRU on bark bands; DeepFilterNet deep filtering; DCCRN complex CRN.
- **Example:** Target = RNNoise inside STATIONARY/DYNAMIC.
- **Judge:** “AI is a *module inside modes*, not the novelty.”

## 16. What is our novelty?
- **Simple:** The system *changes its plan* when the event changes, then *recovers*.
- **Technical:** EAASP = acoustic intelligence + confidence-gated FSM + mode-specific SE.
- **Example:** Engine → DYNAMIC mix → IMPULSE → RECOVERY.
- **Judge:** Use the ComTac MAP contrast: they select a profile; we close the loop.

## 17. What does the adaptive controller do?
- **Simple:** A traffic cop for audio algorithms.
- **Technical:** States, min dwell, confidence gate, impulse priority, recovery timer.
- **Example:** File `controller/adaptive_controller.py`.
- **Judge:** “If you want novelty in one file, it is that file.”

## 18. What happens when an impulse occurs?
- **Simple:** Detect bang → protection mode → don’t update noise memory → shrink the peak.
- **Technical:** Energy ratio vs adaptive floor + kurtosis + flux; hold; freeze α=1.
- **Example:** White dashed line on the timeline.
- **Judge:** Walk the four bullets without saying “AI ON.”

## 19. Why recovery matters
- **Simple:** After a bang the computer is still ‘shocked’. If you denoise immediately, you damage speech.
- **Technical:** Contaminated PSD; release of the gate; dwell before STATIONARY/DYNAMIC.
- **Example:** RECOVERY gold band after red impulse band.
- **Judge:** “Generic denoisers skip this state. That is the intelligibility gap.”

## 20. Why edge deployment matters
- **Simple:** Radios cannot wait for a cloud GPU and cannot die on battery.
- **Technical:** Causal models, 16 kHz, INT8/ONNX, RTF < 1 on SoC.
- **Example:** RNNoise ~0.06 M params vs huge transformers.
- **Judge:** “Laptop measures RTF. Jetson is the same software, not a new invention.”

## Extra vocabulary
- **Boom mic:** mic on the headset arm, radio TX.
- **Hear-through / talk-through:** play outside sound inside the cup, compressed.
- **MAP:** 3M Mission Audio Profiles.
- **RTF:** processing time / audio time. <1 means faster than real time on that CPU.
- **Software-in-the-loop:** algorithm runs on recorded/synthetic audio, no plant/hardware.
