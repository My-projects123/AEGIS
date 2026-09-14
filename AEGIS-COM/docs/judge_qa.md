# Judge Q&A — memorize the short answers

Prefix if unsure: “In the POC we implemented X; Y is target / future.”

1. **What exactly is novel?**  
A confidence-gated FSM that changes speech-enhancement *policy* from acoustic events, with mandatory impulse freeze and recovery. Not “we used AI.”

2. **How is this different from 3M PELTOR?**  
ComTac protects hearing and offers operator-selected MAP listening profiles. We automate comms-path enhancement policy. Complementary, not a headset clone.

3. **How is this different from INVISIO?**  
RA108/RA5100: PNR+ANR and user-toggled hear-through. We do not claim better earcups. We add event-aware radio-speech processing they do not document as an automatic multi-state enhancer with recovery.

4. **Isn’t this just AI noise suppression?**  
A single RNNoise/DeepFilterNet pass is one policy. We route *different* policies (including *less* AI during impulses) from a detector+controller.

5. **Why DSP?**  
Causal, cheap, explainable residual filtering; freezes cleanly during impulses; fallback if the NN is unavailable.

6. **Why AI?**  
Stationary/non-stationary defence noise is nonlinear. A tiny causal net (RNNoise-class) is the *target* enhancer inside STATIONARY/DYNAMIC. Tonight DSP is the reliable core; RNNoise is used only if installed.

7. **Why not ANC alone?**  
Feedback/feedforward ANC is for continuous low-frequency energy at the ear. It is too slow / unstable for impulses and does not clean the boom-mic radio path.

8. **Is your POC actually ANC?**  
No. It is digital speech enhancement + adaptive control. Physical ANC is future, on the headset.

9. **Where is your hardware?**  
We do not have it. Internal hackathon constraint. Software-in-the-loop is honest.

10. **Why a laptop?**  
To prove the algorithm tomorrow with real audio, real detector, real FSM, measured SNR.

11. **How will it run on Jetson?**  
Same Python/C pipeline → ONNX for the enhancer, C/FSM for detector+controller, 16 kHz, 32 ms frames. TensorRT later. ESTIMATE: see `docs/edge_deployment.md`.

12. **Unseen noise?**  
Classifier may be wrong; confidence gate holds the last stable mode. Impulse path uses the detector, not the class label. Retrain class model on field data.

13. **Wrong classification?**  
Min-dwell + confidence gate. Worst case: slightly wrong Wiener aggressiveness, not a hard failure. Impulse still protected by the detector.

14. **Speech looks like an impulse?**  
Plosives have harmonicity and speech-band energy; we penalize those. Residual false positives possible — we measure them on clean synthetic speech in `eval.json`.

15. **False positives?**  
Adaptive threshold (median+MAD), harmonicity penalty, refractory period, min hold. Trade-off: missed quiet impulses vs chopped speech.

16. **How fast is the detector?**  
Same 16 ms hop in the POC (REAL, analysis granularity). Not a 1 µs analog clipper.

17. **After a gunshot?**  
IMPULSE_PROTECTION → RECOVERY (noise PSD frozen/slow) → scene mode. That is the point.

18. **How do you preserve speech?**  
Speech-band gain floor, lighter oversubtraction in SPEECH_PRESERVATION/RECOVERY, low AI mix during impulse.

19. **How do you measure improvement?**  
SNR and SI-SNR vs clean reference (REAL on synthetic pairs). STOI/PESQ if libraries present. Laptop RTF. Impulse peak reduction. Do not quote SIH targets as achieved.

20. **What dataset?**  
Tomorrow: synthetic pairs + optional user WAVs. Training set for the tiny classifier is synthetic features generated at startup. Field data is future (LibriSpeech + DEMAND/DNS + defence noise if DRDO provides).

21. **Why those SNR levels?**  
Standard SE grid covering unintelligible to easy: −5 to 20 dB, matching the problem statement.

22. **Why this AI model?**  
Target: RNNoise-class (tiny, causal). POC: Wiener so the demo cannot fail on missing weights. DeepFilterNet is a future quality option, not tonight’s dependency.

23. **Why not DeepFilterNet?**  
Better quality, more RAM/PyTorch. Fine on Jetson-class later; fragile for a morning internal demo.

24. **Why not RNNoise only?**  
RNNoise is not event-aware. We may *use* it inside modes; we do not *replace* the controller with it.

25. **Computational complexity?**  
Per hop: 512-pt FFT + features + 8-class matvec + Wiener multiply. ESTIMATE tens of MFLOP/s — laptop RTF is the measured number.

26. **Latency?**  
POC: measured wall-clock RTF (batch). Streaming budget ESTIMATE 32–48 ms algorithmic + I/O.

27. **Power?**  
Not measured. FUTURE on target SoC. DSP+tiny NN is the power story, not a large transformer.

28. **Offline?**  
Yes. No cloud.

29. **Without internet?**  
Yes, after pip install.

30. **Next step?**  
Port to USB audio I/O, then Pi/Jetson, then dual-mic reference, then headset ANR integration, then DRDO noise library.

31. **Actual contribution?**  
The event-aware controller + recovery policy, implemented and demonstrated.

32. **Implemented today?**  
Detector, FSM, mode-wise DSP, dashboard, SNR/SI-SNR, synthetic scene.

33. **Future?**  
Headset, ANR, multi-mic, field STOI/PESQ, RNNoise/DeepFilterNet fully integrated, quantization.

34. **Real defence validation?**  
Anechoic + range recordings with IRB/safety, boom mic on intercom, AB listening tests, ITU-T intelligibility, impulse insertion loss on the *ear* path separately.

35. **Engine + gunshot + speech together?**  
Impulse state wins; speech protect stays on; after recovery, stationary/dynamic resumes. That mixed case is the demo.

36. **How does it recover?**  
Fixed recovery dwell, frozen/slow noise PSD, then confidence-gated return.

37. **Reference microphone?**  
FUTURE for ANC/secondary path. POC is single-channel comms audio.

38. **Mic failure?**  
FUTURE: detect silence/clipping, fall back to pass-through / last mode, never open-loop dangerous gain. Not implemented.

39. **Safe for military use?**  
No. Laboratory POC. Hearing protection must remain certified ANR/passive hardware. We must not amplify transients.

40. **Why should government use this?**  
Because MAP/ANR already exist, but operators still lose radio words when the scene jumps. A software controller on Indian comms stacks can be updated without replacing every earcup.

41. **LMS?**  
Team still uses LMS/FxLMS *in the future earcup ANC*. POC does not run FxLMS (no secondary path). Say that clearly.

42. **Dual microphone?**  
Useful later (speech vs noise ref). Not required to prove the controller.

43. **Quantization / TensorRT?**  
Deployment plan, not done.

44. **Can you show the code of the FSM?**  
Yes — `controller/adaptive_controller.py`. Offer to open it.

If they show the baseline table: “A always-on Wiener can beat us on raw SNR because it never goes gentle. We beat it on SI-SNR here, and we add impulse freeze plus recovery, which that baseline does not have.”
