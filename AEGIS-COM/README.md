# AEGIS-COM — SIH26052 POC

**Adaptive Defence Communication System**  
Event-Aware AI + DSP Speech Enhancement  
Problem: SIH26052 · Organisation: DRDO

This is **one frozen solution**. PPT, code, and spoken story describe the same system.

## What we built

A **laptop software-in-the-loop** proof of concept that:

1. Analyses audio every 16 ms  
2. Detects impulsive events  
3. Switches processing mode with a finite-state controller  
4. Enhances speech with real DSP (optional RNNoise if installed)  
5. Shows before/after audio, spectrograms, and **measured** SNR  

This is **not** physical analog ANC and **not** a tactical headset.

## Run tomorrow

```bash
cd sih26052_poc
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate_samples.py
python evaluate.py                 # writes results/eval.json (REAL numbers)
python generate_ppt.py             # 6-slide deck in docs/
streamlit run app.py
```

Optional CLI:

```bash
python run_demo.py
```

Optional API:

```bash
uvicorn backend.api:app --reload --port 8000
```

Then click **START DEMO** in the browser.

## Website dashboard (recommended for judges)

```bash
source .venv/bin/activate
python serve.py
```

Open **http://127.0.0.1:8080**

Click **USE MY MICROPHONE**, allow the browser prompt, and speak. Leave **Mix simulated engine on my voice** checked to hear rumble drop. The gold **Disturbance reduced** meter is non-speech-band energy in vs out on each 16 ms hop.

You can also **Upload WAV** of your own recording and click **PROCESS MIX**.

## Frozen novelty (one sentence)

Existing headsets protect hearing and offer **manual** listening profiles. AEGIS-COM **automatically** changes the **communication-path** enhancement policy from the acoustic event, including **impulse freeze + recovery**.

Read `docs/novelty.md` before you improvise.

## Honesty strip

| Label | Meaning |
|---|---|
| REAL / MEASURED | Code ran; number came from audio or a clock |
| SIMULATED | Synthetic engine/rotor/impulse — not field recordings |
| TARGET | SIH printed goals (SNR>15, STOI>0.85, PESQ>2.5) |
| FUTURE | Jetson, Pi, DSP, headset, dual-mic ANC |

## Measured on the synthetic demo scene (laptop)

From `python evaluate.py` — **do not replace these with SIH targets**:

| Quantity | Value | Label |
|---|---|---|
| Input SNR | ~0.2 dB | REAL (impulse dominates the mix) |
| Output SNR | ~2.1 dB | REAL |
| SNR improvement | ~+1.9 dB | REAL |
| SI-SNR | ~5.4 → ~8.1 dB | REAL (this is the speech-shaped score) |
| STOI | ~0.78 → ~0.79 | REAL (held, not a miracle) |
| PESQ | not installed | TARGET 2.5 |
| Impulse peak reduction | ~11 dB | REAL |
| Detect latency | same 16 ms hop | REAL |
| Recovery time | ~416 ms | REAL |
| Laptop RTF | ~0.011 | REAL (faster than real time on CPU) |
| Physical ANC / Jetson | — | NOT DONE |

Fixed always-on Wiener can look *better on raw SNR* because it always subtracts hard. On this scene AEGIS-COM is better on **SI-SNR**. The novelty you demonstrate is the **mode timeline**, not a 15 dB miracle.

## Study tonight

1. `docs/study_tonight.md`  
2. `docs/demo.md` (2-minute script)  
3. `docs/pitch.md`  
4. `docs/judge_qa.md`  
5. `docs/slides.md` + `docs/AEGIS-COM_SIH26052.pptx`

## Layout

```
sih26052_poc/
  app.py                 Streamlit dashboard
  run_demo.py            CLI
  evaluate.py            Real metrics
  generate_ppt.py        6 slides
  backend/               pipeline, mixer, metrics
  detection/             features, impulse, classifier
  controller/            FSM
  enhancement/           Wiener + optional RNNoise
  visualization/         plots
  frontend/              theme
  audio_samples/         synthetic WAVs
  results/               eval.json
  docs/                  architecture, novelty, Q&A
```
