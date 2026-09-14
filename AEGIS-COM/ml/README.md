# TinyML on Raspberry Pi 4/5 (ONNX Runtime)

Pi 1 (2012, 700 MHz) **cannot** do real-time 16 kHz enhancement. Use **Pi 4 or Pi 5**.

```
PyTorch (laptop train) → ONNX → INT8 → copy .onnx to Pi → ONNX Runtime → website
```

## Train once on a laptop

```bash
cd sih26052_poc
source .venv/bin/activate
pip install torch onnx onnxruntime
python ml/train_tiny.py
```

Writes `ml/models/tiny_mask.onnx` and `tiny_mask.int8.onnx`.

## What the net learns

Two heads on one 48-unit GRU (~13k parameters, ~51 KB INT8):

| Head | Output | Used for |
|---|---|---|
| `gain` | 32 band gains | keep speech energy, drop the rest |
| `vad` | 1 probability | is a human talking in this 32 ms frame |

Training data is **SIMULATED**: `ml/speech_sim.py` builds talkers with varied
f0 (85–210 Hz), vocal-tract scaling, voiced/fricative/plosive segments and —
crucially — **real pauses between words**. One clip in six contains no talker at
all, with mask target 0 and VAD target 0; without those the net learns to pass
whatever is loudest instead of learning what speech looks like. Augmentation
covers SNR −5…20 dB, short reverb, mic clipping and level changes.

VAD accuracy on held-out SIMULATED clips is reported in `tiny_mask.meta.json`
and on the website. Recall on **real** voices will be lower, which is why the
runtime gate lets the DSP speech features open it too (`enhancement/speech_gate.py`).

## Laptop website (same as Pi)

```bash
pip install onnxruntime
python serve.py
# http://127.0.0.1:8080  → TinyML panel + live mic
```

## Raspberry Pi 4/5

```bash
sudo apt update
sudo apt install python3-venv python3-pip
git clone <this-repo>
cd sih26052_poc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-pi.txt
# copy ml/models/*.onnx from the laptop if you did not clone them
python serve.py
# on the Pi: http://<pi-ip>:8080
```

Keep the FSM + Wiener DSP. TinyML is mixed in by mode (`ai_mix`). This is digital enhancement, not analog ANC at the ear.

SNR > 15 dB / STOI > 0.85 / PESQ > 2.5 remain **TARGET**, not claimed from this ~13k-parameter net.
