# Results

`python evaluate.py` writes:

- `eval.json` — REAL measured laptop-POC numbers
- `demo_noisy.wav` / `demo_enhanced.wav` / `demo_clean.wav`

Treat anything not in `eval.json` as unmeasured.

SIH printed targets (SNR > 15 dB, STOI > 0.85, PESQ > 2.5) remain **TARGET** unless this file contains a measured value.
