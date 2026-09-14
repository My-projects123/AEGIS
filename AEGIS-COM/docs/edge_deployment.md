# Edge deployment (FUTURE — estimates)

The POC is a laptop SIL. Same *software roles* map to hardware as follows.

```
Voice boom mic ──┐
Env / ref mic ───┼─→ codec (16–48 kHz) → Event detector (C, <1 ms/hop ESTIMATE)
                 │                         Adaptive controller (FSM, negligible)
                 │                         AI enhancement (RNNoise / ONNX)
                 │                         Residual DSP / limiter
                 └── ANR earcup loop (analog or FxLMS)  → loudspeaker
                              ↑
                    AEGIS-COM does not replace this loop
Radio / intercom TX ← enhanced boom-mic stream
```

## Latency budget (ESTIMATE, not measured on Jetson)

| Stage | Budget |
|---|---|
| Codec / DMA | 2–5 ms |
| 32 ms window, 16 ms hop (algorithmic) | 16–32 ms |
| Features + detector + FSM | <1 ms |
| RNNoise-class infer | 2–8 ms on small CPU / <2 ms NPU |
| I/O / jitter | 5 ms |
| **Total conversational target** | **< 60 ms** |

ComTac-class analog clipping of *ear* impulses can still be faster than this; that is hearing protection, a different path.

## Implementation notes

- Sample rate 16 kHz (radio) or 48 kHz downsample
- Export enhancer to **ONNX**, INT8 quantize, TensorRT on Jetson, TFLite/NNAPI on Pi-class
- Keep FSM and detector in C — not worth a GPU
- Power: ESTIMATE low hundreds of mW for RNNoise-class on a modern SoC; **not measured**
- Pi 4: feasible for RNNoise + FSM. Jetson Nano/Orin: DeepFilterNet possible
- Dual-mic: future beamform / ANC reference; controller stays the same

## TinyML path that is implemented (REAL export)

```
PyTorch Tiny GRU (~13k params: 32-band mask head + VAD head)
    ↓
ONNX (ml/models/tiny_mask.onnx)
    ↓
INT8 dynamic quantize (tiny_mask.int8.onnx)
    ↓
Raspberry Pi 4/5  (not Pi 1)
    ↓
ONNX Runtime CPU  → website http://<host>:8080
```

Train: `python ml/train_tiny.py`  
Run: `python serve.py` (laptop or Pi). The website TinyML panel reports params, file size, and runtime.

**Raspberry Pi 1 is not a real-time target.** The same ONNX file is meant for Pi 4/5.

SNR>15 / STOI>0.85 / PESQ>2.5 stay **TARGET**. The tiny net is for edge feasibility, not those scores.

## What to say

“We trained a tiny GRU on simulated pairs, exported ONNX, quantized INT8, and run ONNX Runtime on the laptop now. The identical artifact goes to a Raspberry Pi 4/5. Jetson/TensorRT and analog ANC are future.”

