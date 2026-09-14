"""Train TinyMaskNet to output speech only, export ONNX, INT8-quantize.

    python ml/train_tiny.py

Two targets per frame:
  mask  ideal ratio mask in 32 bands (keep speech energy, drop the rest)
  vad   is a human talking right now

Roughly one clip in six contains *no speech at all* (noise only), with mask
target 0 and vad target 0. Without those the net learns to pass whatever is
loudest instead of learning what speech looks like.

Honesty: SIMULATED talkers (ml/speech_sim) + engine/rotor/wind/drone/siren/
impulse/broadband noise. Not DRDO field recordings, not LibriSpeech. Tiny model
for Pi 4/5, not Pi 1 real-time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn

from backend.config import HOP, N_FFT, SAMPLE_RATE, WIN_LENGTH
from backend.synthesizer import (
    _butter_filter,
    overlay_impulse,
    synth_drone,
    synth_engine,
    synth_rotor,
    synth_siren,
    synth_wind,
)
from ml.fbanks import filterbank, to_bands
from ml.model import HIDDEN, N_BANDS, TinyMaskNet
from ml.paths import FP32_ONNX, INT8_ONNX, META_JSON, MODELS, WEIGHTS
from ml.speech_sim import synth_talker

EPOCHS = 30
CLIPS = 320
SECONDS = 2.0
BATCH = 16
LR = 2e-3
NOISE_ONLY_EVERY = 6      # 1 clip in 6 has no talker at all
VAD_WEIGHT = 0.5
COMPRESS = 0.3            # power-law magnitude compression for the mask loss

NOISE_KINDS = ("engine", "rotor", "wind", "drone", "siren", "engine_rotor", "broadband")


def _noise(dur: float, kind: str, seed: int) -> np.ndarray:
    if kind == "engine":
        return synth_engine(dur, seed=seed)
    if kind == "rotor":
        return synth_rotor(dur, seed=seed)
    if kind == "wind":
        return synth_wind(dur, seed=seed)
    if kind == "drone":
        return synth_drone(dur, seed=seed)
    if kind == "siren":
        return synth_siren(dur, seed=seed)
    if kind == "broadband":
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(int(dur * SAMPLE_RATE))
        # pink-ish: real rooms and radios are not white
        return _butter_filter(x, 3000.0, SAMPLE_RATE, "lowpass", order=1)
    return 0.55 * synth_engine(dur, seed=seed) + 0.55 * synth_rotor(dur, seed=seed + 1)


def _reverb(x: np.ndarray, rt60_s: float, sr: int, seed: int) -> np.ndarray:
    """Short synthetic room. Applied to the CLEAN target too, so the net is not
    asked to dereverb — only to separate speech from noise."""
    rng = np.random.default_rng(seed)
    n = int(rt60_s * sr)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / max(n, 1))
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
    return np.convolve(x, ir, mode="full")[: len(x)]


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    n = min(len(clean), len(noise))
    c, d = clean[:n], noise[:n]
    c_pow = float(np.mean(c ** 2) + 1e-12)
    d_pow = float(np.mean(d ** 2) + 1e-12)
    scale = np.sqrt(c_pow / (d_pow * (10.0 ** (snr_db / 10.0))))
    return c + scale * d


def make_clip(i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (noisy, clean_target, speech_mask_samples)."""
    rng = np.random.default_rng(1000 + i)
    noise = _noise(SECONDS, NOISE_KINDS[i % len(NOISE_KINDS)], 40 + i)

    if i % NOISE_ONLY_EVERY == 0:
        # no talker: the answer is silence
        n = len(noise)
        noisy = noise * float(rng.uniform(0.15, 0.7)) / (np.max(np.abs(noise)) + 1e-12)
        return noisy.astype(np.float32), np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    clean, mask = synth_talker(SECONDS, sr=SAMPLE_RATE, seed=7 + i)
    if rng.random() < 0.35:
        clean = _reverb(clean, float(rng.uniform(0.12, 0.35)), SAMPLE_RATE, seed=500 + i)

    snr = float(rng.uniform(-5.0, 20.0))
    noisy = _mix_at_snr(clean, noise, snr)
    if rng.random() < 0.2:
        noisy = overlay_impulse(noisy, at_s=float(rng.uniform(0.2, SECONDS - 0.4)), kind="gunshot", seed=90 + i)

    gain = float(rng.uniform(0.25, 1.0))
    noisy = noisy * gain
    clean = clean[: len(noisy)] * gain
    if rng.random() < 0.2:  # mic overload
        lim = float(rng.uniform(0.6, 0.98))
        noisy = np.clip(noisy, -lim, lim)

    n = min(len(clean), len(noisy), len(mask))
    return noisy[:n].astype(np.float32), clean[:n].astype(np.float32), mask[:n].astype(np.float32)


def stft_mag(x: np.ndarray) -> np.ndarray:
    win = np.hanning(WIN_LENGTH)
    if len(x) < WIN_LENGTH:
        x = np.pad(x, (0, WIN_LENGTH - len(x)))
    hops = []
    for start in range(0, len(x) - WIN_LENGTH + 1, HOP):
        spec = np.fft.rfft(x[start : start + WIN_LENGTH] * win, n=N_FFT)
        hops.append(np.abs(spec) + 1e-12)
    return np.stack(hops, axis=0)


def frame_vad(mask: np.ndarray, clean_bands: np.ndarray, n_frames: int) -> np.ndarray:
    """Speech present if the frame is mostly inside a talker segment AND carries
    energy (a segment can be labelled speech but be a plosive closure)."""
    occ = np.zeros(n_frames)
    for f in range(n_frames):
        s = f * HOP
        seg = mask[s : s + WIN_LENGTH]
        occ[f] = float(np.mean(seg)) if len(seg) else 0.0
    energy = clean_bands.sum(axis=1)
    thr = 0.02 * (float(np.max(energy)) + 1e-12)
    return ((occ > 0.5) & (energy > thr)).astype(np.float32)


def clip_tensors(fb: np.ndarray, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    noisy, clean, mask = make_clip(i)
    nm = stft_mag(noisy)
    cm = stft_mag(clean)
    t = min(len(nm), len(cm))
    nb = to_bands(nm[:t], fb)
    cb = to_bands(cm[:t], fb)
    irm = np.clip(cb / (nb + 1e-8), 0.0, 1.0)
    vad = frame_vad(mask, cb, t)
    logn = np.log(nb + 1e-8)
    return (
        torch.from_numpy(logn.astype(np.float32)),
        torch.from_numpy(irm.astype(np.float32)),
        torch.from_numpy(nb.astype(np.float32)),
        torch.from_numpy(np.stack([vad], axis=-1)),
    )


def export_onnx(seq_model: TinyMaskNet) -> None:
    stream = TinyMaskNet(N_BANDS, HIDDEN)
    stream.load_state_dict(seq_model.state_dict())
    stream.eval()
    bands = torch.zeros(1, 1, N_BANDS)
    h = torch.zeros(1, 1, HIDDEN)
    MODELS.mkdir(parents=True, exist_ok=True)
    names = dict(
        input_names=["bands", "h"],
        output_names=["gain", "vad", "h_out"],
        opset_version=14,
    )
    try:
        torch.onnx.export(stream, (bands, h), str(FP32_ONNX), dynamo=False, **names)
    except TypeError:
        torch.onnx.export(stream, (bands, h), str(FP32_ONNX), **names)


def quantize() -> bool:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(FP32_ONNX), str(INT8_ONNX), weight_type=QuantType.QInt8)
        return True
    except Exception as exc:  # noqa: BLE001
        print("INT8 quantize skipped:", exc)
        return False


def main() -> None:
    fb = filterbank()
    model = TinyMaskNet(N_BANDS, HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    bce = nn.BCELoss()
    print(f"params={model.param_count()} bands={N_BANDS} hidden={HIDDEN} clips={CLIPS}")

    t_data = time.perf_counter()
    data = [clip_tensors(fb, i) for i in range(CLIPS)]
    print(f"data: {CLIPS} clips in {time.perf_counter() - t_data:.1f}s")

    n_val = max(16, CLIPS // 8)
    train, val = data[:-n_val], data[-n_val:]

    def batch_loss(batch):
        tmax = max(x.shape[0] for x, _, _, _ in batch)
        xb = torch.zeros(len(batch), tmax, N_BANDS)
        mb = torch.zeros(len(batch), tmax, N_BANDS)
        nbb = torch.zeros(len(batch), tmax, N_BANDS)
        vb = torch.zeros(len(batch), tmax, 1)
        wb = torch.zeros(len(batch), tmax, 1)
        for bi, (x, m, nbands, v) in enumerate(batch):
            tt = x.shape[0]
            xb[bi, :tt] = x
            mb[bi, :tt] = m
            nbb[bi, :tt] = nbands
            vb[bi, :tt] = v
            wb[bi, :tt] = 1.0
        gain, vad = model.forward_seq(xb)
        # compressed-magnitude loss: perceptually closer than raw mask MSE
        est = (gain * nbb).clamp(min=0.0) ** COMPRESS
        tgt = (mb * nbb).clamp(min=0.0) ** COMPRESS
        mask_loss = (((est - tgt) ** 2) * wb).sum() / wb.sum().clamp(min=1.0) / N_BANDS
        vad_loss = bce((vad * wb).clamp(1e-6, 1 - 1e-6), vb * wb)
        return mask_loss + VAD_WEIGHT * vad_loss, mask_loss, vad_loss

    t0 = time.perf_counter()
    for ep in range(EPOCHS):
        model.train()
        order = np.random.default_rng(ep).permutation(len(train))
        run = run_m = run_v = 0.0
        nb = 0
        for s in range(0, len(train), BATCH):
            batch = [train[int(j)] for j in order[s : s + BATCH]]
            loss, ml, vl = batch_loss(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += float(loss)
            run_m += float(ml)
            run_v += float(vl)
            nb += 1
        sched.step()
        if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                vloss, vm, vv = batch_loss(val)
                gain, vad = model.forward_seq(torch.stack([v[0] for v in val]))
                tgt = torch.stack([v[3] for v in val])
                acc = float((((vad > 0.5).float() == tgt).float()).mean())
            print(
                f"epoch {ep + 1:>2}/{EPOCHS}  train {run / nb:.4f} (mask {run_m / nb:.4f} vad {run_v / nb:.4f})"
                f"  val {float(vloss):.4f}  vad acc {acc:.3f}"
            )
    train_s = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        gain, vad = model.forward_seq(torch.stack([v[0] for v in val]))
        tgt = torch.stack([v[3] for v in val])
        vad_acc = float((((vad > 0.5).float() == tgt).float()).mean())
        speech = tgt > 0.5
        recall = float((vad[speech] > 0.5).float().mean()) if speech.any() else float("nan")
        far = float((vad[~speech] > 0.5).float().mean()) if (~speech).any() else float("nan")

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS)
    export_onnx(model)
    q = quantize()
    size = (INT8_ONNX if q and INT8_ONNX.exists() else FP32_ONNX).stat().st_size
    meta = {
        "params": model.param_count(),
        "n_bands": N_BANDS,
        "hidden": HIDDEN,
        "epochs": EPOCHS,
        "clips": CLIPS,
        "train_seconds": round(train_s, 2),
        "size_kb": round(size / 1024.0, 2),
        "quantized": q,
        "heads": ["gain (32-band mask)", "vad (speech present)"],
        "vad_accuracy": round(vad_acc, 3),
        "vad_recall": round(recall, 3),
        "vad_false_alarm": round(far, 3),
        "train_data": "SIMULATED talkers (varied f0/formants/pauses) + engine/rotor/wind/drone/siren/broadband/impulse; 1-in-6 clips noise-only",
        "augmentation": ["SNR -5..20 dB", "reverb RT60 0.12-0.35 s", "clipping", "level 0.25-1.0"],
        "sample_rate": SAMPLE_RATE,
        "win": WIN_LENGTH,
        "hop": HOP,
        "target_device": "Raspberry Pi 4/5 + ONNX Runtime. Pi 1 cannot do real-time.",
        "pipeline": ["PyTorch", "ONNX", "INT8" if q else "FP32", "Raspberry Pi", "ONNX Runtime"],
        "snr_stoi_pesq": "TARGET 15 dB / 0.85 / 2.5 — not claimed from this tiny net",
    }
    META_JSON.write_text(json.dumps(meta, indent=2))
    print(f"VAD: acc {vad_acc:.3f}  recall {recall:.3f}  false-alarm {far:.3f}")
    print("wrote", FP32_ONNX, INT8_ONNX if q else "(no int8)", META_JSON)


if __name__ == "__main__":
    main()
