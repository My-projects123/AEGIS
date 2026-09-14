"""AEGIS-COM web dashboard API.

Run:
    python serve.py
Then open http://127.0.0.1:8080
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import PROJECT_SUBTITLE, PROJECT_TITLE, PS_ID, SAMPLE_RATE, TARGETS
from backend.mixer import load_wav, mix_files
from backend.live_mix import LiveDisturbance, disturbance_catalog, impulse_catalog
from enhancement.enhancer import tiny_ml_status
from backend.processor import AegisProcessor, logs_to_dicts
from backend.synthesizer import (
    make_demo_scene,
    overlay_impulse,
    synth_engine,
    synth_rotor,
    synth_speech,
    synth_wind,
)
from visualization.plots import plot_spectrograms, plot_timeline, plot_waveforms

FRONTEND = ROOT / "frontend"

app = FastAPI(title="AEGIS-COM", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


def _wav_b64(x: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _png_b64(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _json_safe(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        v = obj.item()
        return v if not isinstance(v, float) or np.isfinite(v) else None
    return str(obj)


def _stages(logs, impulse_t):
    windows = [
        (0.0, 2.0, "Clean speech"),
        (2.0, 5.0, "Engine layer"),
        (5.0, 8.0, "Mixed noise"),
        (8.0, 8.2, "Impulse"),
        (8.2, 10.2, "Recovery"),
        (10.2, 12.5, "After"),
    ]
    out = []
    for a, b, name in windows:
        sel = [fr["mode"] for fr in logs if a <= fr["t"] < b]
        mode = max(set(sel), key=sel.count) if sel else "—"
        out.append({"t0": a, "t1": b, "label": name, "mode": mode})
    return out


def _pack(noisy, enhanced, clean, logs, metrics, sr, impulse_t, kind: str) -> dict:
    log_dicts = logs_to_dicts(logs)
    # keep payload light but dense enough to animate
    step = 2 if len(log_dicts) > 400 else 1
    slim = log_dicts[::step]
    return {
        "kind": kind,
        "sr": int(sr),
        "duration": float(len(noisy) / sr),
        "impulse_time_s": impulse_t,
        "audio": {
            "before": _wav_b64(noisy, sr),
            "after": _wav_b64(enhanced, sr),
        },
        "plots": {
            "waveform": _png_b64(plot_waveforms(noisy, enhanced, sr)),
            "spectrogram": _png_b64(plot_spectrograms(noisy, enhanced, sr)),
            "timeline": _png_b64(plot_timeline(logs, impulse_t)),
        },
        "logs": slim,
        "stages": _stages(log_dicts, impulse_t),
        "metrics": _json_safe(metrics),
        "tiny_ml": tiny_ml_status(),
        "project": {
            "title": PROJECT_TITLE,
            "subtitle": PROJECT_SUBTITLE,
            "ps": PS_ID,
            "targets": TARGETS,
        },
    }


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "poc": "software-in-the-loop",
        "physical_anc": False,
        "title": PROJECT_TITLE,
        "disturbances": disturbance_catalog(),
        "impulses": impulse_catalog(),
        "tiny_ml": tiny_ml_status(),
    }


@app.get("/api/ml")
def api_ml():
    return tiny_ml_status()


@app.post("/api/demo")
def api_demo():
    proc = AegisProcessor(use_ai=True)
    scene = make_demo_scene()
    out = proc.process(scene["noisy"], clean=scene["clean"], impulse_time_s=scene["impulse_time_s"])
    return JSONResponse(
        _pack(
            scene["noisy"],
            out["enhanced"],
            scene["clean"],
            out["logs"],
            out["metrics"],
            scene["sr"],
            scene["impulse_time_s"],
            scene["kind"],
        )
    )


@app.post("/api/process")
async def api_process(
    wav: Optional[UploadFile] = File(default=None),
    noise_kind: str = Form(default="none"),
    snr_db: float = Form(default=6.0),
    add_impulse: str = Form(default="false"),
):
    proc = AegisProcessor(use_ai=True)
    impulse_t = None
    clean = None
    force_impulse = str(add_impulse).lower() in {"1", "true", "on", "yes"}
    if wav is not None and wav.filename:
        raw = await wav.read()
        buf = io.BytesIO(raw)
        noisy, sr = load_wav(buf)
        kind = f"uploaded:{wav.filename}"
    else:
        noisy = synth_speech(5.0)
        clean = noisy.copy()
        sr = SAMPLE_RATE
        kind = "synthetic speech mix"

    dur = len(noisy) / SAMPLE_RATE
    if noise_kind == "engine":
        noisy = mix_files(noisy, synth_engine(dur), snr_db)
    elif noise_kind == "rotor":
        noisy = mix_files(noisy, synth_rotor(dur), snr_db)
    elif noise_kind == "wind":
        noisy = mix_files(noisy, synth_wind(dur), snr_db)
    elif noise_kind == "engine+impulse":
        noisy = mix_files(noisy, synth_engine(dur), snr_db)
        impulse_t = 0.5 * dur
        noisy = overlay_impulse(noisy, at_s=impulse_t)
    if force_impulse:
        impulse_t = 0.5 * len(noisy) / SAMPLE_RATE
        noisy = overlay_impulse(noisy, at_s=impulse_t)

    out = proc.process(noisy, clean=clean, impulse_time_s=impulse_t)
    return JSONResponse(
        _pack(noisy, out["enhanced"], clean, out["logs"], out["metrics"], SAMPLE_RATE, impulse_t, kind)
    )


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """REAL mic + optional SIMULATED battlefield overlay → cleaned speech hops."""
    await ws.accept()
    proc = AegisProcessor(use_ai=True)
    proc.stream_start()
    bed = LiveDisturbance()
    try:
        await ws.send_json({"type": "ready", "sr": SAMPLE_RATE, "hop": 256, **bed.status()})
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if not data:
                text = msg.get("text")
                if not text:
                    continue
                if text == "reset":
                    proc.stream_start()
                    bed.handle_text("reset")
                    await ws.send_json({"type": "ready", "sr": SAMPLE_RATE, "hop": 256, **bed.status()})
                    continue
                await ws.send_json({"type": "overlay", **bed.handle_text(text)})
                continue
            pcm = np.frombuffer(data, dtype=np.float32)
            mixed, ref = bed.mix(pcm)
            hops = proc.stream_samples(mixed, ref=ref)
            if hops:
                await ws.send_json({"type": "hops", "hops": hops})
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            return
