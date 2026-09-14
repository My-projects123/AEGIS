"""AEGIS-COM Streamlit dashboard — judged laptop POC.

Run from this folder:
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import streamlit as st

from backend.config import PROJECT_SUBTITLE, PROJECT_TITLE, PS_ID, SAMPLE_RATE, TARGETS
from backend.mixer import load_wav, mix_files
from backend.processor import AegisProcessor
from backend.synthesizer import (
    make_demo_scene,
    overlay_impulse,
    synth_engine,
    synth_impulse,
    synth_rotor,
    synth_speech,
    synth_wind,
)
from frontend.theme import GOLD, css
from visualization.plots import plot_spectrograms, plot_timeline, plot_waveforms

st.set_page_config(page_title="AEGIS-COM", layout="wide", page_icon="◆")
st.markdown(css(), unsafe_allow_html=True)


def _wav_bytes(x: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@st.cache_resource
def get_processor() -> AegisProcessor:
    return AegisProcessor(use_ai=True)


def _mode_style(mode: str) -> str:
    colors = {
        "SPEECH_PRESERVATION": "#3EC6C9",
        "STATIONARY": "#7CB87C",
        "DYNAMIC": "#E0B44A",
        "IMPULSE_PROTECTION": "#D45D5D",
        "RECOVERY": "#C5A46E",
    }
    c = colors.get(mode, GOLD)
    return f'<span class="mode-pill" style="background:{c};color:#0B1C2C">{mode.replace("_", " ")}</span>'


def _dominant_mode(logs, t0, t1):
    sel = [fr.mode for fr in logs if t0 <= fr.t < t1]
    if not sel:
        return "—"
    return max(set(sel), key=sel.count)


def _snapshot(logs, t):
    if not logs:
        return None
    return min(logs, key=lambda fr: abs(fr.t - t))


def render_metrics(m: dict, logs):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    snap = logs[len(logs) // 2] if logs else None
    last_imp = next((fr for fr in reversed(logs) if fr.mode == "IMPULSE_PROTECTION"), snap)
    c1.markdown(f'<div class="card"><div class="metric-label">Input SNR</div><div class="metric-value">{_fmt(m.get("input_snr_db"), "dB")}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div class="metric-label">Output SNR</div><div class="metric-value">{_fmt(m.get("output_snr_db"), "dB")}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><div class="metric-label">SNR improvement</div><div class="metric-value">{_fmt(m.get("snr_improvement_db"), "dB")}</div></div>', unsafe_allow_html=True)
    stoi = m.get("stoi") or {}
    pesq = m.get("pesq") or {}
    c4.markdown(f'<div class="card"><div class="metric-label">STOI</div><div class="metric-value">{_opt(stoi)}</div></div>', unsafe_allow_html=True)
    c5.markdown(f'<div class="card"><div class="metric-label">PESQ</div><div class="metric-value">{_opt(pesq)}</div></div>', unsafe_allow_html=True)
    c6.markdown(f'<div class="card"><div class="metric-label">Laptop RTF</div><div class="metric-value">{m.get("rtf", 0):.2f}×</div></div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Processing latency", f"{m.get('latency_ms_per_s', 0):.0f} ms / s audio", help="REAL laptop wall-clock. Not embedded latency.")
    d2.metric("Impulse detect latency", _fmt(m.get("impulse_detection_latency_ms"), "ms"))
    d3.metric("Recovery time", _fmt(m.get("recovery_time_ms"), "ms"))
    d4.metric("Impulse peak reduction", _fmt(m.get("impulse_peak_reduction_db"), "dB"))


def _fmt(v, unit=""):
    if v is None:
        return "n/a"
    return f"{v:.2f} {unit}".strip()


def _opt(d: dict) -> str:
    if d.get("available") and d.get("value") is not None:
        return f"{d['value']:.3f}"
    return "n/a"


def show_result(noisy, enhanced, logs, metrics, sr, impulse_t, clean=None):
    st.markdown("### System status")
    # pick a representative frame around impulse if present else mid
    t_focus = impulse_t if impulse_t is not None else (logs[len(logs)//2].t if logs else 0)
    fr = _snapshot(logs, t_focus)
    cols = st.columns([1.4, 1, 1, 1, 1, 1])
    if fr:
        cols[0].markdown("**ACTIVE PROCESSING MODE**", unsafe_allow_html=True)
        cols[0].markdown(_mode_style(fr.mode), unsafe_allow_html=True)
        cols[1].metric("Noise class", fr.noise_class)
        cols[2].metric("Severity", f"{fr.severity:.2f}")
        cols[3].metric("Speech P", f"{fr.speech_prob:.2f}")
        cols[4].metric("Impulse P", f"{fr.impulse_prob:.2f}")
        cols[5].metric("Est. SNR", f"{fr.snr_est_db:.1f} dB")

    st.caption(f"Snapshot at t = {t_focus:.2f}s  ·  controller reason: {fr.reason if fr else '—'}")
    render_metrics(metrics, logs)

    st.markdown("### Event timeline (REAL controller states)")
    if impulse_t is not None:
        stages = [
            (0.0, 2.0, "1 Clean speech"),
            (2.0, 5.0, "2 Engine layer"),
            (5.0, 8.0, "3 Mixed noise"),
            (8.0, 8.2, "4 Impulse"),
            (8.2, 10.2, "5 Recovery"),
            (10.2, 12.5, "6 After"),
        ]
        sc = st.columns(6)
        for col, (a, b, label) in zip(sc, stages):
            mode = _dominant_mode(logs, a, b)
            col.markdown(f"**{label}**")
            col.markdown(_mode_style(mode), unsafe_allow_html=True)

    st.image(plot_timeline(logs, impulse_t), use_container_width=True)

    a, b = st.columns(2)
    with a:
        st.markdown("**BEFORE — input**")
        st.audio(_wav_bytes(noisy, sr), format="audio/wav")
    with b:
        st.markdown("**AFTER — AEGIS-COM**")
        st.audio(_wav_bytes(enhanced, sr), format="audio/wav")

    st.markdown("### Waveform")
    st.image(plot_waveforms(noisy, enhanced, sr), use_container_width=True)
    st.markdown("### Spectrogram")
    st.image(plot_spectrograms(noisy, enhanced, sr), use_container_width=True)

    ai = metrics.get("ai_status") or {}
    st.markdown(
        f'<div class="honesty"><b>REAL vs TARGET vs FUTURE</b><br>'
        f"REAL: waveform, spectrogram, impulse detector, FSM mode switching, "
        f"measured SNR / SI-SNR, laptop RTF, peak reduction.<br>"
        f"AI enhancer: {ai.get('name','DSP')} — {ai.get('note','')}<br>"
        f"STOI {'MEASURED' if (metrics.get('stoi') or {}).get('available') else 'UNAVAILABLE (install pystoi)'} · "
        f"PESQ {'MEASURED' if (metrics.get('pesq') or {}).get('available') else 'UNAVAILABLE — TARGET is '+str(TARGETS['pesq'])}.<br>"
        f"SIMULATED: synthetic engine / rotor / impulse if you used START DEMO.<br>"
        f"NOT claimed: physical ANC, Jetson numbers, field STOI/PESQ, military certification.</div>",
        unsafe_allow_html=True,
    )


def main():
    st.markdown(f'<div class="hero-title">{PROJECT_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">{PROJECT_SUBTITLE}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ps">{PS_ID} · DRDO · AEGIS-COM laptop POC · software-in-the-loop · not physical ANC</div>',
        unsafe_allow_html=True,
    )

    proc = get_processor()

    with st.sidebar:
        st.header("AEGIS-COM")
        st.caption("Event-aware adaptive speech protection")
        st.markdown(
            """
**Modes**
- SPEECH PRESERVATION
- STATIONARY
- DYNAMIC
- IMPULSE PROTECTION
- RECOVERY

**Honesty**
- Laptop POC = digital enhancement
- Final system = comms path + headset ANC
"""
        )
        st.divider()
        uploaded = st.file_uploader("Upload WAV (optional)", type=["wav"])
        noise_kind = st.selectbox(
            "Add synthetic noise",
            ["none", "engine", "rotor", "wind", "engine+impulse"],
        )
        snr = st.slider("Mix SNR (dB)", -5, 20, 6)
        add_impulse = st.checkbox("Force impulse at 50% duration", value=False)

    col_l, col_r = st.columns([1, 2])
    with col_l:
        start = st.button("START DEMO", type="primary", use_container_width=True)
        process_upload = st.button("PROCESS UPLOAD / MIX", use_container_width=True)

    if start:
        with st.spinner("Generating simulated scene and running the real pipeline…"):
            scene = make_demo_scene()
            out = proc.process(scene["noisy"], clean=scene["clean"], impulse_time_s=scene["impulse_time_s"])
            st.session_state["result"] = {
                "noisy": scene["noisy"],
                "enhanced": out["enhanced"],
                "clean": scene["clean"],
                "logs": out["logs"],
                "metrics": out["metrics"],
                "sr": scene["sr"],
                "impulse_t": scene["impulse_time_s"],
                "kind": scene["kind"],
            }
        st.success("Demo processed with the real detector + controller + enhancer.")

    if process_upload:
        if uploaded is None:
            speech = synth_speech(5.0)
            noisy = speech.copy()
            clean = speech
            impulse_t = None
            sr = SAMPLE_RATE
        else:
            data, sr = load_wav(uploaded)
            # load_wav already resamples
            noisy = data
            clean = data.copy()
            impulse_t = None
        if noise_kind == "engine":
            noisy = mix_files(noisy, synth_engine(len(noisy) / SAMPLE_RATE), snr)
        elif noise_kind == "rotor":
            noisy = mix_files(noisy, synth_rotor(len(noisy) / SAMPLE_RATE), snr)
        elif noise_kind == "wind":
            noisy = mix_files(noisy, synth_wind(len(noisy) / SAMPLE_RATE), snr)
        elif noise_kind == "engine+impulse":
            noisy = mix_files(noisy, synth_engine(len(noisy) / SAMPLE_RATE), snr)
            noisy = overlay_impulse(noisy, at_s=0.5 * len(noisy) / SAMPLE_RATE)
            impulse_t = 0.5 * len(noisy) / SAMPLE_RATE
        if add_impulse:
            impulse_t = 0.5 * len(noisy) / SAMPLE_RATE
            noisy = overlay_impulse(noisy, at_s=impulse_t)
        with st.spinner("Processing…"):
            out = proc.process(noisy, clean=clean if uploaded is None else None, impulse_time_s=impulse_t)
        st.session_state["result"] = {
            "noisy": noisy,
            "enhanced": out["enhanced"],
            "clean": clean if uploaded is None else None,
            "logs": out["logs"],
            "metrics": out["metrics"],
            "sr": SAMPLE_RATE,
            "impulse_t": impulse_t,
            "kind": "user mix",
        }

    if "result" in st.session_state:
        r = st.session_state["result"]
        st.caption(r.get("kind", ""))
        show_result(r["noisy"], r["enhanced"], r["logs"], r["metrics"], r["sr"], r["impulse_t"], r.get("clean"))
    else:
        st.info("Press **START DEMO** to run the 12.5 s scripted scene: clean → engine → dynamic → impulse → recovery.")


if __name__ == "__main__":
    main()
