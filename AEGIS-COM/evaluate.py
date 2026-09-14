"""Measure REAL laptop-POC metrics. Writes results/eval.json.

Does not invent STOI/PESQ if libraries are missing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import RESULTS_DIR, SAMPLE_RATE
from backend.metrics import si_snr_db, snr_db, try_pesq, try_stoi
from backend.mixer import save_wav
from backend.processor import AegisProcessor, run_baselines
from backend.synthesizer import make_demo_scene, mix_at_snr, overlay_impulse, synth_engine, synth_speech
from enhancement.dsp_fallback import classical_dsp


def _pack_stoi(d):
    if d.get("available"):
        return {"value": d["value"], "status": "REAL / MEASURED"}
    return {"value": None, "status": "UNAVAILABLE", "reason": d.get("reason", "")}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sr = SAMPLE_RATE
    scene = make_demo_scene(sr)
    save_wav(RESULTS_DIR / "demo_noisy.wav", scene["noisy"], sr)

    proc = AegisProcessor(sr=sr, use_ai=True)
    out = proc.process(scene["noisy"], clean=scene["clean"], impulse_time_s=scene["impulse_time_s"])
    save_wav(RESULTS_DIR / "demo_enhanced.wav", out["enhanced"], sr)
    save_wav(RESULTS_DIR / "demo_clean.wav", scene["clean"], sr)

    baselines = run_baselines(scene["clean"], scene["noisy"], sr)

    # SNR sweep on engine noise (no impulse) — REAL
    speech = synth_speech(4.0, sr, 21)
    engine = synth_engine(4.0, sr, 22)
    sweep = []
    for snr in (-5, 0, 5, 10, 15, 20):
        noisy = mix_at_snr(speech, engine, snr)
        noisy = overlay_impulse(noisy, at_s=2.0, sr=sr, kind="gunshot", seed=40 + snr)
        r = AegisProcessor(sr=sr, use_ai=False).process(noisy, clean=speech[: len(noisy)], impulse_time_s=2.0)
        sweep.append(
            {
                "mix_snr_db": snr,
                "input_snr_db": r["metrics"]["input_snr_db"],
                "output_snr_db": r["metrics"]["output_snr_db"],
                "snr_improvement_db": r["metrics"]["snr_improvement_db"],
                "si_snr_in": r["metrics"]["input_si_snr_db"],
                "si_snr_out": r["metrics"]["output_si_snr_db"],
                "stoi": _pack_stoi(r["metrics"]["stoi"]),
                "pesq": _pack_stoi(r["metrics"]["pesq"]) if r["metrics"]["pesq"].get("available") else {"value": None, "status": "UNAVAILABLE / TARGET 2.5"},
                "impulse_peak_reduction_db": r["metrics"].get("impulse_peak_reduction_db"),
                "detection_latency_ms": r["metrics"].get("impulse_detection_latency_ms"),
                "recovery_time_ms": r["metrics"].get("recovery_time_ms"),
                "rtf": r["metrics"]["rtf"],
                "mode_histogram": r["metrics"]["mode_histogram"],
            }
        )

    # Impulse detector sanity: clean speech should not stay in IMPULSE_PROTECTION
    clean_only = AegisProcessor(sr=sr, use_ai=False).process(speech, clean=speech)
    fp_frames = clean_only["metrics"]["mode_histogram"].get("IMPULSE_PROTECTION", 0)
    total = sum(clean_only["metrics"]["mode_histogram"].values())

    payload = {
        "honesty": {
            "hardware": "NONE — laptop software-in-the-loop",
            "signals": "SYNTHETIC demonstration signals",
            "physical_anc": False,
            "targets_not_claimed": {"snr_db": 15, "stoi": 0.85, "pesq": 2.5},
        },
        "demo_scene": {
            "input_snr_db": out["metrics"]["input_snr_db"],
            "output_snr_db": out["metrics"]["output_snr_db"],
            "snr_improvement_db": out["metrics"]["snr_improvement_db"],
            "input_si_snr_db": out["metrics"]["input_si_snr_db"],
            "output_si_snr_db": out["metrics"]["output_si_snr_db"],
            "stoi_noisy": _pack_stoi(out["metrics"].get("stoi_noisy", {})),
            "stoi_enhanced": _pack_stoi(out["metrics"]["stoi"]),
            "pesq_enhanced": out["metrics"]["pesq"],
            "rtf": out["metrics"]["rtf"],
            "latency_ms_per_s": out["metrics"]["latency_ms_per_s"],
            "impulse_peak_reduction_db": out["metrics"].get("impulse_peak_reduction_db"),
            "impulse_detection_latency_ms": out["metrics"].get("impulse_detection_latency_ms"),
            "recovery_time_ms": out["metrics"].get("recovery_time_ms"),
            "mode_histogram": out["metrics"]["mode_histogram"],
            "switches": out["metrics"]["switches"],
            "ai_status": out["metrics"]["ai_status"],
        },
        "baselines_demo_scene": [
            {
                "system": row["system"],
                "snr_db": row["snr_db"],
                "si_snr_db": row["si_snr_db"],
                "stoi": _pack_stoi(row["stoi"]),
            }
            for row in baselines["rows"]
        ],
        "snr_sweep_engine_plus_impulse": sweep,
        "false_impulse_on_clean_speech": {
            "impulse_protection_frames": fp_frames,
            "total_frames": total,
            "fraction": fp_frames / max(total, 1),
            "note": "REAL count on synthetic clean speech. Not a field false-positive rate.",
        },
    }
    path = RESULTS_DIR / "eval.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload["demo_scene"], indent=2, default=str))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
