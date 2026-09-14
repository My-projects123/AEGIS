"""CLI entry: generate demo scene, process, print measured metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import RESULTS_DIR, SAMPLE_RATE
from backend.mixer import load_wav, save_wav
from backend.processor import AegisProcessor
from backend.synthesizer import make_demo_scene


def main() -> None:
    p = argparse.ArgumentParser(description="AEGIS-COM laptop POC")
    p.add_argument("--wav", type=str, default="", help="Optional noisy WAV")
    p.add_argument("--clean", type=str, default="", help="Optional clean reference WAV")
    p.add_argument("--no-ai", action="store_true")
    args = p.parse_args()

    proc = AegisProcessor(sr=SAMPLE_RATE, use_ai=not args.no_ai)
    impulse_t = None
    if args.wav:
        noisy, sr = load_wav(args.wav)
        clean = load_wav(args.clean)[0] if args.clean else None
    else:
        scene = make_demo_scene()
        noisy, clean, sr = scene["noisy"], scene["clean"], scene["sr"]
        impulse_t = scene["impulse_time_s"]
        print("Using SIMULATED demo scene (not field recordings).")

    out = proc.process(noisy, clean=clean, impulse_time_s=impulse_t)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_wav(RESULTS_DIR / "cli_enhanced.wav", out["enhanced"], sr)
    metrics = {k: v for k, v in out["metrics"].items()}
    print(json.dumps({k: metrics[k] for k in metrics if k not in ("")}, indent=2, default=str))
    print("\nModes over time (every ~0.16 s):")
    for fr in out["logs"][::10]:
        print(f"  t={fr.t:6.2f}s  {fr.mode:22s}  class={fr.noise_class:10s}  imp={fr.impulse_prob:.2f}")
    print(f"\nEnhanced WAV → {RESULTS_DIR / 'cli_enhanced.wav'}")
    print("This POC is software-in-the-loop speech enhancement, not physical ANC.")


if __name__ == "__main__":
    main()
