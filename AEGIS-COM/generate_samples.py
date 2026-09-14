"""Write synthetic demonstration WAV files. SIMULATED, not field recordings."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import AUDIO_DIR, SAMPLE_RATE
from backend.mixer import mix_files, save_wav
from backend.synthesizer import (
    make_demo_scene,
    overlay_impulse,
    synth_drone,
    synth_engine,
    synth_impulse,
    synth_rotor,
    synth_siren,
    synth_speech,
    synth_wind,
)


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    sr = SAMPLE_RATE
    speech = synth_speech(6.0, sr, seed=1)
    save_wav(AUDIO_DIR / "clean_speech_synthetic.wav", speech, sr)
    for name, fn in [
        ("noise_engine_synthetic.wav", lambda: synth_engine(6.0, sr, 2)),
        ("noise_rotor_synthetic.wav", lambda: synth_rotor(6.0, sr, 3)),
        ("noise_wind_synthetic.wav", lambda: synth_wind(6.0, sr, 4)),
        ("noise_siren_synthetic.wav", lambda: synth_siren(6.0, sr, 5)),
        ("noise_drone_synthetic.wav", lambda: synth_drone(6.0, sr, 6)),
        ("impulse_gunshot_synthetic.wav", lambda: synth_impulse(0.4, sr, 7, "gunshot")),
        ("impulse_artillery_synthetic.wav", lambda: synth_impulse(0.5, sr, 8, "artillery")),
        ("impulse_explosion_synthetic.wav", lambda: synth_impulse(0.6, sr, 9, "explosion")),
    ]:
        save_wav(AUDIO_DIR / name, fn(), sr)

    engine = synth_engine(6.0, sr, 2)
    for snr in (-5, 0, 5, 10, 15, 20):
        mixed = mix_files(speech, engine, snr)
        save_wav(AUDIO_DIR / f"pair_speech_engine_{snr}dB.wav", mixed, sr)
    mixed_imp = overlay_impulse(mix_files(speech, engine, 5.0), at_s=3.0, sr=sr)
    save_wav(AUDIO_DIR / "pair_speech_engine_impulse_5dB.wav", mixed_imp, sr)

    scene = make_demo_scene(sr)
    save_wav(AUDIO_DIR / "demo_clean.wav", scene["clean"], sr)
    save_wav(AUDIO_DIR / "demo_noisy.wav", scene["noisy"], sr)
    print(f"Wrote synthetic WAVs to {AUDIO_DIR}")
    print("All files are SIMULATED demonstration signals, not defence field recordings.")


if __name__ == "__main__":
    main()
