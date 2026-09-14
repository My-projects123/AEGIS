# Audio samples

This folder is filled by:

```bash
python generate_samples.py
```

Every generated WAV is a **SIMULATED** laboratory signal (formant speech, harmonic engine, modulated rotor, synthetic impulse). They are **not** DRDO field recordings, gunshot captures, or helicopter cockpit audio.

You may drop your own files here:

- any clean speech WAV
- any noise WAV (engine, crowd, fans)
- the mixer will resample to 16 kHz mono

Public datasets for later training (not required tomorrow):

| Role | Source | URL |
|---|---|---|
| Clean speech | LibriSpeech | http://www.openslr.org/12 |
| Clean speech | VoiceBank | https://datashare.ed.ac.uk/handle/10283/2791 |
| General noise | DEMAND | https://zenodo.org/records/1227121 |
| Scalable noisy speech | MS-SNSD | https://github.com/microsoft/MS-SNSD |
| DNS challenge | Microsoft DNS | https://github.com/microsoft/DNS-Challenge |
| SIH-oriented pack (community) | Hugging Face `Panav-Payappagoudar/sih-26-processed-audio` | verify license before use |

Do not claim those datasets were used in tomorrow's POC unless you actually downloaded them.
