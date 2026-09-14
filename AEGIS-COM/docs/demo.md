# Live demo — 2 minutes

Hardware: laptop, headphones, Streamlit. No Jetson. Do not pretend otherwise.

## Before you click

1. `cd sih26052_poc`
2. `python3 -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python generate_samples.py`
5. `streamlit run app.py`
6. Plug in headphones. Volume moderate — the synthetic impulse is loud by design.

## Exact words (memorize)

“This is AEGIS-COM, a software-in-the-loop proof of concept for SIH 26052. We are not claiming a physical ANC headset today. We built the communication-path controller that existing tactical headsets do not automate.”

Click **START DEMO**. Wait for processing. Play BEFORE, then AFTER.

“First you hear simulated radio speech. The controller is in speech-preservation — light processing, keep consonants.”

Point at SPEECH PRESERVATION (first ~3–4 seconds).

“Engine-like rumble comes in. Once that low-frequency noise is established, the controller switches to stationary enhancement — stronger suppression, slow noise estimate.”

Point at STATIONARY.

“We then keep a mixed rumble under the speech. The important change is not a cartoon ‘AI ON’ lamp. It is that the policy is now a noise-suppression policy, not a speech-preservation policy.”

“Now a synthetic impulsive event. This is not a recorded gunshot; it is a laboratory transient with the same problem: a few milliseconds, broadband, huge peak.”

Point at the white dashed line.

“The detector fires in the same 16 ms analysis hop. Impulse protection freezes the noise estimate and gates the transient.”

Point at IMPULSE PROTECTION.

“When the event ends we do not jump back to a random filter. We enter recovery for about 0.4 s, then return to the scene mode. That recovery state is what generic denoisers skip, and it is why the next command is not smeared.”

Play AFTER if you have not already.

“The point: we are not applying one fixed filter, and we are not claiming we invented hearing protection. We change the processing strategy when the acoustic event changes. This timeline is the real controller, not an animation.”

If SNR numbers are on screen: read **only** the numbers on the dashboard. If STOI/PESQ say n/a, say “not computed in this build; SIH figures are targets.”

## If something fails

- No START DEMO output: run `python run_demo.py` and play `results/cli_enhanced.wav`.
- Classifier slow first time: wait; it trains once on synthetic features.
- Browser autoplay blocked: press play manually.

## What not to say

- “This is ANC.”
- “We tested on Jetson.”
- “We achieved STOI 0.85.”
- “Existing headsets cannot handle gunshots.”
