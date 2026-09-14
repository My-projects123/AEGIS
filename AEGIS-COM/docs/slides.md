# 6-slide SIH deck — judges, not researchers

Maximum 6 slides including title. Generate the file with `python generate_ppt.py` → `docs/AEGIS-COM_SIH26052.pptx`.

Visual system: navy #0B1C2C, gold #C5A46E, cyan #3EC6C9, white text. Diagrams over paragraphs.

---

## SLIDE 1 — Problem and proposed solution
**Title:** AEGIS-COM | SIH26052 DRDO

**Exact text:**
- Defence radio speech dies in mixed noise: engines, rotors, and impulses together.
- Classical LMS / Wiener / spectral subtraction: slow, linear, unstable on bangs.
- **We do not ship a fake headset.** We ship an event-aware **comms-path** controller.
- AEGIS-COM: sense the acoustic event → switch enhancement policy → recover.

**Diagram:**  
`Boom mic audio → Acoustic intelligence → FSM controller → {preserve | stationary | dynamic | impulse | recover} → radio TX`  
Side annotation: Earcup ANR = FUTURE hardware companion.

**Visual:** One horizontal pipeline, five mode chips.

**Speaker notes:** Open with the radio-command failure, not “we built AI.” Name DRDO PS. Say laptop POC in sentence two.

**Judge takeaway:** They understood comms intelligibility vs hearing protection.

---

## SLIDE 2 — Existing systems and the real gap
**Title:** What already exists — and what still fails

**Exact text:**
- 3M ComTac VI: certified impulse hearing protection + **manual MAP** listening profiles.
- INVISIO RA108/RA5100: PNR + ANR for continuous vehicle noise; user-toggled hear-through.
- Generic AI denoisers: one policy; smear after transients.
- **Gap:** no automatic, confidence-gated **speech-enhancement policy switch** with **mandatory recovery** on the radio path.

**Diagram:** Two columns: “Ear protection (mature)” vs “Comms policy (missing).”

**Visual:** Honest table, no dunking on 3M.

**Speaker notes:** “If we say they cannot handle gunshots, we lose the room.”

**Judge takeaway:** Novelty is not ANC itself.

---

## SLIDE 3 — Novel architecture
**Title:** Event-aware adaptive speech protection

**Exact text:**
- Features every 16 ms: class, severity, speech P, impulse P, confidence.
- FSM: SPEECH PRESERVATION · STATIONARY · DYNAMIC · IMPULSE PROTECTION · RECOVERY.
- Impulse freezes the noise model. Recovery unfreezes it. Confidence stops chatter.
- AI enhancer is a *module inside modes*, not the invention.

**Diagram:** The stack from `docs/architecture.md`.

**Visual:** State bubbles with arrows (impulse → recover → scene).

**Speaker notes:** Point at RECOVERY. That word is the novelty test.

**Judge takeaway:** Specific mechanism, not a buzzword stack.

---

## SLIDE 4 — Working POC
**Title:** Laptop software-in-the-loop — real audio, real controller

**Exact text:**
- Python, 16 kHz, 32 ms frames. No Jetson on this table.
- REAL: detector, FSM, Wiener enhancement, before/after audio, spectrogram, SNR.
- Demo: clean → engine → rotor mix → synthetic impulse → recovery.
- NOT claimed: physical ANC, field datasets, military certification.

**Diagram:** Screenshot placeholder of the dashboard timeline.

**Visual:** Label strip REAL / SIMULATED / TARGET / FUTURE.

**Speaker notes:** Click START DEMO after this slide if time; else this slide is the evidence.

**Judge takeaway:** Students built the thing they drew.

---

## SLIDE 5 — Results and validation
**Title:** What we measured on the laptop (synthetic scene)

**Exact text:**  
Fill from `results/eval.json` after `python evaluate.py`. Until then print:

- SNR in / out / Δ — REAL on synthetic pairs
- SI-SNR — REAL
- STOI / PESQ — MEASURED if library present, else **TARGET only**
- Impulse detect latency, recovery time, peak reduction — REAL on this scene
- SIH goals SNR>15, STOI>0.85, PESQ>2.5 = **TARGETS**

**Diagram:** 3-row baseline table: Raw | Fixed DSP | AEGIS-COM.

**Visual:** No fake 99% gauges.

**Speaker notes:** Read the JSON numbers. If STOI is n/a, say it.

**Judge takeaway:** Honesty = credibility.

---

## SLIDE 6 — Deployment, impact, recap
**Title:** From this POC to an Indian comms stack

**Exact text:**
- Path: laptop SIL → USB I/O → Raspberry Pi / Jetson → headset ANR + boom mic.
- Latency budget (ESTIMATE): 16 ms hop + 16–32 ms model + I/O < 60 ms conversational.
- Impact: fewer lost commands when the scene jumps; software-updatable; does not require throwing away ComTac-class cups.
- Novelty recap: **automatic event-aware policy + recovery**, not “AI headphones.”

**Diagram:** Mic → codec → detector → FSM → AI/DSP → radio / ear.

**Visual:** Roadmap bar: POC → edge → field trial.

**Speaker notes:** Close on DRDO benefit: policy software vs 10-year headset refresh.

**Judge takeaway:** Feasible next step, frozen story.
