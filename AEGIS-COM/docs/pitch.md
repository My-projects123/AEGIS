# Pitches

## 10 seconds
Defence radios fail when engine noise, changing noise, and bangs hit the same sentence. AEGIS-COM automatically switches how it cleans speech — including a recovery step after an impulse — so the next command is still intelligible.

## 30 seconds
**Problem:** Soldiers lose radio words in mixed defence noise.  
**Gap:** Headsets already protect hearing and let users pick listening profiles. They do not automatically change *comms-path* enhancement after an impulse.  
**Innovation:** AEGIS-COM estimates the event every 16 ms and routes audio through speech-preservation, stationary, dynamic, impulse-protection, or recovery.  
**POC:** Laptop software-in-the-loop — real detector, real controller, real audio, measured SNR. Not a physical ANC headset.  
**Impact:** Software that can sit on Indian radios without claiming we reinvented ComTac.

## 60 seconds
SIH 26052 asks for AI/ML adaptive noise control that keeps speech intelligible in stationary, non-stationary, and impulsive defence noise, in real time on embedded hardware. Traditional LMS and Wiener filters lag or blow up when the scene jumps. Existing tactical headsets — 3M ComTac VI, INVISIO RA108/RA5100 — already do hearing protection, ANR, and even impulse limiting, but their listening profiles are operator-selected and their ANR is built for continuous rumble. Generic AI denoisers use one policy and smear speech after a bang. Our contribution is event-aware adaptive speech protection: a confidence-gated controller that changes processing strategy and then recovers. Tomorrow you will hear a scripted scene — clean speech, engine, rotor mix, synthetic impulse — and see the mode timeline move because the code moved, not because the UI lied. Physical ANC and Jetson deployment are the next hardware steps, not today’s claim.
