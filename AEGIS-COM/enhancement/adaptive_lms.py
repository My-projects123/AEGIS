"""NLMS adaptive noise canceller for the primary + reference microphone pair.

REAL algorithm. This is the classical two-mic ANC front end the problem
statement asks for ("a lightweight adaptive filter (e.g. LMS) for residual
noise suppression"), placed *before* the spectral/AI stage:

    primary  = voice + h * disturbance      (boom mic)
    reference =        disturbance          (shell / environment mic)

NLMS estimates h and subtracts, so the correlated bulk of the disturbance is
removed without touching the voice — a spectral mask cannot do that, because
it can only scale magnitudes it cannot separate.

Honesty:
  NLMS                REAL, running on every hop
  reference channel   SIMULATED in this POC (see backend/live_mix.RefPath)
  adaptation freeze   driven by the EAASP impulse state, so a gunshot cannot
                      blow up the filter coefficients
"""

from __future__ import annotations

import numpy as np

NLMS_TAPS = 64
NLMS_MU = 0.4
NLMS_LEAK = 1e-6


class NLMSCanceller:
    def __init__(self, n_taps: int = NLMS_TAPS, mu: float = NLMS_MU, leak: float = NLMS_LEAK) -> None:
        self.n = int(n_taps)
        self.mu = float(mu)
        self.leak = float(leak)
        self.w = np.zeros(self.n)
        self.tail = np.zeros(self.n - 1)  # reference history across hops
        self.err_rms = 1e-3
        self.cancel_db = 0.0

    def reset(self) -> None:
        self.w = np.zeros(self.n)
        self.tail = np.zeros(self.n - 1)
        self.err_rms = 1e-3
        self.cancel_db = 0.0

    def process(
        self,
        primary: np.ndarray,
        ref: np.ndarray,
        adapt: bool = True,
        mu_scale: float = 1.0,
    ) -> np.ndarray:
        """Return the residual e = primary - w·ref, adapting w sample by sample.

        `mu_scale` slows adaptation while the operator is talking. The voice is
        uncorrelated with the reference, so it acts as gradient noise and drives
        the coefficients off — the two-mic equivalent of double-talk in an echo
        canceller. Adapting mainly in speech pauses is what makes the residual
        small instead of merely smaller.
        """
        d = np.asarray(primary, dtype=np.float64).reshape(-1)
        x = np.asarray(ref, dtype=np.float64).reshape(-1)
        if len(x) != len(d):
            x = np.pad(x, (0, max(0, len(d) - len(x))))[: len(d)]
        xp = np.concatenate([self.tail, x])
        e = np.empty(len(d))
        w = self.w
        n = self.n
        mu = self.mu * float(np.clip(mu_scale, 0.0, 1.0))
        adapt = adapt and mu > 1e-3
        decay = 1.0 - self.leak
        clip_k = 3.0
        for i in range(len(d)):
            seg = xp[i : i + n][::-1]  # newest sample first
            err = d[i] - float(w @ seg)
            e[i] = err
            self.err_rms = 0.999 * self.err_rms + 0.001 * abs(err)
            if adapt:
                # Robust NLMS: clamp the error driving the update. A gunshot in
                # the reference would otherwise yank the coefficients in one
                # sample and ring for a whole hop afterwards.
                lim = clip_k * self.err_rms + 1e-6
                err_u = float(np.clip(err, -lim, lim))
                # normalised step: stable across noise levels without retuning
                p = float(seg @ seg) + 1e-6
                w = decay * w + (mu * err_u / p) * seg
        self.w = w
        self.tail = xp[-(n - 1) :]
        d_pow = float(np.mean(d ** 2) + 1e-12)
        e_pow = float(np.mean(e ** 2) + 1e-12)
        self.cancel_db = float(np.clip(10.0 * np.log10(d_pow / e_pow), -6.0, 40.0))
        return e
