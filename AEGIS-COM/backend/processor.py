"""End-to-end software-in-the-loop processor.

This is digital speech enhancement + event-aware control.
It is NOT physical analog ANC. Labelled honestly in every result dict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from backend.config import HOP, IMPULSE_CEILING, SAMPLE_RATE, WIN_LENGTH
from backend.metrics import Timer, peak_reduction_db, summarize_metrics
from controller.adaptive_controller import AdaptiveController
from detection.acoustic_analyzer import frame_features
from detection.classifier import AcousticClassifier
from detection.impulse_detector import ImpulseDetector
from enhancement.adaptive_lms import NLMSCanceller
from enhancement.dsp_fallback import OutputAGC, classical_dsp
from enhancement.speech_gate import SpeechGate
from enhancement.enhancer import EventAwareEnhancer, rnnoise_status, tiny_ml_status


@dataclass
class FrameLog:
    t: float
    mode: str
    noise_class: str
    confidence: float
    severity: float
    speech_prob: float
    impulse_prob: float
    snr_est_db: float
    reason: str
    energy_ratio: float
    kurtosis: float
    flux: float


class AegisProcessor:
    def __init__(self, sr: int = SAMPLE_RATE, use_ai: bool = True) -> None:
        self.sr = sr
        self.use_ai = use_ai
        self.classifier = AcousticClassifier()
        self.reset()

    def reset(self) -> None:
        self.detector = ImpulseDetector()
        self.controller = AdaptiveController()
        self.enhancer = EventAwareEnhancer(self.sr, use_ai=self.use_ai)
        self.prev_mag = None

    def analyze(self, x: np.ndarray) -> list[FrameLog]:
        x = np.asarray(x, dtype=np.float64)
        pad = np.pad(x, (WIN_LENGTH, WIN_LENGTH), mode="reflect")
        logs: list[FrameLog] = []
        self.detector.reset()
        self.controller.reset()
        self.prev_mag = None
        for i, start in enumerate(range(0, len(pad) - WIN_LENGTH + 1, HOP)):
            frame = pad[start : start + WIN_LENGTH]
            feat = frame_features(frame, self.prev_mag, self.detector.noise_rms, self.sr)
            n_fft = int(2 ** np.ceil(np.log2(len(frame))))
            self.prev_mag = np.abs(np.fft.rfft(frame * np.hanning(len(frame)), n=n_fft))
            impulse = self.detector.update(feat)
            cls = self.classifier.predict_features(feat, impulse.probability)
            st = self.controller.step(feat, impulse, cls)
            t = (start - WIN_LENGTH) / self.sr
            logs.append(
                FrameLog(
                    t=float(t),
                    mode=st.mode,
                    noise_class=st.noise_class,
                    confidence=st.confidence,
                    severity=st.severity,
                    speech_prob=st.speech_prob,
                    impulse_prob=st.impulse_prob,
                    snr_est_db=st.snr_est_db,
                    reason=st.reason,
                    energy_ratio=impulse.energy_ratio,
                    kurtosis=feat.kurtosis,
                    flux=feat.spectral_flux,
                )
            )
        return logs

    def process(
        self,
        noisy: np.ndarray,
        clean: np.ndarray | None = None,
        impulse_time_s: float | None = None,
    ) -> dict:
        self.reset()
        noisy = np.asarray(noisy, dtype=np.float64)
        with Timer() as analysis_timer:
            logs = self.analyze(noisy)
        modes = [fr.mode for fr in logs]
        self.enhancer.prepare_ai(noisy)
        with Timer() as enh_timer:
            enhanced = self.enhancer.process_file(noisy, modes)
        process_s = analysis_timer.elapsed + enh_timer.elapsed

        extra = {
            "impulse_events": int(sum(1 for i, fr in enumerate(logs) if fr.mode == "IMPULSE_PROTECTION" and (i == 0 or logs[i - 1].mode != "IMPULSE_PROTECTION"))),
            "mode_histogram": _hist([fr.mode for fr in logs]),
            "class_histogram": _hist([fr.noise_class for fr in logs]),
            "switches": int(self.controller.switches),
            "ai_status": rnnoise_status(),
            "ai_used": bool(self.enhancer.ai_ok),
            "tiny_ml": tiny_ml_status(),
            "tiny_used": bool(self.enhancer.tiny),
            "poc_type": "SOFTWARE-IN-THE-LOOP digital speech enhancement (NOT physical ANC)",
            "analysis_seconds": analysis_timer.elapsed,
            "enhance_seconds": enh_timer.elapsed,
        }
        if impulse_time_s is not None:
            extra["impulse_peak_reduction_db"] = peak_reduction_db(noisy, enhanced, impulse_time_s, self.sr)
            extra["impulse_detection_latency_ms"] = _detection_latency_ms(logs, impulse_time_s)
            extra["recovery_time_ms"] = _recovery_time_ms(logs)
        metrics = summarize_metrics(clean, noisy, enhanced, self.sr, process_s, extra)
        return {
            "enhanced": enhanced,
            "logs": logs,
            "metrics": metrics,
            "modes": modes,
        }

    def stream_start(self) -> None:
        self.reset()
        self._pending = np.zeros(0)
        self._ref_pending = np.zeros(0)
        self._winbuf = np.zeros(WIN_LENGTH)
        self._errbuf = np.zeros(WIN_LENGTH)
        self._refbuf = np.zeros(WIN_LENGTH)
        self._acc = np.zeros(WIN_LENGTH)
        self._w2 = np.hanning(WIN_LENGTH) ** 2
        self._t = 0.0
        self._has_ref = False
        self._rho = 0.25  # residual-to-reference power after NLMS (measured below)
        self._rho_hist: list[float] = []
        self._prev_speech_prob = 0.0
        self._prev_mag_enh = None
        self._outwin = np.zeros(WIN_LENGTH)
        self.lms = NLMSCanceller()
        self.agc = OutputAGC()
        self.gate = SpeechGate()
        self.live_logs = []
        self.prev_mag = None

    def stream_samples(self, samples: np.ndarray, ref: np.ndarray | None = None) -> list[dict]:
        """`ref` is the reference-mic channel (disturbance only), if available."""
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        if not len(x):
            return []
        self._has_ref = ref is not None
        r = np.asarray(ref, dtype=np.float64).reshape(-1) if self._has_ref else np.zeros(len(x))
        if len(r) != len(x):
            r = np.pad(r, (0, max(0, len(x) - len(r))))[: len(x)]
        self._pending = np.concatenate([self._pending, x])
        self._ref_pending = np.concatenate([self._ref_pending, r])
        hops = []
        while len(self._pending) >= HOP:
            hop = self._pending[:HOP]
            ref_hop = self._ref_pending[:HOP]
            self._pending = self._pending[HOP:]
            self._ref_pending = self._ref_pending[HOP:]
            if self._has_ref:
                # Freeze adaptation while an impulse is active (previous hop's
                # decision — this runs ahead of the detector): a gunshot in the
                # reference would otherwise throw the coefficients off.
                adapt = self.controller.mode != "IMPULSE_PROTECTION"
                err_hop = self.lms.process(
                    hop,
                    ref_hop,
                    adapt=adapt,
                    mu_scale=1.0 - 0.95 * self._prev_speech_prob,
                )
            else:
                err_hop = hop
            self._winbuf = np.concatenate([self._winbuf[HOP:], hop])
            self._errbuf = np.concatenate([self._errbuf[HOP:], err_hop])
            self._refbuf = np.concatenate([self._refbuf[HOP:], ref_hop])
            hops.append(
                self._stream_hop(
                    self._winbuf,
                    hop,
                    self._errbuf,
                    self._refbuf if self._has_ref else None,
                )
            )
        return hops

    def _stream_hop(
        self,
        window: np.ndarray,
        hop: np.ndarray,
        err_window: np.ndarray | None = None,
        ref_window: np.ndarray | None = None,
    ) -> dict:
        feat = frame_features(window, self.prev_mag, self.detector.noise_rms, self.sr)
        n_fft = int(2 ** np.ceil(np.log2(len(window))))
        self.prev_mag = np.abs(np.fft.rfft(window * np.hanning(len(window)), n=n_fft))
        impulse = self.detector.update(feat)
        cls = self.classifier.predict_features(feat, impulse.probability)
        st = self.controller.step(feat, impulse, cls)
        self._prev_speech_prob = float(st.speech_prob)
        # Detection above ran on the raw primary so impulses are seen before
        # cancellation; enhancement runs on the NLMS residual.
        enh_in = err_window if err_window is not None else window
        ref_for_dsp = None
        if ref_window is not None:
            # How much of the reference survived NLMS? Minimum statistics over
            # ~1 s: speech only ever raises this ratio, so the window minimum
            # tracks the noise-only residual. The spectral stage then gets a
            # reference scaled to the residual, not to the full disturbance.
            ref_pow = float(np.mean(ref_window[-HOP:] ** 2))
            err_pow = float(np.mean(enh_in[-HOP:] ** 2))
            if ref_pow > 1e-8:
                self._rho_hist.append(float(np.clip(err_pow / ref_pow, 1e-5, 1.0)))
                if len(self._rho_hist) > 60:
                    self._rho_hist.pop(0)
                target = float(np.min(self._rho_hist))
                self._rho = 0.9 * self._rho + 0.1 * target
            ref_for_dsp = np.sqrt(self._rho) * ref_window
        # rho ≈ 0.05 (−13 dB residual) or worse → work at full strength
        aggression = float(np.clip(self._rho / 0.05, 0.15, 1.0)) if ref_window is not None else 1.0
        rec = self.enhancer.process_frame(enh_in, st.mode, ref=ref_for_dsp, aggression=aggression)
        if self._t < 0.08:
            self.enhancer.dsp.warmup_noise(enh_in)
        self._acc = self._acc + rec
        denom = np.maximum(self._w2[:HOP], 0.42)
        hop_out = self._acc[:HOP] / denom
        self._acc = np.concatenate([self._acc[HOP:], np.zeros(HOP)])
        # Speech-only gate: the tiny GRU VAD and the DSP speech features vote,
        # then the channel closes when nobody is talking. The DSP vote is taken
        # on the *enhanced* signal, never on the primary — on the raw mic it
        # reads "speech" almost permanently, because engine and rotor noise are
        # harmonic too, and with no reference mic there is nothing else to fall
        # back on. The window is pre-gate on purpose: analysing gated audio
        # would let a closed gate keep itself closed.
        self._outwin = np.concatenate([self._outwin[HOP:], hop_out])
        feat_enh = frame_features(self._outwin, self._prev_mag_enh, self.detector.noise_rms, self.sr)
        self._prev_mag_enh = np.abs(np.fft.rfft(self._outwin * np.hanning(len(self._outwin)), n=n_fft))
        tiny_vad = self.enhancer.last_vad
        gate_gain = self.gate.step(
            feat_enh.speech_prob, tiny_vad, level_db=feat_enh.rms_db, impulse_prob=st.impulse_prob
        )
        hop_out = self.gate.apply(hop_out)
        voiced = bool(self.gate.open)
        hop_out = self.agc.process(hop_out, active=voiced)
        if st.mode in ("IMPULSE_PROTECTION", "RECOVERY") or st.impulse_prob > 0.5:
            # The AGC must not undo impulse protection by making up gain on a
            # transient. Keyed off the detector as well as the mode: the FSM can
            # switch a hop late, and the transient's overlap-add tail lands in
            # the *next* hop, which would otherwise escape the ceiling.
            peak = float(np.max(np.abs(hop_out)) + 1e-12)
            if peak > IMPULSE_CEILING:
                hop_out = hop_out * (IMPULSE_CEILING / peak)

        d_in, v_in = _disturbance(hop, self.sr)
        d_out, v_out = _disturbance(hop_out, self.sr)
        reduction_db = float(10.0 * np.log10((d_in + 1e-12) / (d_out + 1e-12)))
        in_rms = float(np.sqrt(np.mean(hop ** 2) + 1e-12))
        out_rms = float(np.sqrt(np.mean(hop_out ** 2) + 1e-12))
        log = FrameLog(
            t=float(self._t),
            mode=st.mode,
            noise_class=st.noise_class,
            confidence=st.confidence,
            severity=st.severity,
            speech_prob=st.speech_prob,
            impulse_prob=st.impulse_prob,
            snr_est_db=st.snr_est_db,
            reason=st.reason,
            energy_ratio=impulse.energy_ratio,
            kurtosis=feat.kurtosis,
            flux=feat.spectral_flux,
        )
        self.live_logs.append(log)
        if len(self.live_logs) > 2500:
            self.live_logs = self.live_logs[-2000:]
        payload = {
            "t": log.t,
            "mode": log.mode,
            "reason": log.reason,
            "noise_class": log.noise_class,
            "confidence": log.confidence,
            "severity": log.severity,
            "speech_prob": log.speech_prob,
            "impulse_prob": log.impulse_prob,
            "snr_est_db": log.snr_est_db,
            "kurtosis": log.kurtosis,
            "flux": log.flux,
            "centroid": feat.spectral_centroid,
            "zcr": feat.zcr,
            "input_db": float(20.0 * np.log10(in_rms + 1e-12)),
            "output_db": float(20.0 * np.log10(out_rms + 1e-12)),
            "disturbance_in": d_in,
            "disturbance_out": d_out,
            "voice_in": v_in,
            "voice_out": v_out,
            "disturbance_reduction_db": float(np.clip(reduction_db, -8.0, 28.0)),
            "warmup": bool(self._t < 0.12),
            "ml": "onnx" if self.enhancer.tiny else "dsp",
            "ref_aided": bool(ref_window is not None),
            "nlms_db": float(self.lms.cancel_db) if ref_window is not None else 0.0,
            "agc_gain_db": self.agc.gain_db(),
            "tiny_vad": None if tiny_vad is None else float(tiny_vad),
            "gate_open": bool(self.gate.open),
            "gate_conf": float(self.gate.confidence),
            "gate_db": float(20.0 * np.log10(gate_gain + 1e-12)),
            "pcm": hop_out.astype(np.float32).tolist(),
            "pcm_in": np.clip(hop, -1.0, 1.0).astype(np.float32).tolist(),
        }
        self._t += HOP / self.sr
        return payload


def _disturbance(x: np.ndarray, sr: int) -> tuple[float, float]:
    """Energy outside vs inside the speech band. REAL, not a marketing score."""
    n = len(x)
    if n < 8:
        return 1e-12, 1e-12
    spec = np.fft.rfft(x * np.hanning(n))
    p = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    speech = (freqs >= 300.0) & (freqs <= 3400.0)
    dist = float(np.sum(p[~speech]) + 1e-12)
    voice = float(np.sum(p[speech]) + 1e-12)
    return dist, voice


def _hist(items: list[str]) -> dict[str, int]:
    d: dict[str, int] = {}
    for x in items:
        d[x] = d.get(x, 0) + 1
    return d


def _detection_latency_ms(logs: list[FrameLog], impulse_time_s: float) -> float | None:
    hits = [fr for fr in logs if fr.mode == "IMPULSE_PROTECTION" and abs(fr.t - impulse_time_s) < 0.25]
    if not hits:
        hits = [fr for fr in logs if fr.mode == "IMPULSE_PROTECTION" and fr.t >= impulse_time_s - 0.05]
    if not hits:
        return None
    first = min(hits, key=lambda fr: fr.t)
    return float(max(0.0, (first.t - impulse_time_s) * 1000.0))


def _recovery_time_ms(logs: list[FrameLog]) -> float | None:
    t_imp = next((fr.t for fr in logs if fr.mode == "IMPULSE_PROTECTION"), None)
    if t_imp is None:
        return None
    after = False
    t_rec_end = None
    for fr in logs:
        if fr.mode == "IMPULSE_PROTECTION":
            after = True
        if after and fr.mode == "RECOVERY":
            t_rec_end = fr.t
        if after and t_rec_end is not None and fr.mode not in ("IMPULSE_PROTECTION", "RECOVERY"):
            return float((fr.t - t_imp) * 1000.0)
    if t_rec_end is not None:
        return float((t_rec_end - t_imp) * 1000.0)
    return None


def run_baselines(clean: np.ndarray, noisy: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Compare raw / classical DSP / our adaptive system. REAL measurements."""
    from backend.metrics import si_snr_db, snr_db, try_stoi

    proc = AegisProcessor(sr=sr, use_ai=True)
    ours = proc.process(noisy, clean=clean)
    dsp = classical_dsp(noisy, sr)
    rows = []
    for name, est in (
        ("Raw noisy", noisy),
        ("Classical DSP (fixed Wiener)", dsp),
        ("AEGIS-COM adaptive (POC)", ours["enhanced"]),
    ):
        row = {
            "system": name,
            "snr_db": snr_db(clean, est),
            "si_snr_db": si_snr_db(clean, est),
            "stoi": try_stoi(clean, est, sr),
        }
        rows.append(row)
    return {"rows": rows, "ours": ours, "dsp": dsp}


def logs_to_dicts(logs: list[FrameLog]) -> list[dict]:
    return [asdict(fr) for fr in logs]
