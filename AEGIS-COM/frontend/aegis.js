/* AEGIS-COM live engine — same hop DSP as the Python SIL, runs in the browser
   so Vercel (static HTTPS) can host the mic demo without a long-lived server. */
(function (global) {
  const SR = 16000;
  const HOP = 256;
  const WIN = 512;
  const NFFT = 512;
  const GATE = 0.38;
  const RECOVERY = 18;
  const DWELL = {
    SPEECH_PRESERVATION: 12,
    STATIONARY: 14,
    DYNAMIC: 12,
    IMPULSE_PROTECTION: 6,
    RECOVERY: 16,
  };
  const MODE_PARAMS = {
    SPEECH_PRESERVATION: { oversubtraction: 1.0, noise_alpha: 0.98, speech_protect: 0.85, impulse_gate: 0.0, ref_over: 1.6, gain_floor: 0.1 },
    STATIONARY: { oversubtraction: 1.7, noise_alpha: 0.97, speech_protect: 0.55, impulse_gate: 0.0, ref_over: 2.4, gain_floor: 0.04 },
    DYNAMIC: { oversubtraction: 1.6, noise_alpha: 0.88, speech_protect: 0.6, impulse_gate: 0.0, ref_over: 2.2, gain_floor: 0.05 },
    IMPULSE_PROTECTION: { oversubtraction: 1.1, noise_alpha: 1.0, speech_protect: 0.9, impulse_gate: 0.97, ref_over: 3.0, gain_floor: 0.03 },
    RECOVERY: { oversubtraction: 1.3, noise_alpha: 0.995, speech_protect: 0.8, impulse_gate: 0.35, ref_over: 2.0, gain_floor: 0.06 },
  };
  const DD_BETA = 0.92;
  const AGC_TARGET = Math.pow(10, -22 / 20);
  const AGC_MAX_GAIN = Math.pow(10, 14 / 20);
  /* Output ceiling while an impulse is being handled, so the make-up gain
     cannot re-amplify a transient IMPULSE_PROTECTION just knocked down. */
  const IMPULSE_CEILING = 0.45;
  /* Speech-only gate. Floor is not silence on purpose: a hard mute sounds
     broken on a radio and hides whether the link is still alive. */
  const GATE_FLOOR_DB = -32;
  const GATE_ON = 0.45;
  const GATE_OFF = 0.32;
  const GATE_HANGOVER = 10;
  const FLOOR_WINDOW = 156; /* ~2.5 s, longer than a spoken phrase */
  /* Acoustic path from the disturbance to the PRIMARY (boom) mic, relative to
     what the REFERENCE mic hears: propagation delay, reflections, level drop.
     SIMULATED. The NLMS filter has to learn it; it is not told. */
  const PRIMARY_PATH = [0, 0, 0, 0.62, 0.28, -0.14, 0.09, -0.05, 0.03, -0.02];
  const REF_SENSOR_NOISE = 0.01;
  /* Fixed headroom on both channels. Must be constant: per-buffer peak
     normalisation makes the path time-varying and NLMS never converges. */
  const MIX_HEADROOM = 0.7;
  const NLMS_TAPS = 64;
  const NLMS_MU = 0.4;
  const CLASSES = ["quiet", "speech", "engine", "rotor", "wind", "siren", "impulse", "mixed"];
  const LOOP_N = SR * 8;

  function hann(n) {
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / Math.max(n - 1, 1)));
    return w;
  }
  function clip(x, a, b) {
    return Math.max(a, Math.min(b, x));
  }
  function mean(a) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i];
    return s / Math.max(a.length, 1);
  }
  function median(a) {
    if (!a.length) return 0;
    const s = Array.from(a).sort((x, y) => x - y);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : 0.5 * (s[m - 1] + s[m]);
  }
  function rmsOf(a) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * a[i];
    return Math.sqrt(s / a.length + 1e-12);
  }
  function normalizePeak(x, peak) {
    let m = 1e-12;
    for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i]));
    const g = peak / m;
    const o = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) o[i] = x[i] * g;
    return o;
  }

  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        const tr = re[i];
        re[i] = re[j];
        re[j] = tr;
        const ti = im[i];
        im[i] = im[j];
        im[j] = ti;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wlenRe = Math.cos(ang);
      const wlenIm = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let wRe = 1;
        let wIm = 0;
        const half = len >> 1;
        for (let j = 0; j < half; j++) {
          const ur = re[i + j];
          const ui = im[i + j];
          const vr = re[i + j + half] * wRe - im[i + j + half] * wIm;
          const vi = re[i + j + half] * wIm + im[i + j + half] * wRe;
          re[i + j] = ur + vr;
          im[i + j] = ui + vi;
          re[i + j + half] = ur - vr;
          im[i + j + half] = ui - vi;
          const nw = wRe * wlenRe - wIm * wlenIm;
          wIm = wRe * wlenIm + wIm * wlenRe;
          wRe = nw;
        }
      }
    }
  }

  function rfftWin(x, win) {
    const re = new Float64Array(NFFT);
    const im = new Float64Array(NFFT);
    const n = Math.min(x.length, WIN, NFFT);
    for (let i = 0; i < n; i++) re[i] = x[i] * (win[i] || 1);
    fft(re, im);
    return { re, im };
  }

  function irfftGain(re, im, mag, win) {
    const n2 = NFFT / 2;
    const rr = new Float64Array(NFFT);
    const ii = new Float64Array(NFFT);
    for (let k = 0; k <= n2; k++) {
      const p = Math.atan2(im[k], re[k]);
      rr[k] = mag[k] * Math.cos(p);
      ii[k] = mag[k] * Math.sin(p);
    }
    for (let k = 1; k < n2; k++) {
      rr[NFFT - k] = rr[k];
      ii[NFFT - k] = -ii[k];
    }
    for (let i = 0; i < NFFT; i++) ii[i] = -ii[i];
    fft(rr, ii);
    const rec = new Float64Array(WIN);
    const s = 1 / NFFT;
    for (let i = 0; i < WIN; i++) rec[i] = rr[i] * s * win[i];
    return rec;
  }

  function biquadLowpass(x, cutoff, sr) {
    const w0 = (2 * Math.PI * cutoff) / sr;
    const q = 0.707;
    const alpha = Math.sin(w0) / (2 * q);
    const cosw = Math.cos(w0);
    const b0 = (1 - cosw) / 2;
    const b1 = 1 - cosw;
    const b2 = b0;
    const a0 = 1 + alpha;
    const a1 = -2 * cosw;
    const a2 = 1 - alpha;
    const y = new Float64Array(x.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < x.length; i++) {
      const xn = x[i];
      const yn = (b0 / a0) * xn + (b1 / a0) * x1 + (b2 / a0) * x2 - (a1 / a0) * y1 - (a2 / a0) * y2;
      y[i] = yn;
      x2 = x1;
      x1 = xn;
      y2 = y1;
      y1 = yn;
    }
    return y;
  }

  function biquadHighpass(x, cutoff, sr) {
    const w0 = (2 * Math.PI * cutoff) / sr;
    const q = 0.707;
    const alpha = Math.sin(w0) / (2 * q);
    const cosw = Math.cos(w0);
    const b0 = (1 + cosw) / 2;
    const b1 = -(1 + cosw);
    const b2 = b0;
    const a0 = 1 + alpha;
    const a1 = -2 * cosw;
    const a2 = 1 - alpha;
    const y = new Float64Array(x.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < x.length; i++) {
      const xn = x[i];
      const yn = (b0 / a0) * xn + (b1 / a0) * x1 + (b2 / a0) * x2 - (a1 / a0) * y1 - (a2 / a0) * y2;
      y[i] = yn;
      x2 = x1;
      x1 = xn;
      y2 = y1;
      y1 = yn;
    }
    return y;
  }

  function synthEngine(n) {
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / SR;
      const rpm = 28 + 3 * Math.sin(2 * Math.PI * 0.15 * t);
      let s = 0;
      for (let k = 1; k <= 8; k++) s += (1 / k) * Math.sin(2 * Math.PI * rpm * k * t);
      x[i] = s;
    }
    const noise = new Float64Array(n);
    for (let i = 0; i < n; i++) noise[i] = Math.random() * 2 - 1;
    const rumble = biquadLowpass(noise, 180, SR);
    for (let i = 0; i < n; i++) x[i] += 0.25 * rumble[i];
    return normalizePeak(biquadHighpass(x, 40, SR), 0.9);
  }
  function synthRotor(n) {
    const noise = new Float64Array(n);
    for (let i = 0; i < n; i++) noise[i] = Math.random() * 2 - 1;
    const carrier = biquadLowpass(noise, 1800, SR);
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / SR;
      const bpf = 36 + 2.5 * Math.sin(2 * Math.PI * 0.2 * t);
      const mod = 0.55 + 0.45 * Math.sin(2 * Math.PI * bpf * t);
      x[i] = (carrier[i] + 0.35 * Math.sin(2 * Math.PI * 85 * t)) * mod;
    }
    return normalizePeak(x, 0.9);
  }
  function synthDrone(n) {
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / SR;
      const f = 1850 + 80 * Math.sin(2 * Math.PI * 6 * t);
      x[i] = Math.sin(2 * Math.PI * f * t) + 0.4 * Math.sin(2 * Math.PI * 2 * f * t) + 0.08 * (Math.random() * 2 - 1);
    }
    return normalizePeak(x, 0.85);
  }
  function synthWind(n) {
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / SR;
      x[i] = (Math.random() * 2 - 1) * (0.7 + 0.3 * Math.sin(2 * Math.PI * 0.35 * t));
    }
    return normalizePeak(biquadHighpass(biquadLowpass(x, 900, SR), 80, SR), 0.85);
  }
  function synthSiren(n) {
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / SR;
      const f = 650 + 400 * Math.sin(2 * Math.PI * 0.85 * t);
      x[i] = 0.7 * Math.sin(2 * Math.PI * f * t) + 0.3 * Math.sin(2 * Math.PI * 1.5 * f * t);
    }
    return normalizePeak(x, 0.8);
  }
  function synthImpulse(kind) {
    const n = Math.floor(0.32 * SR);
    const x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const t = i / SR;
      if (kind === "artillery") {
        x[i] = 0.7 * Math.exp(-t / 0.045) * Math.sin(2 * Math.PI * 55 * t) + 0.9 * Math.exp(-t / 0.012) * (Math.random() * 2 - 1);
      } else if (kind === "explosion") {
        x[i] = 0.85 * Math.exp(-t / 0.08) * Math.sin(2 * Math.PI * 40 * t) + 0.7 * Math.exp(-t / 0.02) * (Math.random() * 2 - 1);
      } else {
        x[i] = Math.exp(-t / 0.006) * (Math.random() * 2 - 1) + 0.45 * Math.exp(-t / 0.018) * Math.sin(2 * Math.PI * 90 * t);
      }
    }
    return normalizePeak(x, 0.99);
  }

  function OverlayBed() {
    this.kind = "none";
    this.gain = 0.42;
    this.pos = 0;
    this.lastImpulse = "";
    this._imp = null;
    this._pathTail = new Float64Array(PRIMARY_PATH.length - 1);
    this.loops = {
      engine: synthEngine(LOOP_N),
      rotor: synthRotor(LOOP_N),
      drone: synthDrone(LOOP_N),
      wind: synthWind(LOOP_N),
      siren: synthSiren(LOOP_N),
    };
    const mix = new Float64Array(LOOP_N);
    for (let i = 0; i < LOOP_N; i++) mix[i] = 0.62 * this.loops.engine[i] + 0.72 * this.loops.rotor[i];
    this.loops.engine_rotor = normalizePeak(mix, 0.9);
  }
  OverlayBed.prototype.set = function (kind, gain) {
    if (kind && kind !== this.kind) this.pos = 0;
    if (kind) this.kind = kind;
    if (gain != null) this.gain = clip(gain, 0, 1);
  };
  OverlayBed.prototype.fire = function (kind) {
    // Pre-compensate the primary-path loss and the mix headroom so the burst
    // still arrives well above the noise bed. The primary may clip, which is
    // what a real mic does on a gunshot.
    const imp = synthImpulse(kind || "gunshot");
    for (let i = 0; i < imp.length; i++) imp[i] *= 2.0;
    this._imp = imp;
    this._impPos = 0;
    this.lastImpulse = kind || "gunshot";
  };
  /* Returns { mixed, ref }.
     mixed = REAL mic + disturbance through the primary-mic path
     ref   = the disturbance as the reference mic hears it, plus sensor noise
     The two channels are deliberately NOT identical: NLMS has to learn the
     path between them, as it would on a real two-mic headset. */
  OverlayBed.prototype.mix = function (mic) {
    const src = new Float64Array(mic.length);
    const loop = this.loops[this.kind];
    if (loop && this.kind !== "none" && this.gain > 0) {
      for (let i = 0; i < src.length; i++) {
        src[i] = this.gain * loop[this.pos];
        this.pos += 1;
        if (this.pos >= loop.length) this.pos = 0;
      }
    }
    if (this._imp) {
      for (let i = 0; i < src.length && this._impPos < this._imp.length; i++, this._impPos++) {
        src[i] += this._imp[this._impPos];
      }
      if (this._impPos >= this._imp.length) this._imp = null;
    }
    let srcPeak = 1e-6;
    for (let i = 0; i < src.length; i++) srcPeak = Math.max(srcPeak, Math.abs(src[i]));
    const mixed = new Float32Array(mic.length);
    const refOut = new Float32Array(mic.length);
    const P = PRIMARY_PATH.length;
    for (let i = 0; i < mic.length; i++) {
      let acc = 0;
      for (let k = 0; k < P; k++) {
        const j = i - k;
        acc += PRIMARY_PATH[k] * (j >= 0 ? src[j] : this._pathTail[this._pathTail.length + j]);
      }
      mixed[i] = clip(MIX_HEADROOM * (mic[i] + acc), -1, 1);
      refOut[i] = MIX_HEADROOM * (src[i] + REF_SENSOR_NOISE * srcPeak * (Math.random() * 2 - 1));
    }
    const tail = new Float64Array(P - 1);
    for (let k = 0; k < P - 1; k++) {
      const j = src.length - (P - 1) + k;
      tail[k] = j >= 0 ? src[j] : this._pathTail[this._pathTail.length + j];
    }
    this._pathTail = tail;
    return { mixed, ref: refOut };
  };

  /* NLMS adaptive noise canceller — the two-mic ANC front end.
     e = primary - w·ref, with w learned on the fly. A spectral mask can only
     scale magnitudes; this actually subtracts the disturbance waveform. */
  function NLMS() {
    this.n = NLMS_TAPS;
    this.w = new Float64Array(this.n);
    this.hist = new Float64Array(this.n); // newest first
    this.cancelDb = 0;
  }
  NLMS.prototype.process = function (primary, ref, adapt, muScale) {
    const n = this.n;
    const w = this.w;
    const hist = this.hist;
    const mu = NLMS_MU * clip(muScale == null ? 1 : muScale, 0, 1);
    const doAdapt = adapt !== false && mu > 1e-3;
    const e = new Float64Array(primary.length);
    let dPow = 1e-12;
    let ePow = 1e-12;
    for (let i = 0; i < primary.length; i++) {
      for (let k = n - 1; k > 0; k--) hist[k] = hist[k - 1];
      hist[0] = ref[i] || 0;
      let y = 0;
      let p = 1e-6;
      for (let k = 0; k < n; k++) {
        y += w[k] * hist[k];
        p += hist[k] * hist[k];
      }
      const err = primary[i] - y;
      e[i] = err;
      if (doAdapt) {
        const step = (mu * err) / p;
        for (let k = 0; k < n; k++) w[k] += step * hist[k];
      }
      dPow += primary[i] * primary[i];
      ePow += err * err;
    }
    this.cancelDb = clip(10 * Math.log10(dPow / ePow), -6, 40);
    return e;
  };

  function frameFeatures(frame, prevMag, noiseRms) {
    const n = frame.length;
    let rms = rmsOf(frame);
    const rmsDb = 20 * Math.log10(rms + 1e-12);
    let zc = 0;
    for (let i = 1; i < n; i++) if (frame[i] >= 0 !== frame[i - 1] >= 0) zc += 1;
    const zcr = zc / Math.max(n - 1, 1);
    const spec = rfftWin(frame, hann(n));
    const bins = NFFT / 2 + 1;
    const mag = new Float64Array(bins);
    let magSum = 1e-12;
    for (let k = 0; k < bins; k++) {
      mag[k] = Math.hypot(spec.re[k], spec.im[k]) + 1e-12;
      magSum += mag[k];
    }
    let centroid = 0;
    let bwAcc = 0;
    for (let k = 0; k < bins; k++) {
      const f = (k * SR) / NFFT;
      centroid += f * mag[k];
    }
    centroid /= magSum;
    for (let k = 0; k < bins; k++) {
      const f = (k * SR) / NFFT;
      bwAcc += (f - centroid) * (f - centroid) * mag[k];
    }
    const bandwidth = Math.sqrt(bwAcc / magSum);
    let flux = 0;
    if (prevMag && prevMag.length === mag.length) {
      let d2 = 0;
      for (let k = 0; k < bins; k++) {
        const d = mag[k] - prevMag[k];
        if (d > 0) d2 += d * d;
      }
      flux = Math.sqrt(d2) / magSum;
    }
    let mu = 0;
    for (let i = 0; i < n; i++) mu += frame[i];
    mu /= n;
    let m2 = 0;
    let m4 = 0;
    for (let i = 0; i < n; i++) {
      const d = frame[i] - mu;
      const d2 = d * d;
      m2 += d2;
      m4 += d2 * d2;
    }
    m2 = m2 / n + 1e-12;
    const kurtosis = m4 / n / (m2 * m2);
    let low = 1e-12, speech = 1e-12, high = 1e-12;
    for (let k = 0; k < bins; k++) {
      const f = (k * SR) / NFFT;
      const e = mag[k] * mag[k];
      if (f >= 20 && f < 400) low += e;
      else if (f >= 300 && f < 3400) speech += e;
      else if (f >= 4000 && f < 7600) high += e;
    }
    const tot = low + speech + high;
    const lowR = low / tot;
    const speechR = speech / tot;
    const highR = high / tot;
    const minLag = Math.max(1, Math.floor(SR / 300));
    const maxLag = Math.min(n - 1, Math.floor(SR / 70));
    let harmonicity = 0;
    if (maxLag > minLag + 2) {
      let ac0 = 0;
      for (let i = 0; i < n; i++) ac0 += frame[i] * frame[i];
      let best = 0;
      for (let lag = minLag; lag < maxLag; lag += 2) {
        let ac = 0;
        const m = n - lag;
        for (let i = 0; i < m; i++) ac += frame[i] * frame[i + lag];
        if (ac > best) best = ac;
      }
      harmonicity = clip(best / (ac0 + 1e-12), 0, 1);
    }
    let logSum = 0;
    for (let k = 0; k < bins; k++) logSum += Math.log(mag[k]);
    const geo = Math.exp(logSum / bins);
    const arith = magSum / bins;
    const flatness = geo / (arith + 1e-12);
    const snrEst = 10 * Math.log10((rms * rms) / (noiseRms * noiseRms + 1e-12));
    let speechProb =
      0.4 * clip(speechR, 0, 1) +
      0.3 * harmonicity +
      0.15 * clip(1 - Math.abs(zcr - 0.12) / 0.3, 0, 1) +
      0.15 * clip((rmsDb + 40) / 30, 0, 1);
    speechProb = clip(speechProb, 0, 1);
    if (kurtosis > 12 && flux > 0.4) speechProb *= 0.25;
    return {
      rms,
      rms_db: rmsDb,
      zcr,
      spectral_centroid: centroid,
      spectral_bandwidth: bandwidth,
      spectral_flux: flux,
      kurtosis,
      low_ratio: lowR,
      speech_ratio: speechR,
      high_ratio: highR,
      harmonicity,
      flatness,
      snr_est_db: snrEst,
      speech_prob: speechProb,
      mag,
    };
  }

  function classify(feat, impulseProb) {
    const scores = {};
    CLASSES.forEach((c) => (scores[c] = 0.02));
    if (feat.rms_db < -42) scores.quiet += 1.4;
    if (feat.harmonicity > 0.35 && feat.speech_ratio > 0.35 && feat.kurtosis < 8) {
      scores.speech += 0.9 + 0.6 * feat.harmonicity;
    }
    if (feat.low_ratio > 0.45 && feat.spectral_centroid < 900 && feat.spectral_flux < 0.16) {
      scores.engine += 1.1 + 0.4 * feat.low_ratio;
    }
    if (feat.low_ratio > 0.35 && feat.spectral_flux >= 0.06) {
      scores.rotor += 0.9 + 0.8 * feat.spectral_flux;
      scores.mixed += 0.8;
    }
    if (feat.flatness > 0.35 && feat.spectral_centroid > 800 && feat.harmonicity < 0.25) scores.wind += 0.8;
    if (feat.spectral_flux > 0.12 && feat.spectral_flux < 0.5 && feat.spectral_centroid > 600 && feat.spectral_centroid < 3500 && feat.harmonicity > 0.2) {
      scores.siren += 0.45;
    }
    if (impulseProb > 0.45 || feat.kurtosis > 8 || feat.high_ratio > 0.3) {
      scores.impulse += 1.2 * Math.max(impulseProb, 0.4);
    }
    if (feat.spectral_flux > 0.12 && feat.low_ratio > 0.25 && feat.speech_ratio > 0.2) scores.mixed += 0.9;
    scores.impulse = Math.max(scores.impulse, 2.2 * impulseProb);
    let mx = -1e9;
    CLASSES.forEach((c) => {
      if (scores[c] > mx) mx = scores[c];
    });
    let sum = 0;
    const probs = {};
    CLASSES.forEach((c) => {
      probs[c] = Math.exp(scores[c] - mx);
      sum += probs[c];
    });
    let label = "quiet";
    let conf = 0;
    CLASSES.forEach((c) => {
      probs[c] /= sum + 1e-12;
      if (probs[c] > conf) {
        conf = probs[c];
        label = c;
      }
    });
    let severity = clip((8 - feat.snr_est_db) / 20, 0, 1);
    severity = 0.55 * severity + 0.45 * clip(feat.low_ratio + 0.5 * feat.spectral_flux, 0, 1);
    if (feat.speech_prob > 0.55 && feat.low_ratio < 0.35) severity *= 0.45;
    if (label === "quiet" || feat.rms_db < -40) severity = Math.min(severity, 0.15);
    return { label, confidence: conf, severity };
  }

  function ImpulseDetector() {
    this.slow_rms = 1e-3;
    this.fast_rms = 1e-3;
    this.db_hist = [];
    this.active = false;
    this.hold = 0;
    this.refract = 0;
    this.active_for = 0;
    this.onset_slow = 1e-3;
    this.frame_i = 0;
    this.noise_rms = 1e-3;
  }
  ImpulseDetector.prototype.update = function (feat) {
    let just_started = false;
    let just_ended = false;
    this.frame_i += 1;
    this.fast_rms = 0.55 * this.fast_rms + 0.45 * feat.rms;
    if (!this.active) {
      if (feat.rms > this.slow_rms) this.slow_rms = 0.98 * this.slow_rms + 0.02 * feat.rms;
      else this.slow_rms = 0.9 * this.slow_rms + 0.1 * feat.rms;
      this.db_hist.push(feat.rms_db);
      if (this.db_hist.length > 80) this.db_hist.shift();
    }
    this.noise_rms = this.slow_rms;
    const energy_ratio = this.fast_rms / (this.slow_rms + 1e-12);
    const med = this.db_hist.length >= 8 ? median(this.db_hist.slice(-20)) : feat.rms_db;
    const db_jump = feat.rms_db - med;
    const k_score = clip((feat.kurtosis - 3.5) / 4, 0, 1);
    const e_score = clip((energy_ratio - 1.4) / 3, 0, 1);
    const f_score = clip(feat.spectral_flux / 0.16, 0, 1);
    const h_score = clip(feat.high_ratio / 0.28, 0, 1);
    const j_score = clip(db_jump / 6, 0, 1);
    const speech_penalty = 0.45 * feat.harmonicity * feat.speech_ratio;
    let probability = clip(0.26 * e_score + 0.18 * k_score + 0.18 * f_score + 0.14 * h_score + 0.24 * j_score - speech_penalty, 0, 1);
    const warmed = this.frame_i > 24;
    const loud = energy_ratio >= 2.6 || db_jump >= 6;
    const shape = feat.kurtosis >= 5.5 || feat.high_ratio > 0.24 || (feat.spectral_flux >= 0.22 && db_jump >= 6);
    const crack = feat.high_ratio > 0.07 || feat.spectral_centroid > 2200;
    const harmonic_block = feat.harmonicity > 0.42 && feat.high_ratio < 0.08;
    const trigger = warmed && !this.active && this.refract === 0 && loud && shape && crack && !harmonic_block && probability >= 0.4;
    if (trigger) {
      this.active = true;
      this.hold = 8;
      this.active_for = 0;
      this.onset_slow = Math.max(this.slow_rms, 1e-4);
      just_started = true;
      probability = Math.max(probability, 0.85);
    } else if (this.active) {
      this.active_for += 1;
      this.hold -= 1;
      const still = feat.rms > 2.4 * this.onset_slow && (feat.kurtosis > 5 || feat.spectral_flux > 0.14 || db_jump > 4);
      if (still) this.hold = Math.max(this.hold, 1);
      if (this.active_for >= 10 || (this.hold <= 0 && !still)) {
        this.active = false;
        this.refract = 5;
        just_ended = true;
      }
      probability = Math.max(probability, this.active ? 0.7 : probability);
    } else if (this.refract > 0) this.refract -= 1;
    return { active: this.active, probability, just_started, just_ended, energy_ratio };
  };

  function Controller() {
    this.mode = "SPEECH_PRESERVATION";
    this.dwell = 0;
    this.recovery_left = 0;
    this.pre = "SPEECH_PRESERVATION";
    this.rms_buf = [];
    this.flux_ema = 0;
  }
  Controller.prototype._enter = function (m, reason) {
    if (m !== this.mode) {
      this.mode = m;
      this.dwell = 0;
    }
    return reason;
  };
  Controller.prototype._scene = function (cls, feat) {
    this.rms_buf.push(feat.rms);
    if (this.rms_buf.length > 16) this.rms_buf.shift();
    let mod = 0;
    if (this.rms_buf.length >= 8) {
      const mu = mean(this.rms_buf) + 1e-12;
      let v = 0;
      for (let i = 0; i < this.rms_buf.length; i++) {
        const d = this.rms_buf[i] - mu;
        v += d * d;
      }
      mod = Math.sqrt(v / this.rms_buf.length) / mu;
    }
    const flux = this.flux_ema;
    const speech_ok = !(feat.low_ratio > 0.42 && cls.severity > 0.22);
    if (speech_ok && (cls.label === "quiet" || (feat.speech_prob > 0.58 && cls.severity < 0.28 && flux < 0.1 && feat.low_ratio < 0.4))) {
      return "SPEECH_PRESERVATION";
    }
    const rumble = feat.low_ratio > 0.34 && flux < 0.16 && cls.label !== "rotor" && cls.label !== "mixed" && cls.label !== "siren";
    if (rumble && (cls.label === "engine" || cls.label === "speech" || cls.label === "wind" || cls.label === "quiet")) return "STATIONARY";
    if (this.mode === "STATIONARY") {
      if (cls.label === "rotor" || cls.label === "mixed" || cls.label === "siren" || (flux > 0.12 && mod > 0.16)) return "DYNAMIC";
      if (speech_ok && feat.speech_prob > 0.62 && cls.severity < 0.24 && feat.low_ratio < 0.35) return "SPEECH_PRESERVATION";
      return "STATIONARY";
    }
    if (this.mode === "DYNAMIC") {
      if (flux < 0.07 && mod < 0.08 && (cls.label === "engine" || cls.label === "wind") && cls.severity >= 0.22) return "STATIONARY";
      if (speech_ok && feat.speech_prob > 0.62 && cls.severity < 0.22 && flux < 0.08 && feat.low_ratio < 0.32) return "SPEECH_PRESERVATION";
      return "DYNAMIC";
    }
    if ((cls.label === "engine" || cls.label === "wind") && flux < 0.11 && mod < 0.12) return "STATIONARY";
    if (cls.label === "rotor" || cls.label === "siren" || cls.label === "mixed" || flux >= 0.11 || mod > 0.12) return "DYNAMIC";
    if (speech_ok && cls.label === "speech" && cls.severity < 0.4) return "SPEECH_PRESERVATION";
    if (speech_ok && cls.severity < 0.28 && feat.low_ratio < 0.38) return "SPEECH_PRESERVATION";
    return flux >= 0.1 || mod > 0.12 ? "DYNAMIC" : "STATIONARY";
  };
  Controller.prototype.step = function (feat, impulse, cls) {
    let reason = "hold";
    this.flux_ema = 0.8 * this.flux_ema + 0.2 * feat.spectral_flux;
    if (impulse.just_started || (impulse.active && this.mode !== "IMPULSE_PROTECTION")) {
      if (this.mode !== "IMPULSE_PROTECTION" && this.mode !== "RECOVERY") this.pre = this.mode;
      reason = this._enter("IMPULSE_PROTECTION", "impulse onset");
    } else if (this.mode === "IMPULSE_PROTECTION") {
      if (impulse.just_ended || !impulse.active) {
        this.recovery_left = RECOVERY;
        reason = this._enter("RECOVERY", "impulse ended → recovery");
      } else reason = "impulse hold";
    } else if (this.mode === "RECOVERY") {
      this.recovery_left -= 1;
      if (this.recovery_left <= 0 && this.dwell >= DWELL.RECOVERY) {
        let target = this._scene(cls, feat);
        if (cls.severity < 0.32) target = "SPEECH_PRESERVATION";
        reason = this._enter(target, "recovery complete");
      } else reason = "recovery hold";
    } else {
      const target = this._scene(cls, feat);
      const gated = cls.confidence < GATE && target !== this.mode;
      if (gated) reason = `confidence gate (${cls.confidence.toFixed(2)}) — hold ${this.mode}`;
      else if (target !== this.mode && this.dwell >= (DWELL[this.mode] || 4)) reason = this._enter(target, `scene → ${target} (${cls.label})`);
      else if (target !== this.mode) reason = "min-dwell hold";
      else reason = "stable";
    }
    this.dwell += 1;
    return {
      mode: this.mode,
      reason,
      noise_class: cls.label,
      confidence: cls.confidence,
      severity: cls.severity,
      speech_prob: feat.speech_prob,
      impulse_prob: impulse.probability,
      snr_est_db: feat.snr_est_db,
    };
  };

  function Wiener() {
    this.win = hann(WIN);
    const bins = NFFT / 2 + 1;
    this.noise_psd = new Float64Array(bins);
    this.ref_psd = new Float64Array(bins);
    this.prev_gain = new Float64Array(bins);
    this.prev_pow = new Float64Array(bins);
    this.prev_frame = new Float64Array(WIN);
    for (let i = 0; i < bins; i++) {
      this.noise_psd[i] = 1e-6;
      this.prev_gain[i] = 1;
    }
  }
  Wiener.prototype.warmup = function (x) {
    if (x.length < WIN) return;
    const spec = rfftWin(x.subarray ? x.subarray(0, WIN) : x, this.win);
    for (let k = 0; k < this.noise_psd.length; k++) {
      this.noise_psd[k] = Math.max(spec.re[k] * spec.re[k] + spec.im[k] * spec.im[k], 1e-8);
    }
  };
  /* `aggression` in (0,1] scales suppression down when NLMS already removed the
     disturbance. Suppressing noise that is no longer there damages the voice. */
  Wiener.prototype.process = function (frame, mode, ref, aggression) {
    const p = MODE_PARAMS[mode] || MODE_PARAMS.STATIONARY;
    const ag = clip(aggression == null ? 1 : aggression, 0.05, 1);
    const spec = rfftWin(frame, this.win);
    const bins = this.noise_psd.length;
    const pow = new Float64Array(bins);
    let meanP = 0;
    let meanN = 0;
    for (let k = 0; k < bins; k++) {
      pow[k] = spec.re[k] * spec.re[k] + spec.im[k] * spec.im[k];
      meanP += pow[k];
      meanN += this.noise_psd[k];
    }
    meanP /= bins;
    meanN /= bins;

    if (ref) {
      const rs = rfftWin(ref, this.win);
      for (let k = 0; k < bins; k++) {
        const pr = rs.re[k] * rs.re[k] + rs.im[k] * rs.im[k];
        this.ref_psd[k] = 0.5 * this.ref_psd[k] + 0.5 * pr; // aligned, so barely smoothed
      }
    } else {
      this.ref_psd.fill(0);
    }

    const alpha = p.noise_alpha;
    if (alpha < 0.999) {
      const allow = mode === "STATIONARY" || mode === "DYNAMIC" ? meanP < 1.8 * meanN : false;
      for (let k = 0; k < bins; k++) {
        const speechLike = pow[k] > 2.5 * this.noise_psd[k];
        if (!speechLike || allow) this.noise_psd[k] = alpha * this.noise_psd[k] + (1 - alpha) * pow[k];
      }
    }

    const over = 1 + (p.oversubtraction - 1) * ag;
    const refOver = p.ref_over * ag;
    const floor = p.gain_floor + (1 - p.gain_floor) * 0.25 * (1 - ag);
    const raw = new Float64Array(bins);
    for (let k = 0; k < bins; k++) {
      const f = (k * SR) / NFFT;
      const refPart = refOver * this.ref_psd[k];
      const noise = Math.max(over * this.noise_psd[k] + refPart, 1e-10);
      // decision-directed a-priori SNR → Wiener gain
      const gamma = pow[k] / noise;
      const xiPrev = (this.prev_gain[k] * this.prev_gain[k] * this.prev_pow[k]) / noise;
      const xi = DD_BETA * xiPrev + (1 - DD_BETA) * Math.max(gamma - 1, 0);
      let g = xi / (1 + xi);
      // protect the voice band, but not in bins the reference flags as noise
      if (f >= 300 && f <= 3400) {
        const prot = p.speech_protect * (1 - clip(refPart / noise, 0, 1));
        g = prot * Math.max(g, 0.35) + (1 - prot) * g;
      }
      raw[k] = clip(g, floor, 1);
    }
    const gain = new Float64Array(bins);
    for (let k = 0; k < bins; k++) {
      // 3-tap frequency smoothing: isolated surviving bins are musical noise
      const lo = raw[Math.max(k - 1, 0)];
      const hi = raw[Math.min(k + 1, bins - 1)];
      const g = 0.6 * (0.25 * lo + 0.5 * raw[k] + 0.25 * hi) + 0.4 * this.prev_gain[k];
      this.prev_gain[k] = g;
      this.prev_pow[k] = pow[k];
      gain[k] = Math.sqrt(pow[k]) * g;
    }
    const rec = irfftGain(spec.re, spec.im, gain, this.win);
    const gate = p.impulse_gate;
    if (gate > 0) {
      // Peak-limit the residual transient only. The old version cross-faded to
      // 0.12 * previous frame, which muted the voice for ~100 ms too.
      let prevPeak = 1e-12;
      let recPeak = 1e-12;
      for (let i = 0; i < WIN; i++) {
        prevPeak = Math.max(prevPeak, Math.abs(this.prev_frame[i]));
        recPeak = Math.max(recPeak, Math.abs(rec[i]));
      }
      const cap = Math.max(prevPeak * (1.4 - 0.6 * gate), 0.06);
      if (recPeak > cap) {
        const s = cap / recPeak;
        for (let i = 0; i < WIN; i++) rec[i] *= s;
      }
    }
    this.prev_frame = rec;
    return rec;
  };

  /* Speech-only output gate. In the browser only the DSP speech features vote
     (the GRU VAD runs server-side via ONNX), so this is a little more
     permissive than the TinyML path. Hysteresis + hangover stop the audio from
     sounding chopped between syllables. */
  function SpeechGate() {
    this.floor = Math.pow(10, GATE_FLOOR_DB / 20);
    this.gain = 1;
    this.open = false;
    this.hold = 0;
    this.confidence = 0;
    this.floorDb = -70;
    this.levels = [];
  }
  /* Speech-likeness alone cannot open the gate: near silence still scores ~0.35,
     above the release threshold, so the gate would latch open forever. The frame
     must also sit above the tracked residual floor. */
  SpeechGate.prototype.step = function (dspSpeechProb, tinyVad, levelDb, impulseProb) {
    let conf = dspSpeechProb;
    if (tinyVad != null) conf = Math.max(conf, tinyVad);
    // A gunshot is loud, broadband and speech-shaped enough to fool the votes,
    // so an impulse may not open the gate (it cannot close an open one either).
    conf *= clip(1 - (impulseProb || 0), 0, 1);
    if (levelDb != null) {
      /* Sliding-window minimum, not an all-time minimum: one quiet moment at
         start-up would pin the floor forever and disable this test. */
      this.levels.push(levelDb);
      if (this.levels.length > FLOOR_WINDOW) this.levels.shift();
      this.floorDb = Math.min.apply(null, this.levels);
      conf *= clip((levelDb - this.floorDb - 1.5) / 5, 0, 1);
    }
    this.confidence = conf;
    if (this.open) {
      if (conf < GATE_OFF) {
        this.hold -= 1;
        if (this.hold <= 0) this.open = false;
      } else this.hold = GATE_HANGOVER;
    } else if (conf >= GATE_ON) {
      this.open = true;
      this.hold = GATE_HANGOVER;
    }
    const target = this.open ? 1 : this.floor;
    /* fast open so onsets survive; close over ~150 ms */
    const rate = target > this.gain ? 0.5 : 0.15;
    this.gain += rate * (target - this.gain);
    return this.gain;
  };

  /* Slow, capped make-up gain. Suppression removes energy, so without this the
     cleaned voice sounds clean but far away. */
  function OutputAGC() {
    this.gain = 1;
  }
  OutputAGC.prototype.process = function (hop, active) {
    const rms = rmsOf(hop);
    let desired = this.gain;
    if (active && rms > 1e-4) desired = clip(AGC_TARGET / rms, 0.5, AGC_MAX_GAIN);
    this.gain = 0.92 * this.gain + 0.08 * desired;
    let peak = 1e-12;
    for (let i = 0; i < hop.length; i++) {
      hop[i] *= this.gain;
      peak = Math.max(peak, Math.abs(hop[i]));
    }
    const s = peak > 0.98 ? 0.98 / peak : 1;
    for (let i = 0; i < hop.length; i++) hop[i] = clip(hop[i] * s, -1, 1);
    return hop;
  };

  function bandEnergy(x) {
    const n = x.length;
    const w = hann(n);
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) re[i] = x[i] * w[i];
    /* n may not be power of two — use NFFT hop path: pad/truncate to 256 via rfftWin on hop */
    const spec = rfftWin(x, hann(Math.min(n, WIN)));
    let dist = 1e-12;
    let voice = 1e-12;
    const bins = NFFT / 2 + 1;
    for (let k = 0; k < bins; k++) {
      const f = (k * SR) / NFFT;
      const e = spec.re[k] * spec.re[k] + spec.im[k] * spec.im[k];
      if (f >= 300 && f <= 3400) voice += e;
      else dist += e;
    }
    return { dist, voice };
  }

  function AegisLive() {
    this.bed = new OverlayBed();
    this.reset();
  }
  AegisLive.prototype.reset = function () {
    this.detector = new ImpulseDetector();
    this.controller = new Controller();
    this.dsp = new Wiener();
    this.lms = new NLMS();
    this.gate = new SpeechGate();
    this.agc = new OutputAGC();
    this.prevMag = null;
    this.prevMagEnh = null;
    this.outWin = new Float64Array(WIN);
    this.pending = new Float64Array(0);
    this.refPending = new Float64Array(0);
    this.winbuf = new Float64Array(WIN);
    this.errbuf = new Float64Array(WIN);
    this.refbuf = new Float64Array(WIN);
    this.rho = 0.25;
    this.rhoHist = [];
    this.prevSpeechProb = 0;
    this.acc = new Float64Array(WIN);
    this.w2 = hann(WIN);
    for (let i = 0; i < WIN; i++) this.w2[i] *= this.w2[i];
    this.t = 0;
  };
  AegisLive.prototype.setOverlay = function (kind, gain) {
    this.bed.set(kind, gain);
  };
  AegisLive.prototype.fireImpulse = function (kind) {
    this.bed.fire(kind);
  };
  AegisLive.prototype.process = function (mic) {
    const out = this.bed.mix(mic);
    const merged = new Float64Array(this.pending.length + out.mixed.length);
    merged.set(this.pending);
    merged.set(out.mixed, this.pending.length);
    const mergedRef = new Float64Array(this.refPending.length + out.ref.length);
    mergedRef.set(this.refPending);
    mergedRef.set(out.ref, this.refPending.length);
    let off = 0;
    const hops = [];
    while (merged.length - off >= HOP) {
      const hop = merged.subarray(off, off + HOP);
      const refHop = mergedRef.subarray(off, off + HOP);
      off += HOP;
      // Freeze adaptation on an active impulse (previous hop's decision, since
      // this runs ahead of the detector) and slow it while the operator talks.
      const errHop = this.lms.process(
        hop,
        refHop,
        this.controller.mode !== "IMPULSE_PROTECTION",
        1 - 0.95 * this.prevSpeechProb
      );
      const next = new Float64Array(WIN);
      next.set(this.winbuf.subarray(HOP));
      next.set(hop, WIN - HOP);
      this.winbuf = next;
      const nextErr = new Float64Array(WIN);
      nextErr.set(this.errbuf.subarray(HOP));
      nextErr.set(errHop, WIN - HOP);
      this.errbuf = nextErr;
      const nextRef = new Float64Array(WIN);
      nextRef.set(this.refbuf.subarray(HOP));
      nextRef.set(refHop, WIN - HOP);
      this.refbuf = nextRef;
      hops.push(this._hop(this.winbuf, hop, this.errbuf, this.refbuf));
    }
    this.pending = merged.subarray(off);
    this.refPending = mergedRef.subarray(off);
    return hops;
  };
  AegisLive.prototype._hop = function (window, hop, errWindow, refWindow) {
    const feat = frameFeatures(window, this.prevMag, this.detector.noise_rms);
    this.prevMag = feat.mag;
    const impulse = this.detector.update(feat);
    const cls = classify(feat, impulse.probability);
    const st = this.controller.step(feat, impulse, cls);
    this.prevSpeechProb = st.speech_prob;
    // Detection ran on the raw primary so impulses are seen before
    // cancellation; enhancement runs on the NLMS residual.
    const enhIn = errWindow || window;
    let refForDsp = null;
    let aggression = 1;
    if (refWindow) {
      // How much of the reference survived NLMS? Minimum statistics over ~1 s:
      // speech only raises this ratio, so the window minimum tracks the
      // noise-only residual.
      let refPow = 0;
      let errPow = 0;
      for (let i = WIN - HOP; i < WIN; i++) {
        refPow += refWindow[i] * refWindow[i];
        errPow += enhIn[i] * enhIn[i];
      }
      refPow /= HOP;
      errPow /= HOP;
      if (refPow > 1e-8) {
        this.rhoHist.push(clip(errPow / refPow, 1e-5, 1));
        if (this.rhoHist.length > 60) this.rhoHist.shift();
        this.rho = 0.9 * this.rho + 0.1 * Math.min.apply(null, this.rhoHist);
      }
      const s = Math.sqrt(this.rho);
      refForDsp = new Float64Array(WIN);
      for (let i = 0; i < WIN; i++) refForDsp[i] = s * refWindow[i];
      aggression = clip(this.rho / 0.05, 0.15, 1);
    }
    const rec = this.dsp.process(enhIn, st.mode, refForDsp, aggression);
    if (this.t < 0.08) this.dsp.warmup(enhIn);
    for (let i = 0; i < WIN; i++) this.acc[i] += rec[i];
    let hopOut = new Float32Array(HOP);
    for (let i = 0; i < HOP; i++) {
      const den = Math.max(this.w2[i], 0.42);
      hopOut[i] = this.acc[i] / den;
    }
    const acc2 = new Float64Array(WIN);
    acc2.set(this.acc.subarray(HOP));
    this.acc = acc2;
    /* The DSP vote is taken on the *enhanced* signal, never on the raw mic,
       where it reads "speech" almost permanently. Pre-gate on purpose:
       analysing gated audio would let a closed gate keep itself closed. */
    this.outWin.copyWithin(0, HOP);
    this.outWin.set(hopOut, WIN - HOP);
    const featEnh = frameFeatures(this.outWin, this.prevMagEnh, this.detector.noise_rms);
    this.prevMagEnh = featEnh.mag;
    const gateGain = this.gate.step(featEnh.speech_prob, null, featEnh.rms_db, st.impulse_prob);
    for (let i = 0; i < HOP; i++) hopOut[i] *= gateGain;
    hopOut = this.agc.process(hopOut, this.gate.open);
    /* Keyed off the detector as well as the mode: the FSM can switch a hop late,
       and the transient's overlap-add tail lands in the *next* hop. */
    if (st.mode === "IMPULSE_PROTECTION" || st.mode === "RECOVERY" || st.impulse_prob > 0.5) {
      let pk = 1e-12;
      for (let i = 0; i < HOP; i++) pk = Math.max(pk, Math.abs(hopOut[i]));
      if (pk > IMPULSE_CEILING) {
        const s = IMPULSE_CEILING / pk;
        for (let i = 0; i < HOP; i++) hopOut[i] *= s;
      }
    }
    const din = bandEnergy(hop);
    const dout = bandEnergy(hopOut);
    const reduction = clip(10 * Math.log10((din.dist + 1e-12) / (dout.dist + 1e-12)), -8, 28);
    const inRms = rmsOf(hop);
    const outRms = rmsOf(hopOut);
    const payload = {
      t: this.t,
      mode: st.mode,
      reason: st.reason,
      noise_class: st.noise_class,
      confidence: st.confidence,
      severity: st.severity,
      speech_prob: st.speech_prob,
      impulse_prob: st.impulse_prob,
      snr_est_db: st.snr_est_db,
      kurtosis: feat.kurtosis,
      flux: feat.spectral_flux,
      centroid: feat.spectral_centroid,
      zcr: feat.zcr,
      input_db: 20 * Math.log10(inRms + 1e-12),
      output_db: 20 * Math.log10(outRms + 1e-12),
      disturbance_in: din.dist,
      disturbance_out: dout.dist,
      voice_in: din.voice,
      voice_out: dout.voice,
      disturbance_reduction_db: reduction,
      ref_aided: !!refWindow,
      nlms_db: refWindow ? this.lms.cancelDb : 0,
      agc_gain_db: 20 * Math.log10(this.agc.gain + 1e-12),
      tiny_vad: null,
      gate_open: this.gate.open,
      gate_conf: this.gate.confidence,
      gate_db: 20 * Math.log10(this.gate.gain + 1e-12),
      warmup: this.t < 0.12,
      pcm: hopOut,
      pcm_in: Float32Array.from(hop),
    };
    this.t += HOP / SR;
    return payload;
  };

  global.AegisLive = AegisLive;
})(typeof window !== "undefined" ? window : globalThis);
