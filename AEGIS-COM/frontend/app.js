const MODE_COLORS = {
  SPEECH_PRESERVATION: "#1F6B3A",
  STATIONARY: "#3D7A45",
  DYNAMIC: "#C4A035",
  IMPULSE_PROTECTION: "#9B2D2D",
  RECOVERY: "#6B8F4E",
};

const $ = (id) => document.getElementById(id);
let STATE = null;
const LIVE_SR = 16000;
const DISTURBANCES = [
  ["none", "Mic only"],
  ["engine", "Vehicle engine"],
  ["rotor", "Helicopter rotor"],
  ["drone", "UAV drone"],
  ["wind", "Field wind"],
  ["siren", "Siren"],
  ["engine_rotor", "Engine + rotor"],
];
const IMPULSES = [
  ["gunshot", "Gunshot-like"],
  ["artillery", "Artillery-like"],
  ["explosion", "Blast-like"],
];
let overlayKind = "none";

$("snr").addEventListener("input", () => {
  $("snrVal").textContent = `${$("snr").value} dB`;
});

$("btnDemo").addEventListener("click", () => run("/api/demo", null));
$("btnProcess").addEventListener("click", () => {
  const fd = new FormData();
  const f = $("wavFile").files[0];
  if (f) fd.append("wav", f);
  fd.append("noise_kind", $("noiseKind").value);
  fd.append("snr_db", $("snr").value);
  fd.append("add_impulse", $("forceImpulse").checked ? "true" : "false");
  run("/api/process", fd);
});

async function run(url, body) {
  setBusy(true);
  $("status").textContent = "Running lab pipeline…";
  try {
    const res = await fetch(url, { method: "POST", body });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    STATE = await res.json();
    render(STATE);
    $("status").textContent = STATE.kind || "Processed.";
    $("labPlots").classList.remove("hidden");
  } catch (err) {
    $("status").textContent = `Failed: ${err.message}`;
  } finally {
    setBusy(false);
  }
}

function setBusy(b) {
  $("btnDemo").disabled = b;
  $("btnProcess").disabled = b;
}

$("wavFile").addEventListener("change", () => {
  const f = $("wavFile").files[0];
  $("status").textContent = f ? `Selected: ${f.name}` : "No file selected.";
});

function fmt(v, unit = "", digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return `${Number(v).toFixed(digits)} ${unit}`.trim();
}

function optMetric(d) {
  if (d && d.available && d.value != null) return Number(d.value).toFixed(3);
  if (d && d.target) return `n/a · TARGET ${d.target}`;
  return "n/a";
}

function render(s) {
  $("yourTake").classList.remove("hidden");
  $("audBefore").src = s.audio.before;
  $("audAfter").src = s.audio.after;
  $("imgWave").src = s.plots.waveform;
  $("imgSpec").src = s.plots.spectrogram;
  const m = s.metrics || {};
  $("mIn").textContent = fmt(m.input_snr_db, "dB");
  $("mOut").textContent = fmt(m.output_snr_db, "dB");
  $("mDelta").textContent = fmt(m.snr_improvement_db, "dB");
  if (m.input_si_snr_db != null) {
    $("mSi").textContent = `${fmt(m.input_si_snr_db)} → ${fmt(m.output_si_snr_db)}`;
  }
  $("mStoi").textContent = optMetric(m.stoi);
  $("mPesq").textContent = optMetric(m.pesq);
  $("mPeak").textContent = fmt(m.impulse_peak_reduction_db, "dB");
  $("mDet").textContent = `${fmt(m.impulse_detection_latency_ms, "ms", 0)} / ${fmt(m.recovery_time_ms, "ms", 0)}`;
  $("mRtf").textContent = m.rtf != null ? `${Number(m.rtf).toFixed(3)}×` : "n/a";
  const box = $("stages");
  box.classList.toggle("hidden", !s.stages || !s.stages.length);
  box.innerHTML = (s.stages || [])
    .map((st) => `<div class="stage"><b>${st.label}</b><div class="pill ${st.mode}">${labelMode(st.mode)}</div></div>`)
    .join("");
  drawTimeline(s);
  applyFrame(frameAt(s, (s.impulse_time_s != null ? s.impulse_time_s : 0)));
  bindAudio(s);
}

function labelMode(mode) {
  return String(mode || "—").replaceAll("_", " ");
}

function frameAt(s, t) {
  const logs = s.logs || [];
  if (!logs.length) return null;
  let best = logs[0];
  let d = Math.abs(best.t - t);
  for (const fr of logs) {
    const dd = Math.abs(fr.t - t);
    if (dd < d) {
      best = fr;
      d = dd;
    }
  }
  return best;
}

function applyFrame(fr) {
  if (!fr) return;
  const pill = $("modePill");
  pill.className = `pill ${fr.mode}`;
  pill.textContent = labelMode(fr.mode);
  $("modeReason").textContent = fr.reason || "";
  $("kClass").textContent = fr.noise_class || "—";
  $("kSev").textContent = fmt(fr.severity, "", 2);
  $("kSpeech").textContent = fmt(fr.speech_prob, "", 2);
  $("kImp").textContent = fmt(fr.impulse_prob, "", 2);
  $("kSnr").textContent = fmt(fr.snr_est_db, "dB", 1);
  $("kTime").textContent = fmt(fr.t, "s", 2);
}

function drawTimeline(s) {
  const canvas = $("timeline");
  const logs = s.logs || [];
  const w = canvas.clientWidth || 900;
  canvas.width = w * 2;
  canvas.height = 88 * 2;
  const ctx = canvas.getContext("2d");
  ctx.scale(2, 2);
  ctx.fillStyle = "#243826";
  ctx.fillRect(0, 0, w, 88);
  if (!logs.length) return;
  const t0 = logs[0].t;
  const t1 = Math.max(s.duration || logs[logs.length - 1].t, logs[logs.length - 1].t);
  const span = Math.max(t1 - t0, 0.01);
  for (let i = 0; i < logs.length - 1; i++) {
    const x0 = ((logs[i].t - t0) / span) * w;
    const x1 = ((logs[i + 1].t - t0) / span) * w;
    ctx.fillStyle = MODE_COLORS[logs[i].mode] || "#5a7188";
    ctx.fillRect(x0, 10, Math.max(x1 - x0, 1), 50);
  }
}

function bindAudio(s) {
  const a = $("audAfter");
  const b = $("audBefore");
  const head = $("playhead");
  const move = (el) => {
    const t = el.currentTime || 0;
    const dur = s.duration || el.duration || 1;
    head.style.display = "block";
    head.style.left = `${(t / dur) * 100}%`;
    applyFrame(frameAt(s, t));
  };
  a.ontimeupdate = () => move(a);
  b.ontimeupdate = () => move(b);
  a.onplay = () => b.pause();
  b.onplay = () => a.pause();
}

let micState = {
  running: false,
  listen: "clean", // "clean" | "raw" | "off"
  play: null,      // monitor jitter buffer
  capRs: null,     // continuous capture resampler
  live: null,
  ws: null,
  stream: null,
  capCtx: null,
  playCtx: null,
  procNode: null,
  srcNode: null,
  uiPending: false,
  pendingHop: null,
  tSample: 0,
  inRing: new Float32Array(LIVE_SR),
  outRing: new Float32Array(LIVE_SR),
  inPos: 0,
  outPos: 0,
  liveLogs: [],
  recIn: [],
  recOut: [],
};

function sendLiveConfig() {
  const gain = Number($("distGain").value) / 100;
  if (micState.live) micState.live.setOverlay(overlayKind, gain);
  if (micState.ws && micState.ws.readyState === 1) {
    micState.ws.send(JSON.stringify({ type: "config", kind: overlayKind, gain }));
  }
}

function sendImpulse(kind) {
  if (micState.live) {
    micState.live.fireImpulse(kind);
    setOverlayLabel(overlayKind, ` · impulse ${kind}`);
    return;
  }
  if (micState.ws && micState.ws.readyState === 1) {
    micState.ws.send(JSON.stringify({ type: "impulse", kind }));
    setOverlayLabel(overlayKind, ` · impulse ${kind}`);
    return;
  }
  $("liveStatus").textContent = "Start the mic first, then fire an impulse into the mix.";
}

function markChips(rowId, kind) {
  document.querySelectorAll(`#${rowId} button`).forEach((b) => {
    b.classList.toggle("on", b.dataset.kind === kind);
  });
}

function setOverlayLabel(kind, extra) {
  const lab = (DISTURBANCES.find((d) => d[0] === kind) || [kind, kind])[1];
  $("overlayNow").textContent = kind === "none" ? extra || "Mic only" : `SIMULATED · ${lab}${extra || ""}`;
}

(function buildChips() {
  const row = $("noiseChips");
  DISTURBANCES.forEach(([id, lab]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.kind = id;
    b.textContent = lab;
    if (id === "none") b.classList.add("on");
    b.addEventListener("click", () => {
      overlayKind = id;
      markChips("noiseChips", id);
      sendLiveConfig();
      setOverlayLabel(id);
    });
    row.appendChild(b);
  });
  const irow = $("impulseChips");
  irow.classList.add("impulse");
  IMPULSES.forEach(([id, lab]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.kind = id;
    b.textContent = lab;
    b.addEventListener("click", () => sendImpulse(id));
    irow.appendChild(b);
  });
})();

$("distGain").addEventListener("input", () => {
  $("distGainVal").textContent = `${$("distGain").value}%`;
  sendLiveConfig();
});

function dbPct(db) {
  return Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
}

/* Continuous resampler. resampleLinear() below restarts its phase at zero for
   every block and drops the tail fraction, which is fine for a whole file but
   puts a discontinuity at every block boundary of a live stream — a buzz at the
   block rate — and slowly loses samples. This one carries the fractional
   position and the last sample across calls. */
function Resampler(fromRate, toRate) {
  this.step = fromRate / toRate;
  this.pos = 0; // read position; -1 <= pos < 0 means "between prev and input[0]"
  this.prev = 0;
}
Resampler.prototype.process = function (input) {
  const n = input.length;
  if (this.step === 1 || n === 0) return input;
  const out = new Float32Array(Math.ceil((n - this.pos) / this.step) + 1);
  let k = 0;
  let p = this.pos;
  while (p < n - 1) {
    const i0 = Math.floor(p);
    const f = p - i0;
    const a = i0 < 0 ? this.prev : input[i0];
    const b = input[i0 + 1];
    out[k++] = a + (b - a) * f;
    p += this.step;
  }
  this.pos = p - n;
  this.prev = input[n - 1];
  return out.subarray(0, k);
};

function resampleLinear(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const n = Math.max(1, Math.floor(input.length / ratio));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const f = x - i0;
    const a = input[i0] || 0;
    const b = input[i0 + 1] || a;
    out[i] = a + (b - a) * f;
  }
  return out;
}

function pushRing(ring, hop, key) {
  let p = micState[key] % ring.length;
  const first = Math.min(hop.length, ring.length - p);
  ring.set(hop.subarray(0, first), p);
  if (first < hop.length) ring.set(hop.subarray(first), 0);
  micState[key] = (p + hop.length) % ring.length;
}

function drawRing(canvasId, ring, color, pos) {
  const canvas = $(canvasId);
  const w = canvas.clientWidth || 400;
  const h = 88;
  // Assigning canvas.width reallocates the backing store and resets state, so
  // only do it when the size actually changed.
  if (canvas.width !== w * 2 || canvas.height !== h * 2) {
    canvas.width = w * 2;
    canvas.height = h * 2;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(2, 0, 0, 2, 0, 0);
  ctx.fillStyle = "#243826";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  const origin = pos % ring.length;
  for (let x = 0; x < w; x++) {
    const idx = (origin + Math.floor((x / w) * ring.length)) % ring.length;
    const y = h / 2 - ring[idx] * (h * 0.42);
    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function clarityPct(voice, dist) {
  return (100 * voice) / (voice + dist + 1e-12);
}

function applyLiveHop(h) {
  const pcm = Float32Array.from(h.pcm || []);
  const pcmIn = Float32Array.from(h.pcm_in && h.pcm_in.length ? h.pcm_in : h.pcm || []);
  micState.recOut.push(pcm);
  micState.recIn.push(pcmIn);
  if (micState.recOut.length > 8000) micState.recOut.shift();
  if (micState.recIn.length > 8000) micState.recIn.shift();
  if (h.warmup) return;

  /* Audio first. The UI work below used to run on every hop — 62 times a
     second of DOM writes, two full canvas reallocations and a timeline redraw —
     which starves the audio callbacks on the same thread and makes the monitor
     stutter. Feed the monitor, then let rendering catch up at frame rate. */
  if (micState.listen !== "off") playHop(micState.listen === "raw" ? pcmIn : pcm);
  pushRing(micState.inRing, pcmIn, "inPos");
  pushRing(micState.outRing, pcm, "outPos");
  micState.liveLogs.push(h);
  if (micState.liveLogs.length > 900) micState.liveLogs.shift();

  micState.pendingHop = h;
  if (!micState.uiPending) {
    micState.uiPending = true;
    requestAnimationFrame(renderLive);
  }
}

function renderLive() {
  micState.uiPending = false;
  const h = micState.pendingHop;
  if (!h) return;
  // ~33 Hz is indistinguishable from 60 for meters and halves the work. No
  // need to re-arm: the next hop is 16 ms away and will schedule another frame.
  const t = performance.now();
  if (t - (micState.lastRender || 0) < 30) return;
  micState.lastRender = t;
  applyFrame(h);
  const cin = clarityPct(h.voice_in, h.disturbance_in);
  const cout = clarityPct(h.voice_out, h.disturbance_out);
  $("liveInDb").textContent = fmt(h.input_db, "dBFS", 1);
  $("liveOutDb").textContent = fmt(h.output_db, "dBFS", 1);
  $("liveRed").textContent = fmt(h.disturbance_reduction_db, "dB", 1);
  $("liveClearIn").textContent = `${cin.toFixed(0)}% speech-band`;
  $("liveClearOut").textContent = `${cout.toFixed(0)}% speech-band`;
  $("liveCent").textContent = `${fmt(h.centroid, "Hz", 0)} / ${fmt(h.zcr, "", 2)}`;
  const nlms = h.ref_aided ? h.nlms_db : null;
  $("liveNlms").textContent = nlms == null ? "no reference mic" : fmt(nlms, "dB", 1);
  $("barNlms").style.width = `${Math.max(0, Math.min(100, ((nlms || 0) / 30) * 100))}%`;
  const conf = h.gate_conf == null ? 0 : h.gate_conf;
  const vadTxt = h.tiny_vad == null ? "DSP vote" : `GRU ${h.tiny_vad.toFixed(2)}`;
  $("liveGate").textContent = h.gate_open
    ? `SPEECH · open (${vadTxt})`
    : `no speech · ${fmt(h.gate_db, "dB", 0)} (${vadTxt})`;
  $("barGate").style.width = `${Math.max(3, Math.min(100, conf * 100))}%`;
  $("barIn").style.width = `${dbPct(h.input_db)}%`;
  $("barOut").style.width = `${dbPct(h.output_db)}%`;
  $("barRed").style.width = `${Math.max(0, Math.min(100, (Math.max(0, h.disturbance_reduction_db) / 18) * 100))}%`;
  $("barClearIn").style.width = `${Math.max(4, Math.min(100, cin))}%`;
  $("barClearOut").style.width = `${Math.max(4, Math.min(100, cout))}%`;

  drawRing("waveIn", micState.inRing, "#DCE8D4", micState.inPos);
  drawRing("waveOut", micState.outRing, "#FFFFFF", micState.outPos);

  // The timeline redraws 900 log entries; 10 Hz is plenty for a mode trace.
  const now = performance.now();
  if (now - (micState.lastTimeline || 0) > 100) {
    micState.lastTimeline = now;
    drawTimeline({ logs: micState.liveLogs, duration: h.t, impulse_time_s: null });
  }
}

/* Monitor playback is a jitter buffer, not one scheduled AudioBufferSource per
   hop. Hops arrive in bursts (one WebSocket message carries several), so
   per-hop scheduling left a gap whenever a message was late — the stutter you
   hear as the voice "buffering". Here hops are written into a 16 kHz ring and a
   single output node drains it at the device rate with a continuous fractional
   read position, so hop boundaries cannot click either. */
const PLAY_TARGET_S = 0.06; // buffer we aim to keep, also the prefill
const PLAY_MAX_S = 0.3;     // emergency cap on monitor latency
const PLAY_TRIM = 0.01;     // ±1% read-rate trim: inaudible, absorbs drift

function startPlayback(ctx) {
  const ring = new Float32Array(LIVE_SR * 3);
  const state = { ring, write: 0, read: 0, primed: false, underruns: 0, drops: 0 };
  const base = LIVE_SR / ctx.sampleRate;
  const target = PLAY_TARGET_S * LIVE_SR;
  const node = ctx.createScriptProcessor(1024, 1, 1);
  node.onaudioprocess = (ev) => {
    const out = ev.outputBuffer.getChannelData(0);
    let avail = state.write - state.read;
    if (avail > PLAY_MAX_S * LIVE_SR) {
      // Should not happen once the trim below is doing its job; dropping audio
      // clicks, so it is the emergency exit, not the mechanism.
      state.read = state.write - target;
      state.drops++;
      avail = target;
    }
    if (!state.primed) {
      if (avail < target) {
        out.fill(0);
        return;
      }
      state.primed = true;
    }
    // The capture and playback AudioContexts have independent clocks, so the
    // buffer slowly fills or empties. Nudging the read rate by up to 1% holds
    // it at target; a hard resync every few minutes would be audible, this is
    // not (1% is ~0.17 semitones, and only while correcting).
    const err = (avail - target) / target;
    const step = base * (1 + Math.max(-PLAY_TRIM, Math.min(PLAY_TRIM, err)));
    for (let i = 0; i < out.length; i++) {
      if (state.write - state.read < 2) {
        // Underrun: re-prime instead of limping along, or we stutter every hop.
        out.fill(0, i);
        state.primed = false;
        state.underruns++;
        return;
      }
      const i0 = Math.floor(state.read);
      const f = state.read - i0;
      const a = ring[i0 % ring.length];
      const b = ring[(i0 + 1) % ring.length];
      out[i] = a + (b - a) * f;
      state.read += step;
    }
  };
  node.connect(ctx.destination);
  state.node = node;
  return state;
}

function playHop(pcm) {
  const st = micState.play;
  if (!st || !pcm.length) return;
  const ring = st.ring;
  for (let i = 0; i < pcm.length; i++) ring[(st.write + i) % ring.length] = pcm[i];
  st.write += pcm.length;
  if (st.write - st.read > ring.length) st.read = st.write - PLAY_TARGET_S * LIVE_SR;
}

function encodeWav(floatChunks, sr) {
  let n = 0;
  for (const c of floatChunks) n += c.length;
  const pcm = new Int16Array(n);
  let o = 0;
  for (const c of floatChunks) {
    for (let i = 0; i < c.length; i++) {
      const s = Math.max(-1, Math.min(1, c[i]));
      pcm[o++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
  }
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const w = (off, str) => {
    for (let i = 0; i < str.length; i++) v.setUint8(off + i, str.charCodeAt(i));
  };
  w(0, "RIFF");
  v.setUint32(4, 36 + pcm.length * 2, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, pcm.length * 2, true);
  new Int16Array(buf, 44).set(pcm);
  return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
}

function publishYourTake() {
  if (!micState.recIn.length || !micState.recOut.length) return;
  $("yourTake").classList.remove("hidden");
  $("audBefore").src = encodeWav(micState.recIn, LIVE_SR);
  $("audAfter").src = encodeWav(micState.recOut, LIVE_SR);
  $("liveStatus").textContent = "Stopped. Play YOUR recording: voice+disturbance vs CLEANED voice.";
}

function setListen(mode) {
  micState.listen = mode; // "clean" | "raw" | "off"
  $("btnHearClean").classList.toggle("on", mode === "clean");
  $("btnHearRaw").classList.toggle("on", mode === "raw");
  $("btnHearOff").classList.toggle("on", mode === "off");
}

$("btnHearClean").addEventListener("click", () => setListen("clean"));
$("btnHearRaw").addEventListener("click", () => setListen("raw"));
$("btnHearOff").addEventListener("click", () => setListen("off"));

async function startMic() {
  $("liveStatus").textContent = "Allow the microphone…";
  $("yourTake").classList.add("hidden");
  micState.recIn = [];
  micState.recOut = [];
  micState.liveLogs = [];
  micState.pendingHop = null;
  setListen("clean");
  // echoCancellation ON, the other two OFF. Without it the monitor output
  // leaves the speakers, re-enters the mic and you hear your own room looping
  // back. AEC only removes *our own playback* from the capture; the room noise
  // we are here to clean is left alone, which noiseSuppression would not be.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false },
    video: false,
  });
  micState.stream = stream;
  micState.playCtx = new (window.AudioContext || window.webkitAudioContext)();
  await micState.playCtx.resume();
  micState.play = startPlayback(micState.playCtx);
  micState.capCtx = new (window.AudioContext || window.webkitAudioContext)();
  micState.capRs = new Resampler(micState.capCtx.sampleRate, LIVE_SR);
  micState.srcNode = micState.capCtx.createMediaStreamSource(stream);
  // 2048 @ 48 kHz ≈ 43 ms of capture buffering. 4096 doubled that, and the
  // round trip is what makes your own voice come back sounding like an echo.
  const node = micState.capCtx.createScriptProcessor(2048, 1, 1);
  const useMl = $("useTinyMl") && $("useTinyMl").checked;

  if (useMl) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/live`);
    ws.binaryType = "arraybuffer";
    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = () => reject(new Error("WebSocket failed — start python serve.py for TinyML."));
    });
    micState.ws = ws;
    node.onaudioprocess = (ev) => {
      if (!micState.running || !micState.ws || micState.ws.readyState !== 1) return;
      const pcm = micState.capRs.process(ev.inputBuffer.getChannelData(0));
      if (pcm.length) micState.ws.send(new Float32Array(pcm).buffer);
    };
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "hops") msg.hops.forEach(applyLiveHop);
      if (msg.type === "overlay" || msg.type === "ready") {
        if (msg.kind) setOverlayLabel(msg.kind, msg.last_impulse ? ` · impulse ${msg.last_impulse}` : "");
      }
      if (msg.type === "error") $("liveStatus").textContent = msg.message;
    };
  } else {
    if (typeof AegisLive !== "function") throw new Error("Live engine failed to load.");
    micState.live = new AegisLive();
    micState.live.setOverlay(overlayKind, Number($("distGain").value) / 100);
    node.onaudioprocess = (ev) => {
      if (!micState.running || !micState.live) return;
      const pcm = micState.capRs.process(ev.inputBuffer.getChannelData(0));
      if (pcm.length) micState.live.process(pcm).forEach(applyLiveHop);
    };
  }

  micState.running = true;
  const mute = micState.capCtx.createGain();
  mute.gain.value = 0;
  micState.srcNode.connect(node);
  node.connect(mute);
  mute.connect(micState.capCtx.destination);
  micState.procNode = node;
  sendLiveConfig();
  setOverlayLabel(overlayKind);
  $("btnMic").disabled = true;
  $("btnStopMic").disabled = false;
  $("liveStatus").textContent = useMl
    ? "LIVE TinyML · ONNX Runtime on this PC/Pi. Speak, then pick a disturbance."
    : "LIVE in-browser DSP.";
}

function stopMic() {
  micState.running = false;
  try {
    micState.ws && micState.ws.close();
  } catch (e) {}
  try {
    micState.procNode && micState.procNode.disconnect();
    micState.srcNode && micState.srcNode.disconnect();
    micState.play && micState.play.node.disconnect();
    micState.capCtx && micState.capCtx.close();
    micState.playCtx && micState.playCtx.close();
  } catch (e) {}
  micState.play = null;
  if (micState.stream) micState.stream.getTracks().forEach((t) => t.stop());
  micState.stream = null;
  micState.live = null;
  micState.ws = null;
  $("btnMic").disabled = false;
  $("btnStopMic").disabled = true;
  publishYourTake();
}

$("btnMic").addEventListener("click", () => {
  startMic().catch((err) => {
    $("liveStatus").textContent = `Mic error: ${err.message}`;
  });
});
$("btnStopMic").addEventListener("click", stopMic);

async function loadMlStatus() {
  try {
    const st = await (await fetch("/api/ml")).json();
    const on = !!st.available;
    $("mlFlag").textContent = on ? (st.quantized ? "REAL INT8 ONNX" : "REAL FP32 ONNX") : "ONNX missing — train first";
    $("mlFlag").className = on ? "flag real" : "flag no";
    $("flagMl").textContent = on ? "TinyML ONNX" : "TinyML off";
    $("mlParams").textContent = st.params != null ? `${st.params}` : "—";
    $("mlSize").textContent = st.size_kb != null ? `${st.size_kb} KB` : "—";
    $("mlBackend").textContent = st.backend || "none";
    $("mlQuant").textContent = st.quantized ? "INT8" : on ? "FP32" : "—";
    $("mlHop").textContent = st.infer_ms_per_hop != null ? `${Number(st.infer_ms_per_hop).toFixed(2)} ms` : "on live";
    $("mlVad").textContent = st.vad_accuracy != null ? `${(st.vad_accuracy * 100).toFixed(0)}%` : "—";
    $("mlTgt").textContent = "not claimed";
    if ($("useTinyMl")) $("useTinyMl").checked = on;
    if (st.hosted_static) staticHostMode(st);
  } catch (err) {
    staticHostMode(null);
  }
}

/* Vercel serves the frontend only: no Python, so no ONNX Runtime and no
   WebSocket. The in-browser engine runs the same DSP chain, so say that
   plainly instead of letting the mic button fail on a dead WebSocket. */
function staticHostMode(meta) {
  $("mlFlag").textContent = meta ? "model exported — runs on a Pi, not on this host" : "no backend on this host";
  $("mlFlag").className = "flag sim";
  $("flagMl").textContent = "in-browser DSP";
  $("mlBackend").textContent = "in-browser DSP (no ONNX here)";
  $("mlHop").textContent = "n/a on static host";
  const toggle = $("useTinyMl");
  if (toggle) {
    toggle.checked = false;
    toggle.disabled = true;
    const label = toggle.closest("label");
    if (label) {
      label.lastChild.textContent =
        " Static hosting has no Python backend — the mic runs the in-browser engine. Run python serve.py for the ONNX path.";
    }
  }
}
loadMlStatus();
