const state = { ws: null, stream: null, context: null, source: null, capture: null, analyser: null, mute: null, samples: [], sourceRate: 0, audioFrames: 0, inputWarningTimer: null, liveText: "", liveSegments: [], history: [], raf: null, accent: "#7aa6ff", signalStrength: .12, sessionWave: [], waveFramePeak: 0, waveFrameSamples: 0 };
const transcript = document.querySelector("#transcript");
const transcriptHistory = document.querySelector("#transcript-history");
const transcriptScroll = document.querySelector("#transcript-scroll");
const stage = document.querySelector(".stage");
const meters = [...document.querySelectorAll(".meter")];
const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const resetButton = document.querySelector("#reset");
const exportButton = document.querySelector("#export");
const connection = document.querySelector("#connection");
const canvas = document.querySelector("#waveform");
const drawing = canvas.getContext("2d");
const pressureBlock = document.querySelector(".pressure");
const tones = { frustration: "#ef5c5f", positive: "#e5c23e", surprise: "#bc8def", uncertainty: "#7aa6ff", low_mood: "#758290", neutral: "#b5bbc5" };
const nonSpeechCaptions = new Set(["blank audio", "silence", "howling wind", "wind", "wind blowing", "crowd cheer", "crowd cheering", "cheering", "applause", "engine revving", "engine reving", "keyboard clicking", "typing", "background noise", "music", "laughter", "non english speech", "speaking in foreign language", "foreign language"]);
const WAVE_SAMPLE_WINDOW = .025;
const MAX_SESSION_WAVE_SAMPLES = 2400;
const SPEECH_WAVE_THRESHOLD = .003;
const WAVE_RENDER_THRESHOLD = .001;

function setText(id, value) { document.querySelector(id).textContent = value; }
function displayMs(value) { return `${Math.round(value)} ms`; }
function displayName(value) { return value ? value.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase()) : "Listening"; }
function isHumanTranscript(text) { return Boolean(text?.trim()) && !nonSpeechCaptions.has(text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim()); }

function sentenceAccent(sentence, supported) {
  return supported ? tones[sentence.sentiment.dominant] || "#b5bbc5" : "#9baecf";
}

function renderLiveTranscript(event) {
  if (event.transcript === state.liveText) return;
  const sentences = (event.sentences?.length ? event.sentences : [{ text: event.transcript, sentiment: event.sentiment }]).filter(sentence => isHumanTranscript(sentence.text));
  let retained = 0;
  while (retained < sentences.length && retained < state.liveSegments.length && sentences[retained].text === state.liveSegments[retained].text && sentences[retained].sentiment.dominant === state.liveSegments[retained].sentiment.dominant) retained += 1;
  if (!state.liveSegments.length) transcript.replaceChildren();
  while (transcript.children.length > retained) transcript.lastElementChild.remove();
  for (const sentence of sentences.slice(retained)) {
    const fragment = document.createElement("span");
    fragment.textContent = sentence.text;
    fragment.style.color = sentenceAccent(sentence, event.sentiment_supported);
    transcript.append(fragment);
  }
  state.liveSegments = sentences;
  state.liveText = event.transcript;
}

function archiveTranscript(event) {
  const sentences = event.sentences?.length ? event.sentences : [{ text: event.transcript, sentiment: event.sentiment }];
  const entry = document.createElement("li");
  const content = document.createElement("p");
  let appended = false;
  for (const sentence of sentences) {
    if (!isHumanTranscript(sentence.text)) continue;
    const color = sentenceAccent(sentence, event.sentiment_supported);
    const fragment = document.createElement("span");
    fragment.textContent = sentence.text;
    fragment.style.color = color;
    content.append(fragment);
    state.history.push({ text: sentence.text, color });
    appended = true;
  }
  if (!appended) return;
  entry.append(content);
  transcriptHistory.append(entry);
  exportButton.disabled = false;
  transcript.replaceChildren();
  transcript.style.color = "#707070";
  state.liveText = "";
  state.liveSegments = [];
  requestAnimationFrame(() => { stage.scrollTop = stage.scrollHeight; });
}

function renderSentiment(result) {
  const dominant = result.dominant || "Listening";
  state.accent = tones[dominant] || "#7aa6ff";
  state.signalStrength = Math.max(.1, ...result.scores.filter(score => score.key !== "neutral").map(score => score.value));
  document.documentElement.style.setProperty("--accent", state.accent);
  setText("#dominant", dominant === "Listening" ? dominant : dominant.replace("_", " "));
  result.scores.forEach((score, index) => {
    const meter = meters[index];
    meter.dataset.active = "true";
    meter.querySelector("strong").textContent = `${score.level} · ${Math.round(score.value * 100)}%`;
    meter.querySelector("b").style.width = `${Math.round(score.value * 100)}%`;
  });
  const pressure = Math.round(result.action_pressure * 100);
  setText("#pressure", pressure ? `${pressure}% urgency cue` : "No urgency cue");
  pressureBlock.dataset.active = String(Boolean(pressure));
  document.querySelector("#pressure-bar").style.width = `${pressure}%`;
}

function renderEvent(event) {
  if (!isHumanTranscript(event.transcript)) return;
  if (event.type === "final") {
    archiveTranscript(event);
  }
  else renderLiveTranscript(event);
  renderSentiment(event.sentiment);
  setText("#stt", `STT ${displayMs(event.timings_ms.stt)}`);
  setText("#classifier", `Text model ${displayMs(event.timings_ms.classifier)}`);
}

function resizeOneCanvas(target, context) {
  const pixelRatio = window.devicePixelRatio || 1;
  target.width = target.clientWidth * pixelRatio;
  target.height = target.clientHeight * pixelRatio;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
}

function resizeCanvas() {
  resizeOneCanvas(canvas, drawing);
}

function compactSessionWave(wave, limit) {
  if (wave.length <= limit) return;
  const compacted = [];
  for (let index = 0; index < wave.length; index += 2) {
    const left = wave[index];
    const right = wave[index + 1] || left;
    compacted.push(left.amplitude >= right.amplitude ? left : right);
  }
  wave.splice(0, wave.length, ...compacted);
}

function appendWaveSample(amplitude) {
  state.sessionWave.push({ amplitude: amplitude >= SPEECH_WAVE_THRESHOLD ? amplitude : 0, color: state.accent });
  compactSessionWave(state.sessionWave, MAX_SESSION_WAVE_SAMPLES);
}

function recordWaveform(input) {
  if (!state.sourceRate) return;
  let sum = 0;
  for (const sample of input) sum += sample * sample;
  state.waveFramePeak = Math.max(state.waveFramePeak, Math.sqrt(sum / input.length));
  state.waveFrameSamples += input.length;
  if (state.waveFrameSamples < state.sourceRate * WAVE_SAMPLE_WINDOW) return;
  appendWaveSample(state.waveFramePeak);
  state.waveFramePeak = 0;
  state.waveFrameSamples = 0;
}

function drawEnvelope(context, target, values, amplitudeOf, colorOf, active) {
  const width = target.clientWidth;
  const height = target.clientHeight;
  const centre = height / 2;
  context.clearRect(0, 0, width, height);
  if (!values.length) return;
  const bars = Math.max(1, Math.min(Math.floor(width / 5), values.length));
  for (let index = 0; index < bars; index += 1) {
    const start = Math.floor(index * values.length / bars);
    const end = Math.max(start + 1, Math.floor((index + 1) * values.length / bars));
    let peak = values[start];
    let amplitude = amplitudeOf(peak);
    for (let sample = start + 1; sample < end; sample += 1) {
      const next = amplitudeOf(values[sample]);
      if (next > amplitude) { peak = values[sample]; amplitude = next; }
    }
    if (amplitude < WAVE_RENDER_THRESHOLD) continue;
    const barHeight = Math.max(3, Math.min(height * .92, amplitude * height * 9.5));
    const x = index * width / bars + 1;
    context.fillStyle = colorOf(peak);
    context.globalAlpha = active ? Math.max(.48, Math.min(.98, amplitude * 8)) : Math.max(.42, Math.min(.88, amplitude * 7));
    context.beginPath();
    context.roundRect(x, centre - barHeight / 2, Math.max(2, width / bars - 3), barHeight, 4);
    context.fill();
  }
  context.globalAlpha = 1;
}

function drawWave() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  drawing.clearRect(0, 0, width, height);
  drawing.fillStyle = "#141414";
  drawing.fillRect(0, 0, width, height);
  drawEnvelope(drawing, canvas, state.sessionWave, sample => sample.amplitude, sample => sample.color, Boolean(state.stream));
  state.raf = requestAnimationFrame(drawWave);
}

function downsample(input) {
  const ratio = state.sourceRate / 16000;
  for (let position = 0; position < input.length; position += ratio) {
    const end = Math.min(input.length, Math.ceil(position + ratio));
    let total = 0;
    for (let index = Math.floor(position); index < end; index += 1) total += input[index];
    state.samples.push(Math.max(-1, Math.min(1, total / Math.max(1, end - Math.floor(position)))));
  }
  while (state.samples.length >= 320 && state.ws?.readyState === WebSocket.OPEN) {
    const frames = state.samples.splice(0, 320);
    const pcm = new Int16Array(frames.length);
    for (let index = 0; index < pcm.length; index += 1) pcm[index] = Math.round(frames[index] * 32767);
    state.ws.send(pcm.buffer);
  }
}

async function startMicrophone() {
  state.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  state.context = new AudioContext();
  await state.context.resume();
  if (state.context.state !== "running") throw new Error("the browser did not start the microphone audio context");
  state.sourceRate = state.context.sampleRate;
  await state.context.audioWorklet.addModule("/static/capture-worklet.js");
  state.source = state.context.createMediaStreamSource(state.stream);
  state.analyser = state.context.createAnalyser();
  state.analyser.fftSize = 256;
  state.capture = new AudioWorkletNode(state.context, "pulse-capture");
  state.mute = state.context.createGain();
  state.mute.gain.value = 0;
  state.audioFrames = 0;
  state.capture.port.onmessage = ({ data }) => {
    state.audioFrames += 1;
    if (state.audioFrames === 1) setText("#wave-copy", "Audio input received · transcribing locally");
    recordWaveform(data);
    downsample(data);
  };
  state.source.connect(state.analyser);
  state.source.connect(state.capture);
  state.capture.connect(state.mute);
  state.mute.connect(state.context.destination);
  document.querySelector(".record-dot").classList.add("active");
  setText("#wave-copy", "Session waveform · listening locally");
  clearTimeout(state.inputWarningTimer);
  state.inputWarningTimer = setTimeout(() => {
    if (state.stream && state.audioFrames === 0) setText("#wave-copy", "No microphone signal · check browser input");
  }, 2000);
}

function closeMicrophone() {
  clearTimeout(state.inputWarningTimer);
  state.capture?.disconnect();
  state.source?.disconnect();
  state.mute?.disconnect();
  state.stream?.getTracks().forEach(track => track.stop());
  state.context?.close();
  Object.assign(state, { stream: null, context: null, source: null, capture: null, analyser: null, mute: null, samples: [], sourceRate: 0, audioFrames: 0, inputWarningTimer: null });
  document.querySelector(".record-dot").classList.remove("active");
  setText("#wave-copy", "Session waveform · microphone inactive");
}

function resetDisplay() {
  transcriptHistory.replaceChildren();
  state.history = [];
  state.liveText = "";
  state.liveSegments = [];
  state.accent = "#7aa6ff";
  state.signalStrength = .12;
  state.sessionWave = [];
  state.waveFramePeak = 0;
  state.waveFrameSamples = 0;
  transcript.textContent = "Press start and speak naturally.";
  transcript.style.color = "#707070";
  meters.forEach(meter => { meter.dataset.active = "false"; meter.querySelector("strong").textContent = "—"; meter.querySelector("b").style.width = "0%"; });
  setText("#dominant", "Listening");
  setText("#pressure", "Waiting for speech");
  pressureBlock.dataset.active = "false";
  document.querySelector("#pressure-bar").style.width = "0%";
  setText("#stt", "STT —");
  setText("#classifier", "Text model —");
  exportButton.disabled = true;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[character]);
}

function exportTranscript() {
  if (!state.history.length) return;
  const rows = state.history.map(entry => `<p class="utterance" style="color:${entry.color}">${escapeHtml(entry.text)}</p>`).join("\n");
  const documentText = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Pulse transcript</title><style>body{margin:0;padding:44px;max-width:860px;background:#111;color:#f4f3ef;font-family:Inter,system-ui,sans-serif}h1{font-size:1.4rem;margin:0 0 26px}.utterance{margin:0 0 14px;font-size:1.2rem;line-height:1.5}</style></head><body><h1>Pulse · Session transcript</h1>${rows}</body></html>`;
  const download = document.createElement("a");
  download.href = URL.createObjectURL(new Blob([documentText], { type: "text/html" }));
  download.download = `pulse-transcript-${new Date().toISOString().replaceAll(":", "-")}.html`;
  document.body.append(download);
  download.click();
  download.remove();
  setTimeout(() => URL.revokeObjectURL(download.href), 0);
}

async function start() {
  startButton.disabled = true;
  setText("#connection", "Opening local session…");
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${protocol}://${location.host}/ws/live`);
  state.ws.onopen = () => state.ws.send(JSON.stringify({ type: "start" }));
  state.ws.onerror = () => { setText("#connection", "Could not reach local dashboard server"); startButton.disabled = false; };
  state.ws.onmessage = async ({ data }) => {
    const message = JSON.parse(data);
    if (message.type === "ready") { setText("#connection", "Local engine ready"); return; }
    if (message.type === "started") {
      try { await startMicrophone(); stopButton.disabled = false; setText("#connection", "Listening"); }
      catch (error) { setText("#connection", `Microphone permission failed: ${error.message}`); state.ws.send(JSON.stringify({ type: "discard" })); startButton.disabled = false; }
      return;
    }
    if (message.type === "stopped") {
      closeMicrophone();
      state.ws?.close();
      state.ws = null;
      stopButton.disabled = true;
      startButton.disabled = false;
      setText("#connection", message.discarded ? "Stopped · audio discarded" : "Stopped · final transcript retained");
      return;
    }
    if (message.type === "error") { setText("#connection", `Local error: ${message.message}`); setText("#wave-copy", "Recovering local speech recognition…"); return; }
    if (message.type === "partial" || message.type === "final") renderEvent(message);
  };
  state.ws.onclose = () => {
    if (state.stream) closeMicrophone();
    stopButton.disabled = true;
    startButton.disabled = false;
  };
}

function stop(discard = false) {
  stopButton.disabled = true;
  closeMicrophone();
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: discard ? "discard" : "stop" }));
    return;
  }
  state.ws?.close();
  state.ws = null;
  startButton.disabled = false;
}

function resetSession() {
  stop(true);
  resetDisplay();
}

window.addEventListener("resize", resizeCanvas);
startButton.addEventListener("click", start);
stopButton.addEventListener("click", stop);
resetButton.addEventListener("click", resetSession);
exportButton.addEventListener("click", exportTranscript);
resizeCanvas();
drawWave();
