const state = { ws: null, stream: null, context: null, source: null, capture: null, analyser: null, mute: null, samples: [], sourceRate: 0, liveText: "", history: [], raf: null, accent: "#7aa6ff", signalStrength: .12, currentWave: [], sessionWave: [], waveFramePeak: 0, waveFrameSamples: 0, currentWaveOpen: false };
const transcript = document.querySelector("#transcript");
const transcriptHistory = document.querySelector("#transcript-history");
const transcriptScroll = document.querySelector("#transcript-scroll");
const meters = [...document.querySelectorAll(".meter")];
const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const resetButton = document.querySelector("#reset");
const exportButton = document.querySelector("#export");
const connection = document.querySelector("#connection");
const canvas = document.querySelector("#waveform");
const drawing = canvas.getContext("2d");
const signalCanvas = document.querySelector("#signal-wave");
const signalDrawing = signalCanvas.getContext("2d");
const tones = { frustration: "#ef5c5f", positive: "#e5c23e", surprise: "#bc8def", uncertainty: "#7aa6ff", low_mood: "#758290", neutral: "#b5bbc5" };
const nonSpeechCaptions = new Set(["blank audio", "silence", "howling wind", "wind", "wind blowing", "crowd cheer", "crowd cheering", "cheering", "applause", "engine revving", "engine reving", "keyboard clicking", "typing", "background noise", "music", "laughter", "non english speech", "speaking in foreign language", "foreign language"]);
const WAVE_SAMPLE_WINDOW = .025;
const MAX_SESSION_WAVE_SAMPLES = 2400;
const MAX_CURRENT_WAVE_SAMPLES = 420;
const SPEECH_WAVE_THRESHOLD = .012;

function setText(id, value) { document.querySelector(id).textContent = value; }
function displayMs(value) { return `${Math.round(value)} ms`; }
function displayName(value) { return value ? value.replaceAll("_", " ").replace(/\b\w/g, character => character.toUpperCase()) : "Listening"; }
function isHumanTranscript(text) { return Boolean(text?.trim()) && !nonSpeechCaptions.has(text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim()); }

function renderTranscript(text, accent) {
  if (text !== state.liveText) transcript.textContent = text;
  transcript.style.color = accent;
  state.liveText = text;
}

function sentenceAccent(sentence, supported) {
  return supported ? tones[sentence.sentiment.dominant] || "#b5bbc5" : "#9baecf";
}

function archiveTranscript(event) {
  const sentences = event.sentences?.length ? event.sentences : [{ text: event.transcript, sentiment: event.sentiment }];
  for (const sentence of sentences) {
    if (!isHumanTranscript(sentence.text)) continue;
    const entry = document.createElement("li");
    const content = document.createElement("p");
    const color = sentenceAccent(sentence, event.sentiment_supported);
    content.textContent = sentence.text;
    content.style.color = color;
    entry.append(content);
    transcriptHistory.append(entry);
    state.history.push({ text: sentence.text, color });
  }
  exportButton.disabled = false;
  transcript.textContent = "Listening for the next thought…";
  transcript.style.color = "#707070";
  state.liveText = "";
  requestAnimationFrame(() => { transcriptScroll.scrollTop = transcriptScroll.scrollHeight; });
}

function renderSentiment(result) {
  const dominant = result.dominant || "Listening";
  state.accent = tones[dominant] || "#7aa6ff";
  state.signalStrength = Math.max(.1, ...result.scores.filter(score => score.key !== "neutral").map(score => score.value));
  document.documentElement.style.setProperty("--accent", state.accent);
  setText("#dominant", dominant === "Listening" ? dominant : dominant.replace("_", " "));
  result.scores.forEach((score, index) => {
    const meter = meters[index];
    meter.querySelector("strong").textContent = `${score.level} · ${Math.round(score.value * 100)}%`;
    meter.querySelector("b").style.width = `${Math.round(score.value * 100)}%`;
  });
  const pressure = Math.round(result.action_pressure * 100);
  setText("#pressure", pressure ? `${pressure}% urgency cue` : "No urgency cue");
  document.querySelector("#pressure-bar").style.width = `${pressure}%`;
  setText("#pressure-detail", pressure ? "Urgency wording is present in the current transcript." : "No urgency wording is present in the current transcript.");
}

function renderLanguage(event) {
  const language = event.language ? displayName(event.language) : "Detecting language";
  setText("#language", `Language ${language}`);
}

function renderEvent(event) {
  if (!isHumanTranscript(event.transcript)) return;
  const accent = event.sentiment_supported ? tones[event.sentiment.dominant] || "#7aa6ff" : "#9baecf";
  if (event.type === "final") {
    archiveTranscript(event);
    finishCurrentWave();
  }
  else renderTranscript(event.transcript, accent);
  renderSentiment(event.sentiment);
  renderLanguage(event);
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
  resizeOneCanvas(signalCanvas, signalDrawing);
}

function compactWave(wave, limit) {
  if (wave.length <= limit) return;
  const compacted = [];
  for (let index = 0; index < wave.length; index += 2) compacted.push(Math.max(wave[index], wave[index + 1] || 0));
  wave.splice(0, wave.length, ...compacted);
}

function appendWaveSample(amplitude) {
  state.sessionWave.push(amplitude);
  compactWave(state.sessionWave, MAX_SESSION_WAVE_SAMPLES);
  if (!state.currentWaveOpen && amplitude >= SPEECH_WAVE_THRESHOLD) {
    state.currentWave = [];
    state.currentWaveOpen = true;
    setText("#current-wave-state", "Speaking");
  }
  if (state.currentWaveOpen) {
    state.currentWave.push(amplitude);
    compactWave(state.currentWave, MAX_CURRENT_WAVE_SAMPLES);
  }
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

function finishCurrentWave() {
  if (!state.currentWaveOpen && !state.currentWave.length) return;
  state.currentWaveOpen = false;
  state.currentWave = [];
  setText("#current-wave-state", "Awaiting speech");
}

function drawEnvelope(context, target, values, color, active) {
  const width = target.clientWidth;
  const height = target.clientHeight;
  context.clearRect(0, 0, width, height);
  const centre = height / 2;
  context.fillStyle = "#2e2e2e";
  context.fillRect(0, centre, width, 1);
  if (!values.length) return;
  const bars = Math.max(1, Math.min(Math.floor(width / 5), values.length));
  for (let index = 0; index < bars; index += 1) {
    const start = Math.floor(index * values.length / bars);
    const end = Math.max(start + 1, Math.floor((index + 1) * values.length / bars));
    let amplitude = 0;
    for (let sample = start; sample < end; sample += 1) amplitude = Math.max(amplitude, values[sample]);
    const barHeight = Math.max(2, Math.min(height * .92, amplitude * height * 8.5));
    const x = index * width / bars + 1;
    context.fillStyle = color;
    context.globalAlpha = active ? Math.max(.4, Math.min(.96, amplitude * 7)) : Math.max(.2, Math.min(.58, amplitude * 4));
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
  drawEnvelope(drawing, canvas, state.sessionWave, "#98a0ac", Boolean(state.stream));
  drawEnvelope(signalDrawing, signalCanvas, state.currentWave, state.accent, state.currentWaveOpen);
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
  state.sourceRate = state.context.sampleRate;
  await state.context.audioWorklet.addModule("/static/capture-worklet.js");
  state.source = state.context.createMediaStreamSource(state.stream);
  state.analyser = state.context.createAnalyser();
  state.analyser.fftSize = 256;
  state.capture = new AudioWorkletNode(state.context, "pulse-capture");
  state.mute = state.context.createGain();
  state.mute.gain.value = 0;
  state.capture.port.onmessage = ({ data }) => { recordWaveform(data); downsample(data); };
  state.source.connect(state.analyser);
  state.source.connect(state.capture);
  state.capture.connect(state.mute);
  state.mute.connect(state.context.destination);
  document.querySelector(".record-dot").classList.add("active");
  setText("#wave-copy", "Session waveform · listening locally");
}

function closeMicrophone() {
  state.capture?.disconnect();
  state.source?.disconnect();
  state.mute?.disconnect();
  state.stream?.getTracks().forEach(track => track.stop());
  state.context?.close();
  Object.assign(state, { stream: null, context: null, source: null, capture: null, analyser: null, mute: null, samples: [], sourceRate: 0 });
  document.querySelector(".record-dot").classList.remove("active");
  if (state.currentWaveOpen) finishCurrentWave();
  setText("#wave-copy", "Session waveform · microphone inactive");
}

function resetDisplay() {
  transcriptHistory.replaceChildren();
  state.history = [];
  state.liveText = "";
  state.accent = "#7aa6ff";
  state.signalStrength = .12;
  state.currentWave = [];
  state.sessionWave = [];
  state.waveFramePeak = 0;
  state.waveFrameSamples = 0;
  state.currentWaveOpen = false;
  transcript.textContent = "Press start and speak naturally.";
  transcript.style.color = "#707070";
  meters.forEach(meter => { meter.querySelector("strong").textContent = "—"; meter.querySelector("b").style.width = "0%"; });
  setText("#dominant", "Listening");
  setText("#pressure", "Waiting for speech");
  setText("#pressure-detail", "Shown separately from sentiment when the transcript contains urgency wording.");
  document.querySelector("#pressure-bar").style.width = "0%";
  setText("#language", "Language —");
  setText("#stt", "STT —");
  setText("#classifier", "Text model —");
  setText("#current-wave-state", "Awaiting speech");
  exportButton.disabled = true;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, character => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" })[character]);
}

function exportTranscript() {
  if (!state.history.length) return;
  const rows = state.history.map(entry => `<p class="utterance" style="color:${entry.color}">${escapeHtml(entry.text)}</p>`).join("\n");
  const documentText = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Pulse Local transcript</title><style>body{margin:0;padding:44px;max-width:860px;background:#111;color:#f4f3ef;font-family:Inter,system-ui,sans-serif}h1{font-size:1.4rem;margin:0 0 26px}.utterance{margin:0 0 14px;font-size:1.2rem;line-height:1.5}</style></head><body><h1>Pulse Local · Session transcript</h1>${rows}</body></html>`;
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
      catch (error) { setText("#connection", `Microphone permission failed: ${error.message}`); state.ws.send(JSON.stringify({ type: "stop" })); startButton.disabled = false; }
      return;
    }
    if (message.type === "stopped") { setText("#connection", "Stopped · audio discarded"); return; }
    if (message.type === "error") { setText("#connection", `Local error: ${message.message}`); return; }
    if (message.type === "partial" || message.type === "final") renderEvent(message);
  };
}

function stop() {
  stopButton.disabled = true;
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: "stop" }));
  closeMicrophone();
  state.ws?.close();
  state.ws = null;
  startButton.disabled = false;
}

function resetSession() {
  stop();
  resetDisplay();
}

window.addEventListener("resize", resizeCanvas);
startButton.addEventListener("click", start);
stopButton.addEventListener("click", stop);
resetButton.addEventListener("click", resetSession);
exportButton.addEventListener("click", exportTranscript);
resizeCanvas();
drawWave();
