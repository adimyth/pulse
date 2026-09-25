const state = { ws: null, stream: null, context: null, source: null, capture: null, analyser: null, mute: null, samples: [], sourceRate: 0, previousWords: [], raf: null, accent: "#7aa6ff" };
const transcript = document.querySelector("#transcript");
const transcriptHistory = document.querySelector("#transcript-history");
const transcriptScroll = document.querySelector("#transcript-scroll");
const meters = [...document.querySelectorAll(".meter")];
const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const connection = document.querySelector("#connection");
const canvas = document.querySelector("#waveform");
const drawing = canvas.getContext("2d");
const tones = { frustration: "#ef5c5f", positive: "#e5c23e", surprise: "#bc8def", uncertainty: "#7aa6ff", low_mood: "#758290", neutral: "#b5bbc5" };

function setText(id, value) { document.querySelector(id).textContent = value; }
function displayMs(value) { return `${Math.round(value)} ms`; }

function renderTranscript(text, accent) {
  const words = text.split(/\s+/).filter(Boolean);
  let common = 0;
  while (common < words.length && common < state.previousWords.length && words[common].toLowerCase() === state.previousWords[common].toLowerCase()) common += 1;
  transcript.replaceChildren(...words.flatMap((word, index) => {
    const part = document.createElement("span");
    part.className = index < common ? "stable" : "fresh";
    part.style.setProperty("--accent", accent);
    part.textContent = word;
    return index === words.length - 1 ? [part] : [part, document.createTextNode(" ")];
  }));
  state.previousWords = words;
}

function archiveTranscript(text, accent) {
  const entry = document.createElement("li");
  const label = document.createElement("span");
  const content = document.createElement("p");
  label.textContent = "Final";
  content.textContent = text;
  content.style.setProperty("--accent", accent);
  entry.append(label, content);
  transcriptHistory.append(entry);
  transcript.textContent = "Listening for the next thought…";
  state.previousWords = [];
  requestAnimationFrame(() => { transcriptScroll.scrollTop = transcriptScroll.scrollHeight; });
}

function renderSentiment(result) {
  const dominant = result.dominant || "Listening";
  state.accent = tones[dominant] || "#7aa6ff";
  document.documentElement.style.setProperty("--accent", state.accent);
  document.querySelector("#orb").style.filter = `drop-shadow(0 0 24px ${state.accent}66)`;
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
  setText("#context-state", result.abstained ? "Listening for enough context…" : "Text signals update independently as the transcript changes.");
}

function renderEvent(event) {
  const accent = tones[event.sentiment.dominant] || "#7aa6ff";
  if (event.type === "final") archiveTranscript(event.transcript, accent);
  else renderTranscript(event.transcript, accent);
  renderSentiment(event.sentiment);
  setText("#utterance-state", event.type === "final" ? "Final" : "Live");
  setText("#event-kind", event.type === "final" ? "Utterance finalized" : "Refreshing every 300 ms");
  setText("#latency", `Audio → UI ${displayMs(event.timings_ms.audio_to_ui)}`);
  setText("#stt", `STT ${displayMs(event.timings_ms.stt)}`);
  setText("#classifier", `Text model ${displayMs(event.timings_ms.classifier)}`);
}

function resizeCanvas() {
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * pixelRatio;
  canvas.height = canvas.clientHeight * pixelRatio;
  drawing.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
}

function drawWave() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  drawing.clearRect(0, 0, width, height);
  drawing.fillStyle = "#141414";
  drawing.fillRect(0, 0, width, height);
  const bars = 84;
  const values = new Uint8Array(state.analyser?.fftSize || 256);
  if (state.analyser) state.analyser.getByteTimeDomainData(values);
  const centre = height * .43;
  drawing.fillStyle = "#343434";
  drawing.fillRect(0, centre, width, 1);
  for (let index = 0; index < bars; index += 1) {
    const sourceIndex = Math.floor(index / bars * values.length);
    const amplitude = state.analyser ? Math.abs(values[sourceIndex] - 128) / 128 : 0.035;
    const barHeight = Math.max(4, amplitude * height * .92);
    const x = (index + .5) * width / bars;
    const active = state.stream && index > 6 && index < bars - 7;
    drawing.fillStyle = active ? state.accent : "#444";
    drawing.globalAlpha = active ? Math.max(.38, amplitude * 1.5) : .45;
    drawing.beginPath();
    drawing.roundRect(x, centre - barHeight / 2, Math.max(3, width / bars - 5), barHeight, 5);
    drawing.fill();
  }
  drawing.globalAlpha = 1;
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
  state.capture.port.onmessage = ({ data }) => downsample(data);
  state.source.connect(state.analyser);
  state.source.connect(state.capture);
  state.capture.connect(state.mute);
  state.mute.connect(state.context.destination);
  document.querySelector(".record-dot").classList.add("active");
  setText("#wave-copy", "Listening locally · 16 kHz PCM");
}

function closeMicrophone() {
  state.capture?.disconnect();
  state.source?.disconnect();
  state.mute?.disconnect();
  state.stream?.getTracks().forEach(track => track.stop());
  state.context?.close();
  Object.assign(state, { stream: null, context: null, source: null, capture: null, analyser: null, mute: null, samples: [], sourceRate: 0 });
  document.querySelector(".record-dot").classList.remove("active");
  setText("#wave-copy", "Microphone inactive");
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
      try { await startMicrophone(); stopButton.disabled = false; setText("#connection", "Live on this Mac"); }
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

window.addEventListener("resize", resizeCanvas);
startButton.addEventListener("click", start);
stopButton.addEventListener("click", stop);
resizeCanvas();
drawWave();
