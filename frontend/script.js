/**
 * TheONE AI — Voice-First AI Companion
 * Frontend Client Logic
 *
 * Implements:
 * 1. Minimal, voice-first ChatGPT Advanced Voice Mode style interface
 * 2. 5-state Voice Orb visualizer (Idle, Listening, Thinking, Speaking, Error)
 * 3. Real-time audio reactivity via Web Audio API AnalyserNode
 * 4. Dual personality modes: Always Funny & Roast Companion
 * 5. User-selectable Gnani Timbre TTS voices (Yashvi & Zayan) independent per mode
 * 6. Gnani Prisma STT (16 kHz Linear PCM downsampling & dispatch)
 * 7. Compact live floating transcript card & slide-over history drawer
 * 8. Strict 1-sentence response limit preserved across all interactions
 */

// =============================================================================
// Application State
// =============================================================================
const state = {
  activeMode: 'unified_comedy',
  activeVoice: 'Yashvi', // 'Yashvi' | 'Zayan' (independent of humour style)
  conversationId: null,
  orbState: 'idle', // 'idle' | 'listening' | 'thinking' | 'speaking' | 'error'
  isRecording: false,
  audioContext: null,
  mediaStream: null,
  scriptProcessor: null,
  micAnalyser: null,
  playbackAnalyser: null,
  activeAudioSource: null,
  activeAudioBuffer: null,
  lastAudioBlobUrl: null,
  pcmChunks: [],
  timerInterval: null,
  recordingStartTime: null,
  transcriptCount: 0,
  currentAmplitude: 0.0,
  apiBaseUrl: window.location.protocol.startsWith('http')
    ? window.location.origin
    : 'http://127.0.0.1:8000',
};

// Load persisted voice preferences
try {
  const savedVoice = localStorage.getItem('theone_voice') || localStorage.getItem('theone_selected_voice');
  if (savedVoice && (savedVoice === 'Yashvi' || savedVoice === 'Zayan')) {
    state.activeVoice = savedVoice;
  }
} catch (_) {}

// =============================================================================
// DOM References
// =============================================================================
const voiceStage = document.querySelector('.voice-stage');
const voiceOrb = document.getElementById('voice-orb');
const orbCanvas = document.getElementById('orb-canvas');
const orbAmbientGlow = document.getElementById('orb-ambient-glow');
const orbStatusText = document.getElementById('orb-status-text');
const recordingState = document.getElementById('recording-state');
const recordingTimer = document.getElementById('recording-timer');
const statusWaveBars = document.getElementById('status-wave-bars');

const brandAvatar = document.getElementById('brand-avatar');
const backendStatus = document.getElementById('backend-status');
const backendStatusText = document.getElementById('backend-status-text');
const ttsVoiceBadge = document.getElementById('tts-voice-badge');
const ttsVoiceBadgeText = document.getElementById('tts-voice-badge-text');

const btnNavVoiceYashvi = document.getElementById('btn-nav-voice-yashvi');
const btnNavVoiceZayan = document.getElementById('btn-nav-voice-zayan');

const btnMic = document.getElementById('btn-mic');
const chatForm = document.getElementById('chat-form');
const textInput = document.getElementById('text-input');

// Floating Transcript Card
const floatingTranscript = document.getElementById('floating-transcript');
const floatModePill = document.getElementById('float-mode-pill');
const floatUserText = document.getElementById('float-user-text');
const floatBotText = document.getElementById('float-bot-text');
const btnFloatReplay = document.getElementById('btn-float-replay');
const btnOpenHistoryLink = document.getElementById('btn-open-history-link');

// Slide Drawers & Backdrops
const btnToggleTranscript = document.getElementById('btn-toggle-transcript');
const transcriptDrawer = document.getElementById('transcript-drawer');
const transcriptBackdrop = document.getElementById('transcript-backdrop');
const btnCloseTranscript = document.getElementById('btn-close-transcript');
const transcriptCount = document.getElementById('transcript-count');
const chatStream = document.getElementById('chat-stream');
const btnClearChat = document.getElementById('btn-clear-chat');

const btnOpenSettings = document.getElementById('btn-open-settings');
const settingsDrawer = document.getElementById('settings-drawer');
const settingsBackdrop = document.getElementById('settings-backdrop');
const btnCloseSettings = document.getElementById('btn-close-settings');
const voiceModeLabel = document.getElementById('voice-mode-label');
const btnVoiceYashvi = document.getElementById('btn-voice-yashvi');
const btnVoiceZayan = document.getElementById('btn-voice-zayan');
const voiceOptionCards = document.querySelectorAll('.voice-option-card');
const languageSelect = document.getElementById('language-select');

// Theme Toggle
const btnToggleTheme = document.getElementById('btn-toggle-theme');

// Notification Toasts
const errorBox = document.getElementById('error-box');
const errorMessage = document.getElementById('error-message');
const btnDismissError = document.getElementById('btn-dismiss-error');
const sttStatusBanner = document.getElementById('stt-status-banner');
const sttStatusMessage = document.getElementById('stt-status-message');
const aiLoading = document.getElementById('ai-loading');
const loadingText = document.getElementById('loading-text');

// =============================================================================
// Helper: Shared Web Audio Context
// =============================================================================
function getAudioContext() {
  if (!state.audioContext || state.audioContext.state === 'closed') {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    state.audioContext = new AudioContextClass();
  }
  if (state.audioContext.state === 'suspended') {
    state.audioContext.resume().catch(() => {});
  }
  return state.audioContext;
}

// =============================================================================
// State Machine: 5 Voice Orb States
// =============================================================================
/**
 * Changes the Orb state and updates visual feedback
 * @param {'idle' | 'listening' | 'thinking' | 'speaking' | 'error'} newState
 * @param {string} [customStatus]
 * @param {string} [customSubstatus]
 */
function setOrbState(newState, customStatus, customSubstatus) {
  state.orbState = newState;
  if (voiceStage) {
    voiceStage.setAttribute('data-state', newState);
  }

  // Update button mic visual classes
  if (btnMic) {
    btnMic.classList.remove('idle', 'recording', 'thinking', 'speaking', 'error');
    btnMic.classList.add(newState);
  }

  // Default labels per state
  let statusText = 'Ready';
  let substatusText = 'Tap the orb or mic to speak with TheONE AI';

  switch (newState) {
    case 'idle':
      statusText = 'Ready';
      substatusText = 'Tap the orb or mic to speak with TheONE AI';
      if (recordingTimer) recordingTimer.style.display = 'none';
      break;

    case 'listening':
      statusText = 'Listening...';
      substatusText = 'Speak clearly. Tap again to send.';
      if (recordingTimer) recordingTimer.style.display = 'inline-flex';
      break;

    case 'thinking':
      statusText = 'Thinking...';
      substatusText = 'TheONE AI is crafting a witty comeback...';
      if (recordingTimer) recordingTimer.style.display = 'none';
      break;

    case 'speaking':
      statusText = 'Speaking...';
      const activeVoice = getActiveVoice();
      substatusText = `TheONE AI is replying (${activeVoice} voice)`;
      if (recordingTimer) recordingTimer.style.display = 'none';
      break;

    case 'error':
      statusText = 'Connection Error';
      substatusText = 'Tap the orb or mic to retry';
      if (recordingTimer) recordingTimer.style.display = 'none';
      break;
  }

  if (orbStatusText) orbStatusText.textContent = customStatus || statusText;
  if (recordingState) recordingState.textContent = customSubstatus || substatusText;
}

// =============================================================================
// Audio-Reactive Canvas Visualizer (The Orb Outer Rings & Waves)
// =============================================================================
let canvasCtx = null;
let animationFrameId = null;
let ripplePhase = 0;
let rotationAngle = 0;

function initCanvasVisualizer() {
  if (!orbCanvas) return;
  canvasCtx = orbCanvas.getContext('2d');
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  requestAnimationFrame(renderCanvasLoop);
}

function resizeCanvas() {
  if (!orbCanvas) return;
  const rect = orbCanvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  orbCanvas.width = (rect.width || 480) * dpr;
  orbCanvas.height = (rect.height || 480) * dpr;
}

function getActiveAccentColors() {
  return {
    primary: 'rgba(6, 182, 212, ', // Electric Cyan
    secondary: 'rgba(59, 130, 246, ', // Electric Blue
    tertiary: 'rgba(139, 92, 246, ', // Violet
  };
}

function updateAudioAmplitude() {
  let targetAmp = 0.0;

  if (state.orbState === 'listening' && state.micAnalyser) {
    const data = new Uint8Array(state.micAnalyser.frequencyBinCount);
    state.micAnalyser.getByteFrequencyData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i];
    const avg = sum / data.length;
    targetAmp = Math.min(1.0, (avg / 128) * 1.6);
  } else if (state.orbState === 'speaking' && state.playbackAnalyser) {
    const data = new Uint8Array(state.playbackAnalyser.frequencyBinCount);
    state.playbackAnalyser.getByteFrequencyData(data);
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i];
    const avg = sum / data.length;
    targetAmp = Math.min(1.0, (avg / 120) * 1.5);
  } else if (state.orbState === 'thinking') {
    targetAmp = 0.35 + 0.15 * Math.sin(ripplePhase * 3);
  } else if (state.orbState === 'idle') {
    targetAmp = 0.08 + 0.04 * Math.sin(ripplePhase);
  }

  // Smooth lerp for fluid organic motion
  state.currentAmplitude += (targetAmp - state.currentAmplitude) * 0.18;
}

function renderCanvasLoop() {
  if (!canvasCtx || !orbCanvas) return;
  const w = orbCanvas.width;
  const h = orbCanvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const baseRadius = Math.min(w, h) * 0.32;

  updateAudioAmplitude();
  ripplePhase += 0.04;
  rotationAngle += (state.orbState === 'thinking' ? 0.045 : 0.015);

  canvasCtx.clearRect(0, 0, w, h);

  const colors = getActiveAccentColors();
  const amp = state.currentAmplitude;

  if (state.orbState === 'idle') {
    // STATE 1: Calm gentle breathing waves
    for (let i = 0; i < 2; i++) {
      const ringR = baseRadius + (i * 22) + Math.sin(ripplePhase + i) * 6;
      const alpha = (0.12 - i * 0.04) * (0.8 + amp);
      canvasCtx.beginPath();
      canvasCtx.arc(cx, cy, ringR, 0, Math.PI * 2);
      canvasCtx.strokeStyle = `${colors.primary}${alpha})`;
      canvasCtx.lineWidth = 1.5;
      canvasCtx.stroke();
    }
  } else if (state.orbState === 'listening') {
    // STATE 2: Expands and contracts, reacts to microphone input volume
    const ringCount = 3;
    for (let i = 0; i < ringCount; i++) {
      const spread = (ripplePhase * 40 + i * 45) % 110;
      const r = baseRadius + 10 + spread + amp * 35;
      const alpha = Math.max(0, (1 - spread / 110) * (0.25 + amp * 0.65));

      canvasCtx.beginPath();
      canvasCtx.arc(cx, cy, r, 0, Math.PI * 2);
      canvasCtx.strokeStyle = `${colors.primary}${alpha})`;
      canvasCtx.lineWidth = 2 + amp * 3;
      canvasCtx.stroke();
    }

    // Secondary subtle outer glow ring
    canvasCtx.beginPath();
    canvasCtx.arc(cx, cy, baseRadius + 18 + amp * 20, 0, Math.PI * 2);
    canvasCtx.strokeStyle = `${colors.secondary}${0.35 + amp * 0.45})`;
    canvasCtx.lineWidth = 2;
    canvasCtx.stroke();
  } else if (state.orbState === 'thinking') {
    // STATE 3: Smooth rotating flowing orbital arcs
    const arcRadius = baseRadius + 18;
    for (let i = 0; i < 3; i++) {
      const startAngle = rotationAngle + (i * (Math.PI * 2 / 3));
      const endAngle = startAngle + (Math.PI * 0.45);
      canvasCtx.beginPath();
      canvasCtx.arc(cx, cy, arcRadius + i * 6, startAngle, endAngle);
      canvasCtx.strokeStyle = `${colors.primary}${0.65 - i * 0.15})`;
      canvasCtx.lineWidth = 2.5;
      canvasCtx.lineCap = 'round';
      canvasCtx.stroke();
    }
  } else if (state.orbState === 'speaking') {
    // STATE 4: Dynamic vibrant ripples reacting to audio playback amplitude
    const ringCount = 4;
    for (let i = 0; i < ringCount; i++) {
      const spread = (ripplePhase * 50 + i * 35) % 130;
      const r = baseRadius + 12 + spread + (amp * 40);
      const alpha = Math.max(0, (1 - spread / 130) * (0.3 + amp * 0.7));

      canvasCtx.beginPath();
      canvasCtx.arc(cx, cy, r, 0, Math.PI * 2);
      canvasCtx.strokeStyle = (i % 2 === 0 ? colors.primary : colors.secondary) + `${alpha})`;
      canvasCtx.lineWidth = 2.5 + amp * 3.5;
      canvasCtx.stroke();
    }
  } else if (state.orbState === 'error') {
    // STATE 5: Subtle warm warning indication
    canvasCtx.beginPath();
    canvasCtx.arc(cx, cy, baseRadius + 15, 0, Math.PI * 2);
    canvasCtx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
    canvasCtx.lineWidth = 2;
    canvasCtx.stroke();
  }

  animationFrameId = requestAnimationFrame(renderCanvasLoop);
}

// =============================================================================
// Voice Selection Management (Independent Per Mode)
// =============================================================================
// Voice Selection Management (Independent of Humour Style)
// =============================================================================
function getActiveVoice() {
  return state.activeVoice || 'Yashvi';
}

function updateVoiceUI() {
  const currentVoice = getActiveVoice();

  // Update navigation voice pills
  if (btnNavVoiceYashvi && btnNavVoiceZayan) {
    btnNavVoiceYashvi.classList.toggle('active', currentVoice === 'Yashvi');
    btnNavVoiceZayan.classList.toggle('active', currentVoice === 'Zayan');
  }

  // Update settings drawer cards
  voiceOptionCards.forEach((card) => {
    const cardVoice = card.dataset.voice;
    const statusPill = card.querySelector('.voice-status-pill');
    if (cardVoice === currentVoice) {
      card.classList.add('active');
      if (statusPill) statusPill.textContent = 'Selected';
    } else {
      card.classList.remove('active');
      if (statusPill) statusPill.textContent = 'Select';
    }
  });

  if (ttsVoiceBadgeText) {
    const genderTag = currentVoice === 'Yashvi' ? 'Female' : 'Male';
    ttsVoiceBadgeText.textContent = `${currentVoice} (${genderTag})`;
    if (ttsVoiceBadge) {
      ttsVoiceBadge.title = `Active Gnani Timbre Voice: ${currentVoice} (${genderTag})`;
    }
  }
}

function selectVoice(voiceName) {
  if (!voiceName) return;
  state.activeVoice = voiceName;
  try {
    localStorage.setItem('theone_voice', voiceName);
  } catch (_) {}
  updateVoiceUI();
}

// =============================================================================
// Intelligence Engine & Chat Integration
// =============================================================================
async function sendChatMessage(messageText) {
  const text = (messageText || '').trim();
  if (!text) return;

  hideError();
  setOrbState('thinking');

  // Append user bubble to transcript history
  appendMessageBubble('user', text, state.activeMode);
  if (textInput) textInput.value = '';

  try {
    const payload = {
      conversation_id: state.conversationId,
      message: text,
      mode: state.activeMode,
      language: languageSelect ? (languageSelect.value.split('-')[0] || 'en') : 'en',
    };

    const res = await fetch(`${state.apiBaseUrl}/api/v1/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json().catch(() => null);

    if (!res.ok) {
      const detailMsg = (data && (data.error || data.detail)) || `Chat error (HTTP ${res.status})`;
      throw new Error(detailMsg);
    }

    if (data && data.success) {
      state.conversationId = data.conversation_id;
      const reply = data.assistant_response || '';

      // Append bot response to transcript history drawer
      const botBubble = appendMessageBubble('bot', reply, data.mode, data.model);

      // Update Live Floating Transcript Card on main stage
      updateFloatingTranscript(text, reply, data.mode);

      // Synthesize and play audio response via Gnani Timbre TTS
      await synthesizeAndPlayAudio(botBubble, reply, data.mode);
    } else {
      throw new Error((data && data.error) || 'Failed to receive a response.');
    }
  } catch (err) {
    setOrbState('error', 'Error', err.message || 'Chat processing failed.');
    showError(err.message || 'Chat processing failed.');
    setTimeout(() => {
      if (state.orbState === 'error') setOrbState('idle');
    }, 4000);
  }
}

// =============================================================================
// Audio Synthesis & Web Audio Playback (Gnani Timbre TTS)
// =============================================================================
async function synthesizeAndPlayAudio(bubble, text, mode) {
  const btnReplay = bubble ? bubble.querySelector('.btn-replay-audio') : null;
  const voicePill = bubble ? bubble.querySelector('.voice-badge-pill') : null;

  if (btnReplay) {
    btnReplay.classList.add('loading');
    btnReplay.innerHTML = '<span>⏳</span><span>Synthesizing voice...</span>';
  }

  try {
    const activeVoice = getActiveVoice();
    const res = await fetch(`${state.apiBaseUrl}/api/v1/tts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        mode: 'unified_comedy',
        voice: activeVoice,
        language: languageSelect ? languageSelect.value : 'en-IN',
      }),
    });

    const data = await res.json().catch(() => null);

    if (!res.ok || !data || !data.success || !data.audio_url) {
      const errMsg = (data && (data.error || data.detail)) || 'Voice synthesis unavailable';
      if (btnReplay) {
        btnReplay.classList.remove('loading');
        btnReplay.innerHTML = '<span>🔇</span><span>Voice unavailable</span>';
        btnReplay.title = errMsg;
      }
      setOrbState('idle');
      return;
    }

    if (voicePill && data.voice) {
      voicePill.textContent = `🎙️ ${data.voice}`;
    }

    // Fetch binary audio data for Web Audio API analysis & playback
    const audioRes = await fetch(data.audio_url);
    const audioBlob = await audioRes.blob();
    const arrayBuffer = await audioBlob.arrayBuffer();

    const audioCtx = getAudioContext();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    state.activeAudioBuffer = audioBuffer;
    state.lastAudioBlobUrl = URL.createObjectURL(audioBlob);

    // Setup Replay Button handlers in transcript
    if (btnReplay) {
      btnReplay.classList.remove('loading');
      btnReplay.innerHTML = '<span>▶️</span><span>Play Voice</span>';
      btnReplay.onclick = () => playAudioBuffer(audioBuffer, btnReplay);
    }

    // Setup Replay Button on the Floating Transcript
    if (btnFloatReplay) {
      btnFloatReplay.style.display = 'inline-flex';
      btnFloatReplay.onclick = () => playAudioBuffer(audioBuffer, btnFloatReplay);
    }

    // Automatically trigger audio playback and transition to STATE 4 – SPEAKING
    await playAudioBuffer(audioBuffer, btnReplay);
  } catch (err) {
    console.warn('Audio synthesis/playback failed:', err);
    if (btnReplay) {
      btnReplay.classList.remove('loading');
      btnReplay.innerHTML = '<span>🔇</span><span>Voice error</span>';
      btnReplay.title = err.message || 'TTS request failed';
    }
    setOrbState('idle');
  }
}

async function playAudioBuffer(buffer, buttonElement) {
  if (!buffer) return;
  const audioCtx = getAudioContext();

  // Stop any currently playing audio source
  stopActiveAudio();

  // Create Source and AnalyserNode for real-time reactivity
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;

  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.8;

  source.connect(analyser);
  analyser.connect(audioCtx.destination);

  state.activeAudioSource = source;
  state.playbackAnalyser = analyser;

  // Set Orb to STATE 4 – SPEAKING
  setOrbState('speaking');

  if (buttonElement) {
    buttonElement.classList.add('playing');
    buttonElement.innerHTML = '<span>🔊</span><span>Playing...</span>';
  }

  source.onended = () => {
    stopActiveAudio();
    if (buttonElement) {
      buttonElement.classList.remove('playing');
      buttonElement.innerHTML = '<span>🔁</span><span>Replay Voice</span>';
    }
    // Return to STATE 1 – IDLE
    setOrbState('idle');
  };

  try {
    source.start(0);
  } catch (err) {
    console.warn('Buffer start error:', err);
    stopActiveAudio();
    setOrbState('idle');
  }
}

function stopActiveAudio() {
  if (state.activeAudioSource) {
    try {
      state.activeAudioSource.stop();
      state.activeAudioSource.disconnect();
    } catch (_) {}
    state.activeAudioSource = null;
  }
  state.playbackAnalyser = null;
}

// =============================================================================
// Floating Transcript Card & History Drawer
// =============================================================================
function updateFloatingTranscript(userText, botText, mode) {
  if (!floatingTranscript) return;

  if (floatUserText) floatUserText.textContent = userText;
  if (floatBotText) floatBotText.textContent = botText;

  if (floatModePill) {
    floatModePill.textContent = 'TheONE AI';
    floatModePill.className = 'transcript-mode-pill tag-comedy';
  }

  floatingTranscript.style.display = 'flex';
}

function appendMessageBubble(role, content, mode, model) {
  state.transcriptCount++;
  if (transcriptCount) {
    transcriptCount.textContent = String(state.transcriptCount);
    transcriptCount.style.display = 'inline-block';
  }

  if (!chatStream) return null;

  const isBot = role === 'bot';
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${isBot ? 'bot-bubble' : 'user-bubble'}`;

  const avatar = isBot ? '✨' : '👤';
  const senderName = isBot ? 'TheONE AI' : 'You';
  const tagClass = 'tag-comedy';
  const tagLabel = 'Comedy AI';
  const modelText = model ? ` • ${model}` : '';
  const initialVoiceName = getActiveVoice();

  const footerHtml = isBot ? `
    <div class="bubble-footer">
      <div class="audio-control-bar">
        <button type="button" class="btn-replay-audio" title="Listen to response via Gnani Timbre">
          <span>🔊</span>
          <span class="replay-label">Play Voice</span>
        </button>
        <span class="voice-badge-pill">🎙️ ${initialVoiceName}</span>
      </div>
    </div>
  ` : '';

  bubble.innerHTML = `
    <div class="bubble-header">
      <span class="bubble-avatar">${avatar}</span>
      <span class="bubble-sender">${senderName}</span>
      ${isBot ? `<span class="bubble-tag ${tagClass}">${tagLabel}${modelText}</span>` : ''}
      <span class="bubble-time">Just now</span>
    </div>
    <div class="bubble-content">${escapeHtml(content)}</div>
    ${footerHtml}
  `;

  chatStream.appendChild(bubble);
  chatStream.scrollTop = chatStream.scrollHeight;
  return bubble;
}

async function clearConversationHistory() {
  if (state.conversationId) {
    try {
      await fetch(`${state.apiBaseUrl}/api/v1/chat/clear`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id: state.conversationId }),
      });
    } catch (_) {}
  }

  state.conversationId = null;
  state.transcriptCount = 0;
  if (transcriptCount) transcriptCount.style.display = 'none';

  if (chatStream) {
    chatStream.innerHTML = `
      <div class="chat-bubble bot-bubble">
        <div class="bubble-header">
          <span class="bubble-avatar">✨</span>
          <span class="bubble-sender">TheONE AI</span>
          <span class="bubble-tag tag-funny">Always Funny</span>
          <span class="bubble-time">Just now</span>
        </div>
        <div class="bubble-content">
          Conversation history cleared. Ready for fresh jokes or roasts! Tap the orb or speak.
        </div>
      </div>
    `;
  }

  if (floatingTranscript) {
    floatingTranscript.style.display = 'none';
  }

  setOrbState('idle');
}

// =============================================================================
// Voice STT Recording (Gnani Prisma v3) -> Automatic Chat Pipeline
// =============================================================================
async function startRecording() {
  hideError();
  if (state.isRecording) return;

  // Stop any active speech playback
  stopActiveAudio();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showError('Microphone recording is not supported in this browser.');
    return;
  }

  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (err) {
    setOrbState('error', 'Mic Denied', 'Microphone permission was denied.');
    showError(err.name === 'NotAllowedError' ? 'Microphone permission denied.' : `Microphone error: ${err.message}`);
    return;
  }

  try {
    const audioCtx = getAudioContext();
    const source = audioCtx.createMediaStreamSource(state.mediaStream);

    // Connect AnalyserNode for real-time microphone volume visualization
    state.micAnalyser = audioCtx.createAnalyser();
    state.micAnalyser.fftSize = 256;
    state.micAnalyser.smoothingTimeConstant = 0.6;
    source.connect(state.micAnalyser);

    state.scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
    state.pcmChunks = [];

    state.scriptProcessor.onaudioprocess = (e) => {
      if (!state.isRecording) return;
      state.pcmChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };

    source.connect(state.scriptProcessor);
    state.scriptProcessor.connect(audioCtx.destination);

    state.isRecording = true;
    state.recordingStartTime = Date.now();
    state.timerInterval = setInterval(updateRecordingTimer, 500);

    // Transition to STATE 2 – LISTENING
    setOrbState('listening');
  } catch (err) {
    releaseMicrophone();
    setOrbState('error', 'Recorder Error', err.message);
    showError(`Error opening audio recorder: ${err.message}`);
  }
}

async function stopRecording() {
  if (!state.isRecording) return;
  state.isRecording = false;

  clearInterval(state.timerInterval);
  state.timerInterval = null;

  setOrbState('thinking', 'Processing...', 'Transcribing speech with Gnani Prisma...');

  const sourceSampleRate = state.audioContext ? state.audioContext.sampleRate : 44100;
  releaseMicrophone();

  let totalLength = 0;
  for (const chunk of state.pcmChunks) totalLength += chunk.length;

  if (totalLength === 0) {
    showError('No audio was captured.');
    setOrbState('idle');
    return;
  }

  const combined = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of state.pcmChunks) {
    combined.set(chunk, offset);
    offset += chunk.length;
  }

  // Downsample to 16 kHz Mono WAV for Prisma STT
  const targetRate = 16000;
  const resampled = downsampleBuffer(combined, sourceSampleRate, targetRate);
  const wavBlob = encodeWAV(resampled, targetRate);

  if (resampled.length / targetRate < 0.25) {
    showError('Recording was too short.');
    setOrbState('idle');
    return;
  }

  // Upload to Prisma STT
  await processVoiceSTT(wavBlob);
}

function releaseMicrophone() {
  if (state.scriptProcessor) {
    try { state.scriptProcessor.disconnect(); } catch (_) {}
    state.scriptProcessor = null;
  }
  if (state.micAnalyser) {
    try { state.micAnalyser.disconnect(); } catch (_) {}
    state.micAnalyser = null;
  }
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((t) => t.stop());
    state.mediaStream = null;
  }
}

function updateRecordingTimer() {
  if (!state.recordingStartTime) return;
  const elapsed = Math.floor((Date.now() - state.recordingStartTime) / 1000);
  const m = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const s = String(elapsed % 60).padStart(2, '0');
  if (recordingTimer) recordingTimer.textContent = `${m}:${s}`;
  if (elapsed >= 60) stopRecording();
}

function downsampleBuffer(buffer, sampleRate, outSampleRate) {
  if (outSampleRate >= sampleRate) return buffer;
  const ratio = sampleRate / outSampleRate;
  const newLen = Math.round(buffer.length / ratio);
  const res = new Float32Array(newLen);
  let offsetRes = 0;
  let offsetBuf = 0;
  while (offsetRes < res.length) {
    const nextOffsetBuf = Math.round((offsetRes + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuf; i < nextOffsetBuf && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    res[offsetRes] = count > 0 ? accum / count : 0;
    offsetRes++;
    offsetBuf = nextOffsetBuf;
  }
  return res;
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

async function processVoiceSTT(wavBlob) {
  const formData = new FormData();
  formData.append('audio_file', wavBlob, 'recording.wav');
  formData.append('language_code', languageSelect ? languageSelect.value : 'en-IN');

  try {
    const res = await fetch(`${state.apiBaseUrl}/api/v1/stt/transcribe`, {
      method: 'POST',
      body: formData,
    });
    const data = await res.json().catch(() => null);

    if (!res.ok) {
      throw new Error((data && (data.error || data.detail)) || `STT Error (HTTP ${res.status})`);
    }

    if (data && data.success) {
      const transcript = (data.transcript || '').trim();
      if (transcript) {
        showSttBanner(`Transcribed: "${transcript}"`);
        // Forward to Chat Pipeline
        await sendChatMessage(transcript);
      } else {
        showSttBanner('No speech detected in audio clip.');
        setOrbState('idle');
      }
    }
  } catch (err) {
    setOrbState('error', 'STT Failed', err.message);
    showError(err.message || 'Speech recognition failed.');
    setTimeout(() => {
      if (state.orbState === 'error') setOrbState('idle');
    }, 4000);
  }
}

function showSttBanner(msg) {
  if (!sttStatusBanner || !sttStatusMessage) return;
  sttStatusMessage.textContent = msg;
  sttStatusBanner.classList.remove('hidden');
  setTimeout(() => sttStatusBanner.classList.add('hidden'), 4500);
}

// =============================================================================
// Backend Health Check
// =============================================================================
async function checkBackendHealth() {
  try {
    const res = await fetch(`${state.apiBaseUrl}/health`);
    if (res.ok) {
      const data = await res.json();
      if (data && data.status === 'healthy') {
        if (backendStatus) backendStatus.className = 'live-status-dot online';
        if (backendStatusText) backendStatusText.textContent = 'Online';
        return;
      }
    }
    setDisconnected();
  } catch (_) {
    setDisconnected();
  }
}

function setDisconnected() {
  if (backendStatus) backendStatus.className = 'live-status-dot offline';
  if (backendStatusText) backendStatusText.textContent = 'Offline';
}

function showError(msg) {
  if (!errorBox || !errorMessage) return;
  errorMessage.textContent = msg;
  errorBox.classList.remove('hidden');
}

function hideError() {
  if (errorBox) errorBox.classList.add('hidden');
}

function escapeHtml(unsafe) {
  return String(unsafe)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// =============================================================================
// Drawer Toggles & Theme Control
// =============================================================================
function openTranscriptDrawer() {
  if (transcriptDrawer) transcriptDrawer.classList.add('open');
  if (transcriptBackdrop) transcriptBackdrop.classList.add('open');
}

function closeTranscriptDrawer() {
  if (transcriptDrawer) transcriptDrawer.classList.remove('open');
  if (transcriptBackdrop) transcriptBackdrop.classList.remove('open');
}

function openSettingsDrawer() {
  if (settingsDrawer) settingsDrawer.classList.add('open');
  if (settingsBackdrop) settingsBackdrop.classList.add('open');
}

function closeSettingsDrawer() {
  if (settingsDrawer) settingsDrawer.classList.remove('open');
  if (settingsBackdrop) settingsBackdrop.classList.remove('open');
}

function toggleTheme() {
  const isDark = document.body.classList.contains('dark-mode');
  if (isDark) {
    document.body.classList.remove('dark-mode');
    document.body.classList.add('light-mode');
  } else {
    document.body.classList.remove('light-mode');
    document.body.classList.add('dark-mode');
  }

  const iconDark = document.querySelector('.theme-icon-dark');
  const iconLight = document.querySelector('.theme-icon-light');
  if (iconDark && iconLight) {
    iconDark.style.display = isDark ? 'none' : 'inline';
    iconLight.style.display = isDark ? 'inline' : 'none';
  }
}

// =============================================================================
// Event Handlers & Initialization
// =============================================================================
function setupEventListeners() {
  // Voice selection (Yashvi & Zayan) in top nav
  if (btnNavVoiceYashvi) btnNavVoiceYashvi.addEventListener('click', () => selectVoice('Yashvi'));
  if (btnNavVoiceZayan) btnNavVoiceZayan.addEventListener('click', () => selectVoice('Zayan'));

  // The Orb click toggles voice conversation
  if (voiceOrb) {
    voiceOrb.addEventListener('click', () => {
      if (state.isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    });
  }

  // Mic Button click
  if (btnMic) {
    btnMic.addEventListener('click', () => {
      if (state.isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    });
  }

  // Text message submission
  if (chatForm) {
    chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      if (textInput && textInput.value.trim()) {
        sendChatMessage(textInput.value.trim());
      }
    });
  }

  // Clear chat
  if (btnClearChat) btnClearChat.addEventListener('click', clearConversationHistory);

  // Voice selections in Settings
  if (btnVoiceYashvi) btnVoiceYashvi.addEventListener('click', () => selectVoice('Yashvi'));
  if (btnVoiceZayan) btnVoiceZayan.addEventListener('click', () => selectVoice('Zayan'));

  // Drawer toggles
  if (btnToggleTranscript) btnToggleTranscript.addEventListener('click', openTranscriptDrawer);
  if (btnCloseTranscript) btnCloseTranscript.addEventListener('click', closeTranscriptDrawer);
  if (transcriptBackdrop) transcriptBackdrop.addEventListener('click', closeTranscriptDrawer);
  if (btnOpenHistoryLink) btnOpenHistoryLink.addEventListener('click', openTranscriptDrawer);

  if (btnOpenSettings) btnOpenSettings.addEventListener('click', openSettingsDrawer);
  if (btnCloseSettings) btnCloseSettings.addEventListener('click', closeSettingsDrawer);
  if (settingsBackdrop) settingsBackdrop.addEventListener('click', closeSettingsDrawer);

  // Theme toggle
  if (btnToggleTheme) btnToggleTheme.addEventListener('click', toggleTheme);

  // Toast close
  if (btnDismissError) btnDismissError.addEventListener('click', hideError);
}

document.addEventListener('DOMContentLoaded', () => {
  initCanvasVisualizer();
  updateVoiceUI();
  setOrbState('idle');
  setupEventListeners();
  checkBackendHealth();
  setInterval(checkBackendHealth, 5000);
});
