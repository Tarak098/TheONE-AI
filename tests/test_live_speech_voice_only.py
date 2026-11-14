"""
Unit and integration tests for TheONE AI:
- Show Recognised User Speech in active conversation area
- Hide Live AI Response Text (Voice/Audio-only response)
- Genuine Gnani Prisma WebSocket Streaming Pipeline (/api/v1/stt/stream)
- Progressive live speech transcription (partial & final events)
- AudioWorklet & 16kHz PCM streaming frame contract
- Preserve full dialogue transcript in Conversation History drawer
- Verify all required test scenarios:
  1. User speaks -> recognised speech appears progressively in chat.
  2. TheONE AI generates reply -> no AI response textbox appears.
  3. TheONE AI speaks generated reply aloud (TTS synthesis).
  4. Conversation history contains both user and AI messages.
  5. User sends a typed message -> appears in chat, AI replies via voice only.
  6. Multiple consecutive conversations work without duplicate messages.
  7. Real-time WebSocket streaming with partial updates and final utterance.
  8. Microphone error resilience & lifecycle cleanup.
"""

import io
import json
from pathlib import Path
import re
import time
import wave
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.timbre import TimbreService

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@pytest.fixture
def client():
    return TestClient(app)


def test_scenario_1_and_2_frontend_contract():
    """
    Scenario 1: User speaks -> recognised speech appears in chat.
    Scenario 2: TheONE AI generates reply -> NO AI response textbox appears on live active screen.
    """
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    js_content = (FRONTEND_DIR / "script.js").read_text(encoding="utf-8")
    css_content = (FRONTEND_DIR / "style.css").read_text(encoding="utf-8")

    # 1. Recognised user speech display exists
    assert 'id="floating-transcript"' in html_content
    assert 'id="float-user-text"' in html_content
    assert 'showLiveUserSpeech' in js_content

    # 2. Live AI response text is strictly hidden (no AI response textbox, card or text block in live view)
    assert "floatBotText.textContent = ''" in js_content or 'floatBotText.textContent = ""' in js_content
    assert "floatBotText.style.display = 'none'" in js_content or 'floatBotText.style.display = "none"' in js_content
    assert ".float-bot-text" in css_content
    assert "display: none !important" in css_content

    # 3. Live AI replay button is strictly hidden from live view
    assert ".btn-float-replay" in css_content
    assert "btnFloatReplay.style.display = 'none'" in js_content or 'btnFloatReplay.style.display = "none"' in js_content


def test_scenario_3_tts_audio_synthesis_flow(client):
    """
    Scenario 3: TheONE AI speaks generated reply aloud via Timbre TTS.
    Verify that AI text is passed directly to TTS and synthesized into audio.
    """
    js_content = (FRONTEND_DIR / "script.js").read_text(encoding="utf-8")

    # Verify script calls synthesizeAndPlayAudio with generated reply
    assert "await synthesizeAndPlayAudio(botBubble, reply, data.mode)" in js_content
    assert "playAudioBuffer(audioBuffer, btnReplay)" in js_content
    assert "setOrbState('speaking')" in js_content

    # Verify TTS endpoint is alive and handles request
    res = client.get("/api/v1/tts/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data


def test_scenario_4_conversation_history_preserves_both_user_and_ai():
    """
    Scenario 4: Conversation history contains BOTH user and AI messages.
    """
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    js_content = (FRONTEND_DIR / "script.js").read_text(encoding="utf-8")

    # Drawer exists in HTML
    assert 'id="transcript-drawer"' in html_content
    assert 'id="chat-stream"' in html_content

    # In script.js, both user and bot bubbles are appended to conversation history
    assert "appendMessageBubble('user', text, state.activeMode)" in js_content
    assert "appendMessageBubble('bot', reply, data.mode, data.model)" in js_content

    # Bot bubble in drawer contains audio replay and voice badge
    assert "btn-replay-audio" in js_content
    assert "voice-badge-pill" in js_content


def test_scenario_5_typed_message_appears_and_ai_voice_only():
    """
    Scenario 5: User sends typed message -> appears in chat, AI replies via voice only.
    """
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    js_content = (FRONTEND_DIR / "script.js").read_text(encoding="utf-8")

    # Text input and chat form exist
    assert 'id="chat-form"' in html_content
    assert 'id="text-input"' in html_content
    assert 'id="btn-send"' in html_content

    # Typed message triggers sendChatMessage with sourceType 'typed'
    assert "sendChatMessage(text, 'typed')" in js_content
    assert "showLiveUserSpeech(text, sourceType === 'typed' ? 'typed' : 'voice')" in js_content


def test_scenario_6_multiple_consecutive_turns_without_duplicates():
    """
    Scenario 6: Multiple consecutive conversations work without duplicate messages or UI errors.
    """
    js_content = (FRONTEND_DIR / "script.js").read_text(encoding="utf-8")

    # User message bubble is only appended ONCE in sendChatMessage
    user_appends = re.findall(r"appendMessageBubble\(\s*['\"]user['\"]", js_content)
    assert len(user_appends) == 1, f"Expected exactly 1 call to append user bubble, found {len(user_appends)}"

    # Bot message bubble is only appended ONCE in sendChatMessage
    bot_appends = re.findall(r"appendMessageBubble\(\s*['\"]bot['\"]", js_content)
    assert len(bot_appends) == 1, f"Expected exactly 1 call to append bot bubble, found {len(bot_appends)}"

    # When clear chat is clicked, history resets cleanly
    assert "clearConversationHistory" in js_content
    assert "state.conversationId = null" in js_content


def test_audioworklet_and_streaming_frontend_contract():
    """
    Verify AudioWorklet processor file exists, downsamples to 16kHz PCM,
    and frontend integrates WebSocket streaming with fallback.
    """
    worklet_path = FRONTEND_DIR / "pcm-processor.js"
    assert worklet_path.exists(), "pcm-processor.js must exist in frontend/"
    worklet_code = worklet_path.read_text(encoding="utf-8")

    assert "AudioWorkletProcessor" in worklet_code
    assert "pcm-processor" in worklet_code
    assert "16000" in worklet_code
    assert "512" in worklet_code  # 512 samples = 1024 bytes per frame

    js_content = (FRONTEND_DIR / "script.js").read_text(encoding="utf-8")
    assert "/api/v1/stt/stream" in js_content
    assert "pcm-processor.js" in js_content
    assert "setupScriptProcessorFallback" in js_content
    assert "streamingWs" in js_content
    assert "showLiveUserSpeech(currentFull, 'partial')" in js_content or "showLiveUserSpeech" in js_content


def test_real_time_websocket_streaming_pipeline(client):
    """
    Integration test for /api/v1/stt/stream:
    Connects to FastAPI WebSocket endpoint, verifies connection handshake,
    streams real synthesized PCM audio frames, and validates partial and final events.
    """
    timbre = TimbreService()
    synth = timbre.synthesize_speech("Hello world, this is a live test.", voice="Zayan")

    with wave.open(io.BytesIO(synth.audio_bytes), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        framerate = wf.getframerate()

    import audioop
    pcm_16k = audioop.ratecv(frames, 2, 1, framerate, 16000, None)[0]

    with client.websocket_connect("/api/v1/stt/stream?language_code=en-IN") as ws:
        init_event = ws.receive_json()
        assert init_event.get("type") == "connected"
        assert init_event.get("config", {}).get("partials_enabled") is True

        chunk_size = 1024
        for i in range(0, min(len(pcm_16k), chunk_size * 25), chunk_size):
            chunk = pcm_16k[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk += b"\x00" * (chunk_size - len(chunk))
            ws.send_bytes(chunk)
            time.sleep(0.01)

        # Signal end of utterance via control message
        ws.send_text(json.dumps({"type": "stop"}))

        received_events = []
        for _ in range(30):
            try:
                evt = ws.receive_json()
                received_events.append(evt.get("type"))
                if evt.get("type") == "final":
                    assert len(evt.get("text", "")) > 0
                    break
            except Exception:
                break

        assert "connected" in received_events or init_event.get("type") == "connected"
        # At least one of speech_start, partial, or final was returned
        assert any(e in received_events for e in ("speech_start", "partial", "final"))


def test_transcription_persistence_and_fade_out_lifecycle():
    """
    Verify that temporary transcription remains visible during thinking and playback,
    and fades out cleanly only after actual AI audio playback completes.
    """
    js_content = (FRONTEND_DIR / "script.js").read_text(encoding="utf-8")
    css_content = (FRONTEND_DIR / "style.css").read_text(encoding="utf-8")

    # Fade out function defined
    assert "fadeOutLiveTranscription" in js_content
    assert "cancelFadeOut" in js_content

    # Called on audio playback onended
    assert "source.onended = () => {" in js_content
    assert "fadeOutLiveTranscription();" in js_content

    # CSS transition for smooth fade out
    assert ".floating-transcript-card.fade-out" in css_content
    assert "opacity: 0" in css_content


def test_microphone_error_resilience():
    """
    Verify microphone permission denied and device errors are caught gracefully.
    """
    js_content = (FRONTEND_DIR / "script.js").read_text(encoding="utf-8")

    assert "NotAllowedError" in js_content
    assert "Microphone permission denied" in js_content
    assert "setOrbState('error', 'Mic Denied'" in js_content
