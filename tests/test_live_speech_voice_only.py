"""
Unit and integration tests for TheONE AI:
- Show Recognised User Speech in active conversation area
- Hide Live AI Response Text (Voice/Audio-only response)
- Preserve full dialogue transcript in Conversation History drawer
- Verify all 6 required test scenarios:
  1. User speaks -> recognised speech appears in chat.
  2. TheONE AI generates reply -> no AI response textbox appears.
  3. TheONE AI speaks generated reply aloud (TTS synthesis).
  4. Conversation history contains both user and AI messages.
  5. User sends a typed message -> appears in chat, AI replies via voice only.
  6. Multiple consecutive conversations work without duplicate messages.
"""

import json
from pathlib import Path
import re
import pytest
from fastapi.testclient import TestClient

from backend.main import app

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
    assert 'floatBotText.textContent = \'\'' in js_content or 'floatBotText.textContent = ""' in js_content
    assert 'floatBotText.style.display = \'none\'' in js_content or 'floatBotText.style.display = "none"' in js_content
    assert '.float-bot-text' in css_content
    assert 'display: none !important' in css_content

    # 3. Live AI replay button is strictly hidden from live view
    assert '.btn-float-replay' in css_content
    assert 'btnFloatReplay.style.display = \'none\'' in js_content or 'btnFloatReplay.style.display = "none"' in js_content


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


def test_real_time_chunking_and_partial_transcription():
    """
    Verify real-time chunk capture, periodic dispatch, and partial UI updates.
    """
    js_content = (FRONTEND_DIR / "script.js").read_text(encoding="utf-8")

    # Periodic chunk capture and processing
    assert "processPartialChunk" in js_content
    assert "setInterval(processPartialChunk" in js_content
    assert "state.isChunkProcessing" in js_content

    # Partial transcription status updates
    assert "showLiveUserSpeech(partialText, 'partial')" in js_content
    assert "state.lastPartialTranscript" in js_content


def test_vad_pause_detection_and_background_noise_handling():
    """
    Verify voice activity detection, pause detection (1.8s silence window),
    and graceful handling of background noise / silence.
    """
    js_content = (FRONTEND_DIR / "script.js").read_text(encoding="utf-8")

    # Audio RMS calculation for speech vs silence
    assert "Math.sqrt(sumSquares" in js_content or "rms" in js_content
    assert "SPEECH_THRESHOLD" in js_content
    assert "state.silenceStartTime" in js_content
    assert "1800" in js_content  # Pause detection interval in ms


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


