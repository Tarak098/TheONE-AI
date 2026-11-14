"""
TheONE AI - Unit and Integration Tests for:
1. Natural Spoken Voice Switching (Yashvi & Zayan across 10 Indian Languages & Code-Mixing)
2. Application Action routing (no unnecessary LLM joke calls for voice switches)
3. Confirmation spoken aloud in newly selected voice
4. Ambiguous command handling ("Yashvi or Zayan?")
5. Minimal Premium Interface Redesign UI Contract (Screenshots 1 & 2 inspired)
"""

from pathlib import Path
import re
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.voice_intent import (
    CONFIRMATIONS_AMBIGUOUS,
    CONFIRMATIONS_YASHVI,
    CONFIRMATIONS_ZAYAN,
    detect_voice_switch_intent,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@pytest.fixture
def client():
    return TestClient(app)


# ==============================================================================
# 1. Voice Intent Unit Tests: Explicit Switching to Zayan (Male Voice)
# ==============================================================================

@pytest.mark.parametrize(
    "phrase,lang",
    [
        ("Switch to Zayan", "en-IN"),
        ("Change your voice to Zayan", "en-IN"),
        ("Can you talk in Zayan's voice?", "en-IN"),
        ("Use a male voice", "en-IN"),
        ("I want a boy voice", "en-IN"),
        ("Speak like a guy", "en-IN"),
        ("Brother voice please", "en-IN"),
        ("జాయన్ వాయిస్‌కి మార్చు", "te-IN"),
        ("అబ్బాయి వాయిస్‌లో మాట్లాడు", "te-IN"),
        ("ज़यान की आवाज़ लगाओ", "hi-IN"),
        ("लड़के की आवाज़ में बोलो", "hi-IN"),
        ("சயான் குரலுக்கு மாற்று", "ta-IN"),
        ("ஆண் குரலில் பேசு", "ta-IN"),
        ("ಜಯಾನ್ ಧ್ವನಿಗೆ ಬದಲಾಯಿಸಿ", "kn-IN"),
        ("സയാന്റെ ശബ്ദത്തിലേക്ക് മാറ്റുക", "ml-IN"),
        ("জায়ান ভয়েসে বদল করো", "bn-IN"),
        ("झयानच्या आवाजात बदल करा", "mr-IN"),
        ("ઝયાનના અવાજમાં બદલો", "gu-IN"),
        ("ਜ਼ਯਾਨ ਦੀ ਆਵਾਜ਼ ਚਲਾਓ", "pa-IN"),
    ],
)
def test_voice_intent_explicit_zayan(phrase, lang):
    res = detect_voice_switch_intent(phrase, current_voice="Yashvi", language_code=lang)
    assert res.is_voice_switch is True, f"Failed to detect voice switch for '{phrase}'"
    assert res.target_voice == "Zayan"
    assert res.is_ambiguous is False
    assert res.voice_to_speak == "Zayan"
    assert len(res.confirmation_text) > 0


# ==============================================================================
# 2. Voice Intent Unit Tests: Explicit Switching to Yashvi (Female Voice)
# ==============================================================================

@pytest.mark.parametrize(
    "phrase,lang",
    [
        ("Switch to Yashvi", "en-IN"),
        ("Change your voice to Yashvi", "en-IN"),
        ("Talk in Yashvi voice", "en-IN"),
        ("Use a female voice", "en-IN"),
        ("I want a girl voice", "en-IN"),
        ("Speak like a lady", "en-IN"),
        ("యశ్వి వాయిస్ కి మార్చు", "te-IN"),
        ("అమ్మాయి వాయిస్‌లో మాట్లాడు", "te-IN"),
        ("यशवी की आवाज़ लगाओ", "hi-IN"),
        ("लड़की की आवाज़ में बात करो", "hi-IN"),
        ("யஷ்வி குரலுக்கு மாற்று", "ta-IN"),
        ("பெண் குரலில் பேசு", "ta-IN"),
        ("ಯಶ್ವಿ ಧ್ವನಿಗೆ ಬದಲಾಯಿಸಿ", "kn-IN"),
        ("യശ്വിയുടെ ശബ്ദത്തിലേക്ക് മാറ്റുക", "ml-IN"),
        ("যশবী ভয়েসে সুইচ করো", "bn-IN"),
        ("यशवीच्या आवाजात बोला", "mr-IN"),
        ("યશવીના અવાજમાં બદલો", "gu-IN"),
        ("ਯਸ਼ਵੀ ਦੀ ਆਵਾਜ਼ ਵਿੱਚ ਬੋਲੋ", "pa-IN"),
    ],
)
def test_voice_intent_explicit_yashvi(phrase, lang):
    res = detect_voice_switch_intent(phrase, current_voice="Zayan", language_code=lang)
    assert res.is_voice_switch is True, f"Failed to detect voice switch for '{phrase}'"
    assert res.target_voice == "Yashvi"
    assert res.is_ambiguous is False
    assert res.voice_to_speak == "Yashvi"
    assert len(res.confirmation_text) > 0


# ==============================================================================
# 3. Voice Intent Unit Tests: Relative Switching ("other voice", "previous voice")
# ==============================================================================

def test_voice_intent_relative_switches():
    # If current voice is Yashvi, "other voice" switches to Zayan
    res1 = detect_voice_switch_intent("I want your other voice", current_voice="Yashvi")
    assert res1.is_voice_switch is True
    assert res1.target_voice == "Zayan"
    assert res1.voice_to_speak == "Zayan"

    # If current voice is Zayan, "previous voice" switches to Yashvi
    res2 = detect_voice_switch_intent("Switch back to your previous voice", current_voice="Zayan")
    assert res2.is_voice_switch is True
    assert res2.target_voice == "Yashvi"
    assert res2.voice_to_speak == "Yashvi"

    # Hindi relative: "dusri awaz me bolo"
    res3 = detect_voice_switch_intent("dusri awaz me baat karo", current_voice="Yashvi", language_code="hi-IN")
    assert res3.is_voice_switch is True
    assert res3.target_voice == "Zayan"


# ==============================================================================
# 4. Voice Intent Unit Tests: Ambiguous Requests ("Change your voice")
# ==============================================================================

@pytest.mark.parametrize(
    "phrase,lang,expected_prompt",
    [
        ("Change your voice", "en-IN", "Yashvi or Zayan?"),
        ("Switch voice", "en-IN", "Yashvi or Zayan?"),
        ("వాయిస్ మార్చు", "te-IN", "యశ్వి లేదా జాయన్?"),
        ("आवाज़ बदलो", "hi-IN", "यशवी या ज़यान?"),
        ("குரல் மாற்று", "ta-IN", "யஷ்வி அல்லது சயான்?"),
        ("ಧ್ವನಿ ಬದಲಾಯಿಸಿ", "kn-IN", "ಯಶ್ವಿ ಅಥವಾ ಜಯಾನ್?"),
        ("ശബ്ദം മാറ്റുക", "ml-IN", "യശ്വി അതോ സയാൻ?"),
    ],
)
def test_voice_intent_ambiguous_requests(phrase, lang, expected_prompt):
    res = detect_voice_switch_intent(phrase, current_voice="Yashvi", language_code=lang)
    assert res.is_voice_switch is True
    assert res.target_voice is None
    assert res.is_ambiguous is True
    assert res.confirmation_text == expected_prompt
    # Must speak in current voice when asking for clarification
    assert res.voice_to_speak == "Yashvi"


# ==============================================================================
# 5. Non-voice messages must NOT trigger voice switch
# ==============================================================================

@pytest.mark.parametrize(
    "phrase",
    [
        "I'm feeling bored today",
        "Tell me a joke about engineers",
        "I failed my math exam",
        "Why is pizza round?",
        "Bro you are so savage",
    ],
)
def test_voice_intent_regular_conversations(phrase):
    res = detect_voice_switch_intent(phrase, current_voice="Yashvi")
    assert res.is_voice_switch is False
    assert res.target_voice is None


# ==============================================================================
# 6. End-to-End FastAPI Chat Endpoint Integration
# ==============================================================================

def test_api_chat_endpoint_voice_switch_zayan(client):
    """
    Verify /api/v1/chat intercepts voice switch, executes application action,
    updates active voice, and designates newly selected voice for speech output.
    """
    payload = {
        "message": "Switch to Zayan",
        "conversation_id": "test_voice_switch_session_01",
        "voice": "Yashvi",
        "language": "en-IN",
    }
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["is_voice_switch"] is True
    assert data["target_voice"] == "Zayan"
    assert data["voice_to_speak"] == "Zayan"
    assert data["assistant_response"] == "New voice, same attitude."
    assert data["model"] == "voice_switcher"


def test_api_chat_endpoint_voice_switch_yashvi(client):
    """Verify switching back to Yashvi works identically."""
    payload = {
        "message": "Use a female voice",
        "conversation_id": "test_voice_switch_session_02",
        "voice": "Zayan",
        "language": "en-IN",
    }
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["is_voice_switch"] is True
    assert data["target_voice"] == "Yashvi"
    assert data["voice_to_speak"] == "Yashvi"
    assert "Yashvi" in data["assistant_response"]


def test_api_chat_endpoint_voice_switch_ambiguous(client):
    """Verify ambiguous 'Change your voice' asks 'Yashvi or Zayan?'."""
    payload = {
        "message": "Change your voice",
        "conversation_id": "test_voice_switch_session_03",
        "voice": "Yashvi",
        "language": "en-IN",
    }
    res = client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["success"] is True
    assert data["is_voice_switch"] is True
    assert data["target_voice"] is None
    assert data["is_ambiguous"] is True
    assert data["assistant_response"] == "Yashvi or Zayan?"


# ==============================================================================
# 7. UI Contract: Minimal Header, No Emojis, No Visible Voice Tabs
# ==============================================================================

def test_ui_header_minimal_and_no_emojis():
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    css_content = (FRONTEND_DIR / "style.css").read_text(encoding="utf-8")

    # Header has brand text logo and settings icon toggle
    assert 'TheONE <span class="brand-highlight">AI</span>' in html_content
    assert 'id="btn-open-settings"' in html_content

    # No decorative emojis anywhere in the main UI
    assert not re.search(r"[\U0001F300-\U0001FAFF]", html_content)

    # Voice tabs and badges in header are hidden (no visible manual switcher)
    assert 'id="voice-selector-nav"' in html_content
    assert 'class="hidden"' in html_content


# ==============================================================================
# 8. UI Contract: Screenshot 1 Central Blue Glowing Orb
# ==============================================================================

def test_ui_central_blue_glowing_orb_contract():
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    css_content = (FRONTEND_DIR / "style.css").read_text(encoding="utf-8")

    # Central orb elements exist
    assert 'id="voice-orb"' in html_content
    assert 'id="orb-ambient-glow"' in html_content
    assert 'id="orb-canvas"' in html_content

    # Central blue orb styling in CSS
    assert ".voice-orb" in css_content
    assert "radial-gradient" in css_content
    assert "box-shadow" in css_content

    # State-driven animations exist in CSS
    assert '.voice-stage[data-state="idle"]' in css_content
    assert '.voice-stage[data-state="listening"]' in css_content
    assert '.voice-stage[data-state="thinking"]' in css_content
    assert '.voice-stage[data-state="speaking"]' in css_content
    assert '.voice-stage[data-state="error"]' in css_content


# ==============================================================================
# 9. UI Contract: Screenshot 2 Minimal Bottom Controls & Transcription
# ==============================================================================

def test_ui_bottom_controls_and_transcription_contract():
    html_content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    css_content = (FRONTEND_DIR / "style.css").read_text(encoding="utf-8")
    js_content = (FRONTEND_DIR / "script.js").read_text(encoding="utf-8")

    # Prominent central mic button
    assert 'id="btn-mic"' in html_content
    assert ".prominent-mic-btn" in css_content

    # Conversation history button & minimal text input
    assert 'id="btn-toggle-transcript"' in html_content
    assert ".history-icon-btn" in css_content
    assert 'id="chat-form"' in html_content
    assert 'id="text-input"' in html_content
    assert 'id="btn-send"' in html_content

    # Temporary live transcription contract:
    # User speech displayed, AI response text strictly hidden
    assert 'id="floating-transcript"' in html_content
    assert 'id="float-user-text"' in html_content
    assert ".float-bot-text" in css_content
    assert "display: none !important" in css_content
    assert ".floating-transcript-card.fade-out" in css_content
    assert "opacity: 0" in css_content

    # Script voice preference persistence
    assert "theone_voice_preference" in js_content
    assert "selectVoice" in js_content
