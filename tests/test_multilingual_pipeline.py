"""
TheONE AI - Multilingual Pipeline Comprehensive Test Suite.

Validates end-to-end multilingual conversation and voice support across
all 10 officially supported Indian languages:
1. Telugu (te-IN)
2. Hindi (hi-IN)
3. Tamil (ta-IN)
4. Kannada (kn-IN)
5. Malayalam (ml-IN)
6. Bengali (bn-IN)
7. Marathi (mr-IN)
8. Gujarati (gu-IN)
9. Punjabi (pa-IN)
10. English (en-IN)

Plus:
- Telugu-English & Hindi-English code-mixed conversations
- Language switching across consecutive dialogue turns
- Automatic language identification (indic-auto)
- Yashvi and Zayan voice selection across all languages
- Gnani Timbre TTS multilingual synthesis
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services.languages import (
    SUPPORTED_LANGUAGES,
    INDIC_AUTO_CODE,
    detect_language,
    detect_script_language,
    detect_codemixed_language,
    normalize_language_code,
    get_language_prompt_instruction,
    resolve_tts_voice_and_model,
)
from backend.services.intelligence import IntelligenceEngine
from backend.services.conversation import ConversationManager
from backend.services.personality import PersonalityManager
from backend.services.timbre import TimbreService, TimbreSynthesisResult
from backend.services.prisma import PrismaService, TranscriptionResult
from backend.services.prisma_stream import PrismaStreamService


# ==============================================================================
# 1. Supported Languages & Normalization Tests
# ==============================================================================
def test_all_10_languages_registered():
    expected_codes = {
        "en-IN", "te-IN", "hi-IN", "ta-IN", "kn-IN",
        "ml-IN", "bn-IN", "mr-IN", "gu-IN", "pa-IN"
    }
    assert expected_codes.issubset(set(SUPPORTED_LANGUAGES.keys()))
    assert len(SUPPORTED_LANGUAGES) >= 10


def test_language_normalization_aliases():
    assert normalize_language_code("te") == "te-IN"
    assert normalize_language_code("telugu") == "te-IN"
    assert normalize_language_code("hi") == "hi-IN"
    assert normalize_language_code("hindi") == "hi-IN"
    assert normalize_language_code("ta") == "ta-IN"
    assert normalize_language_code("tamil") == "ta-IN"
    assert normalize_language_code("kn") == "kn-IN"
    assert normalize_language_code("kannada") == "kn-IN"
    assert normalize_language_code("ml") == "ml-IN"
    assert normalize_language_code("malayalam") == "ml-IN"
    assert normalize_language_code("bn") == "bn-IN"
    assert normalize_language_code("bengali") == "bn-IN"
    assert normalize_language_code("mr") == "mr-IN"
    assert normalize_language_code("marathi") == "mr-IN"
    assert normalize_language_code("gu") == "gu-IN"
    assert normalize_language_code("gujarati") == "gu-IN"
    assert normalize_language_code("pa") == "pa-IN"
    assert normalize_language_code("punjabi") == "pa-IN"
    assert normalize_language_code("en") == "en-IN"
    assert normalize_language_code("auto") == "indic-auto"
    assert normalize_language_code("indic-auto") == "indic-auto"


# ==============================================================================
# 2. Automatic Language Detection Tests (Native Scripts & Code-Mixed)
# ==============================================================================
def test_native_script_detection():
    # Telugu
    assert detect_language("నేను రేపటి నుంచి చదువుతాను") == "te-IN"
    # Hindi
    assert detect_language("मैं कल से पढूंगा भाई") == "hi-IN"
    # Tamil
    assert detect_language("நான் நாளைக்கு படிக்க போறேன்") == "ta-IN"
    # Kannada
    assert detect_language("ನಾನು ನಾಳೆಯಿಂದ ಓದುತ್ತೇನೆ") == "kn-IN"
    # Malayalam
    assert detect_language("ഞാൻ നാളെ മുതൽ പഠിക്കും") == "ml-IN"
    # Bengali
    assert detect_language("আমি কাল থেকে পড়ব ভাই") == "bn-IN"
    # Marathi (with Marathi marker शब्द)
    assert detect_language("मी उद्यापासून अभ्यास करणार आहे भावा") == "mr-IN"
    # Gujarati
    assert detect_language("હું કાલથી વાંચીશ ભાઈ") == "gu-IN"
    # Punjabi
    assert detect_language("ਮੈਂ ਕੱਲ੍ਹ ਤੋਂ ਪੜ੍ਹਾਂਗਾ ਬਾਈ") == "pa-IN"
    # English
    assert detect_language("I am going to study tomorrow bro") == "en-IN"


def test_codemixed_indian_language_detection():
    # Telugu-English (Tenglish)
    assert detect_language("Naku exam lo fail ayya bro, chala bore kodtundi") == "te-IN"
    # Hindi-English (Hinglish)
    assert detect_language("Bhai aaj padhai karne ka bilkul mann nahi hai") == "hi-IN"
    # Tamil-English (Tanglish)
    assert detect_language("Enna machan romba bore adikuthu pa") == "ta-IN"


# ==============================================================================
# 3. Gemma Multilingual Prompt & Response Generation Tests
# ==============================================================================
def test_all_10_languages_intelligence_turns():
    """Verify that IntelligenceEngine routes and responds in all 10 languages."""
    async def _run():
        mock_gemini = MagicMock()
        mock_gemini.model = "gemma-4-26b-a4b-it"

        language_samples = [
            ("te-IN", "నేను రేపటి నుంచి చదువుతాను.", "రేపు ఆల్రెడీ నిన్ను బ్లాక్ చేసేసింది బ్రో."),
            ("hi-IN", "मैं कल से पढूंगा भाई.", "कल ने तुझे पहले ही ब्लॉक कर दिया भाई."),
            ("ta-IN", "நான் நாளைக்கு படிக்க போறேன்.", "தலைவா, நாளைக்கு உன்னை ஆல்ரெடி பிளாக் பண்ணிடுச்சு."),
            ("kn-IN", "ನಾನು ನಾಳೆಯಿಂದ ಓದುತ್ತೇನೆ.", "ಬ್ರೋ, ನಾಳೆ ಆಲ್ರೆಡಿ ನಿನ್ನನ್ನು ಬ್ಲಾಕ್ ಮಾಡಿದೆ."),
            ("ml-IN", "ഞാൻ നാളെ മുതൽ പഠിക്കും.", "ബ്രോ, നാളെ നിന്നെ ഓൾറെഡി ബ്ലോക്ക് ചെയ്തു."),
            ("bn-IN", "আমি কাল থেকে পড়ব ভাই.", "ভাই, কাল তোকে আগেই ব্লক করে দিয়েছে."),
            ("mr-IN", "मी उद्यापासून अभ्यास करणार आहे.", "भावा, उद्याने तुला आधीच ब्लॉक केलंय."),
            ("gu-IN", "હું કાલથી વાંચીશ ભાઈ.", "ભાઈ, કાલે તને પહેલેથી બ્લોક કરી દીધો છે."),
            ("pa-IN", "ਮੈਂ ਕੱਲ੍ਹ ਤੋਂ ਪੜ੍ਹਾਂਗਾ ਬਾਈ.", "ਬਾਈ, ਕੱਲ੍ਹ ਨੇ ਤੈਨੂੰ ਪਹਿਲਾਂ ਹੀ ਬਲਾਕ ਕਰਤਾ।"),
            ("en-IN", "I am going to study tomorrow.", "Bro, tomorrow has blocked you already."),
        ]

        conv_manager = ConversationManager()
        personality_manager = PersonalityManager()

        engine = IntelligenceEngine(
            gemini_service=mock_gemini,
            personality_manager=personality_manager,
            conversation_manager=conv_manager,
        )

        for lang_code, user_msg, mock_reply in language_samples:
            mock_gemini.generate_response = AsyncMock(return_value=mock_reply)

            result = await engine.process_turn(
                message=user_msg,
                conversation_id=f"test_conv_{lang_code}",
                language=lang_code,
            )

            assert result.success is True
            assert result.language == lang_code
            assert result.assistant_response == mock_reply
            assert result.user_message == user_msg

    asyncio.run(_run())


def test_codemixed_turns_generation():
    """Verify code-mixed Hinglish and Tenglish turns."""
    async def _run():
        mock_gemini = MagicMock()
        mock_gemini.model = "gemma-4-26b-a4b-it"

        engine = IntelligenceEngine(
            gemini_service=mock_gemini,
            personality_manager=PersonalityManager(),
            conversation_manager=ConversationManager(),
        )

        # 1. Telugu-English mixed
        mock_gemini.generate_response = AsyncMock(return_value="Question paper tho fight cheste alage untadi bro.")
        res_te = await engine.process_turn(
            message="Naku exam lo fail ayya bro",
            conversation_id="conv_mixed_te",
            language="indic-auto",
        )
        assert res_te.language == "te-IN"
        assert "bro" in res_te.assistant_response.lower()

        # 2. Hindi-English mixed
        mock_gemini.generate_response = AsyncMock(return_value="Tera mann to kabhi padhai ke liye bana hi nahi tha.")
        res_hi = await engine.process_turn(
            message="Bhai aaj padhai karne ka mann nahi hai",
            conversation_id="conv_mixed_hi",
            language="indic-auto",
        )
        assert res_hi.language == "hi-IN"

    asyncio.run(_run())


def test_language_switching_consecutive_turns():
    """Test switching languages across turns in the same conversation session."""
    async def _run():
        mock_gemini = MagicMock()
        mock_gemini.model = "gemma-4-26b-a4b-it"
        conv_mgr = ConversationManager()

        engine = IntelligenceEngine(
            gemini_service=mock_gemini,
            personality_manager=PersonalityManager(),
            conversation_manager=conv_mgr,
        )
        cid = "switch_session_1"

        # Turn 1: Telugu
        mock_gemini.generate_response = AsyncMock(return_value="రేపు ఆల్రెడీ నిన్ను బ్లాక్ చేసేసింది బ్రో.")
        turn1 = await engine.process_turn(
            message="నేను రేపటి నుంచి చదువుతాను.",
            conversation_id=cid,
            language="te-IN",
        )
        assert turn1.language == "te-IN"

        # Turn 2: Switch to English
        mock_gemini.generate_response = AsyncMock(return_value="Even boredom got tired of you.")
        turn2 = await engine.process_turn(
            message="Okay, I am bored now.",
            conversation_id=cid,
            language="en-IN",
        )
        assert turn2.language == "en-IN"

        # Turn 3: Switch to Hindi
        mock_gemini.generate_response = AsyncMock(return_value="भाई, कल ने तुझे पहले ही ब्लॉक कर दिया।")
        turn3 = await engine.process_turn(
            message="अच्छा भाई, अब मैं सोने जा रहा हूँ।",
            conversation_id=cid,
            language="hi-IN",
        )
        assert turn3.language == "hi-IN"

        # Verify session history preserves exact language per turn
        session = conv_mgr.get_conversation(cid)
        assert len(session.messages) == 6
        assert session.messages[0].language == "te-IN"
        assert session.messages[1].language == "te-IN"
        assert session.messages[2].language == "en-IN"
        assert session.messages[3].language == "en-IN"
        assert session.messages[4].language == "hi-IN"
        assert session.messages[5].language == "hi-IN"

    asyncio.run(_run())


# ==============================================================================
# 4. Gnani Timbre TTS Multilingual Voice & Model Resolution Tests
# ==============================================================================
def test_timbre_voice_resolution_yashvi_zayan():
    """Verify Yashvi and Zayan compatibility across all 10 languages on timbre-v2.5."""
    for lang in ["te-IN", "hi-IN", "ta-IN", "kn-IN", "ml-IN", "bn-IN", "mr-IN", "gu-IN", "pa-IN", "en-IN"]:
        # Yashvi
        v, m = resolve_tts_voice_and_model("Yashvi", lang)
        assert v == "Yashvi"
        assert m == "timbre-v2.5"

        # Zayan
        v2, m2 = resolve_tts_voice_and_model("Zayan", lang)
        assert v2 == "Zayan"
        assert m2 == "timbre-v2.5"


def test_timbre_voice_resolution_v3_voices():
    """Verify that official v3 voices map to timbre-v3.0."""
    v, m = resolve_tts_voice_and_model("Shivani", "te-IN")
    assert v == "Shivani"
    assert m == "timbre-v3.0"

    v2, m2 = resolve_tts_voice_and_model("Avinash", "hi-IN")
    assert v2 == "Avinash"
    assert m2 == "timbre-v3.0"


def test_timbre_synthesis_multilingual_request():
    """Verify TimbreService sends correct language code and voice in request payload."""
    service = TimbreService()
    # 64-byte valid RIFF WAVE buffer
    mock_audio_bytes = b"RIFF" + b"\x34\x00\x00\x00" + b"WAVEfmt " + b"\x10\x00\x00\x00" + b"\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00" + b"data" + b"\x10\x00\x00\x00" + b"\x00" * 16

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = mock_audio_bytes
    mock_resp.headers = {"Content-Type": "audio/wav"}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        # 1. Telugu request with Yashvi
        res_te = service.synthesize_speech(
            text="రేపు ఆల్రెడీ నిన్ను బ్లాక్ చేసేసింది బ్రో.",
            language="te-IN",
            voice="Yashvi",
        )
        assert res_te.voice == "Yashvi"
        assert res_te.language == "te-IN"
        called_payload = mock_post.call_args[1]["json"]
        assert called_payload["language"] == "te-IN"
        assert called_payload["voice"] == "Yashvi"
        assert called_payload["model"] == "timbre-v2.5"

        # 2. Hindi request with Zayan
        res_hi = service.synthesize_speech(
            text="भाई, कल ने तुझे पहले ही ब्लॉक कर दिया।",
            language="hi-IN",
            voice="Zayan",
        )
        assert res_hi.voice == "Zayan"
        assert res_hi.language == "hi-IN"
        called_payload_hi = mock_post.call_args[1]["json"]
        assert called_payload_hi["language"] == "hi-IN"
        assert called_payload_hi["voice"] == "Zayan"
        assert called_payload_hi["model"] == "timbre-v2.5"


# ==============================================================================
# 5. Gnani Prisma STT Multilingual & Auto-Detection Tests
# ==============================================================================
def test_prisma_rest_indic_auto_language_resolution():
    """Verify Prisma REST returns accurate detected language."""
    service = PrismaService()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "transcript": "నేను రేపటి నుంచి చదువుతాను",
        "detected_language_code": "te-IN",
        "request_id": "req_te_123",
    }

    with patch("requests.post", return_value=mock_resp):
        res = service.transcribe_audio(
            audio_bytes=b"RIFF" + b"\x00" * 40,
            language_code="indic-auto",
        )
        assert res.success is True
        assert res.language_code == "te-IN"
        assert res.transcript == "నేను రేపటి నుంచి చదువుతాను"


def test_prisma_stream_headers_indic_auto():
    """Verify WebSocket streaming headers configure indic-auto."""
    stream_service = PrismaStreamService()
    headers = stream_service.build_headers("indic-auto")
    assert headers["language_code"] == "indic-auto"
    assert headers["model"] == "prisma-v3.0"
    assert headers["x-partials"] == "true"
