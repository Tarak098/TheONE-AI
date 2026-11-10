"""
Unit and Integration Tests for LaughBot AI — Step 4: Gnani Timbre Text-to-Speech (TTS) Integration.

Tests:
1. Timbre Configuration Loading & Model Normalization (timbre-2.5 -> timbre-v2.5)
2. Missing Credentials Handling (TimbreServiceUnavailableError)
3. Voice Selection for Modes (Always Funny -> Yashvi, Roast Companion -> Zayan)
4. Input Validation (Empty Text & Oversized Text Rejection)
5. Request Timeout Handling (Mocked Timeout -> TimbreTimeoutError)
6. Rate Limit Handling (Mocked HTTP 429 -> TimbreRateLimitError)
7. Upstream API Error Handling (Mocked HTTP 500 -> TimbreAPIError)
8. Invalid Audio Format Handling (Mocked Corrupted Stream -> TimbreInvalidAudioError)
9. Successful Synthesis & Audio Encoding (Mocked WAV Generation)
10. FastAPI POST /api/v1/tts Endpoint (Mocked)
11. FastAPI GET /api/v1/tts/health Endpoint
12. API Status Endpoint reporting Timbre readiness
13. Live Gnani Timbre Standalone API Verification (Real Credentials)
"""

import io
from unittest.mock import MagicMock, PropertyMock, patch
import wave
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings, get_settings
from backend.main import app
from backend.services.timbre import (
    TimbreAPIError,
    TimbreConfigError,
    TimbreInvalidAudioError,
    TimbreRateLimitError,
    TimbreService,
    TimbreServiceUnavailableError,
    TimbreSynthesisResult,
    TimbreTimeoutError,
    TimbreValidationError,
)

client = TestClient(app)

# Helper to create a synthetic valid WAV byte sequence for mock tests
def create_mock_wav_bytes(num_frames: int = 1000, sample_rate: int = 48000) -> bytes:
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames)
    return wav_io.getvalue()


# ------------------------------------------------------------------------------
# 1. Configuration & Voice Selection Tests
# ------------------------------------------------------------------------------
def test_timbre_config_loading():
    """Verify Timbre settings are loaded and model is normalized."""
    settings = get_settings()
    assert settings.GNANI_TTS_LANGUAGE == "en-IN"
    assert settings.normalized_tts_model == "timbre-v2.5"
    assert settings.GNANI_TTS_SAMPLE_RATE == 48000
    assert settings.GNANI_TTS_VOICE_FUNNY == "Yashvi"
    assert settings.GNANI_TTS_VOICE_ROAST == "Zayan"
    assert settings.GNANI_TIMBRE_TIMEOUT >= 1


def test_timbre_unconfigured_credentials_raises_unavailable():
    """Verify TimbreServiceUnavailableError is raised when credentials are missing."""
    mock_settings = Settings(GNANI_TIMBRE_API_KEY=None, GNANI_API_KEY=None)
    service = TimbreService(settings=mock_settings)

    assert not service.is_configured
    health = service.health_check()
    assert health["configured"] is False
    assert health["status"] == "not_configured"

    with pytest.raises(TimbreServiceUnavailableError, match="not configured"):
        service.validate_configuration()

    with pytest.raises(TimbreServiceUnavailableError, match="not configured"):
        service.synthesize_speech("Test speech")


def test_timbre_voice_selection_by_mode():
    """Verify personality mode determines active comedic voice."""
    service = TimbreService()
    assert service.get_voice_for_mode("always_funny") == "Yashvi"
    assert service.get_voice_for_mode("funny") == "Yashvi"
    assert service.get_voice_for_mode("roast_companion") == "Zayan"
    assert service.get_voice_for_mode("roast") == "Zayan"
    assert service.get_voice_for_mode(None) == service.default_voice


# ------------------------------------------------------------------------------
# 2. Input Validation Tests
# ------------------------------------------------------------------------------
def test_timbre_empty_text_validation():
    """Verify empty or whitespace text raises TimbreValidationError."""
    service = TimbreService()
    if not service.is_configured:
        service._api_key = "test_key"

    with pytest.raises(TimbreValidationError, match="cannot be empty"):
        service.synthesize_speech("")

    with pytest.raises(TimbreValidationError, match="cannot be empty"):
        service.synthesize_speech("   \n\t  ")


def test_timbre_oversized_text_validation():
    """Verify text exceeding character limit raises TimbreValidationError."""
    service = TimbreService()
    if not service.is_configured:
        service._api_key = "test_key"

    oversized = "a" * 5000
    with pytest.raises(TimbreValidationError, match="exceeds maximum limit"):
        service.synthesize_speech(oversized)


# ------------------------------------------------------------------------------
# 3. Mocked Error Handling Tests
# ------------------------------------------------------------------------------
def test_timbre_timeout_handling_mocked():
    """Verify requests timeout raises TimbreTimeoutError."""
    import requests
    service = TimbreService()
    if not service.is_configured:
        service._api_key = "test_key"

    with patch("requests.post", side_effect=requests.exceptions.Timeout("Timeout")):
        with pytest.raises(TimbreTimeoutError, match="timed out"):
            service.synthesize_speech("Test timeout")


def test_timbre_rate_limit_handling_mocked():
    """Verify HTTP 429 raises TimbreRateLimitError."""
    service = TimbreService()
    if not service.is_configured:
        service._api_key = "test_key"

    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.json.return_value = {"message": "Rate limit exceeded"}

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(TimbreRateLimitError, match="quota or rate limit exceeded"):
            service.synthesize_speech("Test rate limit")


def test_timbre_api_error_handling_mocked():
    """Verify HTTP 500 raises TimbreAPIError."""
    service = TimbreService()
    if not service.is_configured:
        service._api_key = "test_key"

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {"message": "Internal server error"}

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(TimbreAPIError, match="Gnani Timbre error"):
            service.synthesize_speech("Test error")


def test_timbre_invalid_audio_handling_mocked():
    """Verify non-WAV or truncated audio raises TimbreInvalidAudioError."""
    service = TimbreService()
    if not service.is_configured:
        service._api_key = "test_key"

    # Truncated content (< 44 bytes)
    mock_resp_trunc = MagicMock()
    mock_resp_trunc.status_code = 200
    mock_resp_trunc.content = b"Too short"

    with patch("requests.post", return_value=mock_resp_trunc):
        with pytest.raises(TimbreInvalidAudioError, match="empty or truncated"):
            service.synthesize_speech("Test truncated audio")

    # Non-WAV content (>= 44 bytes but missing RIFF/WAVE header)
    mock_resp_non_wav = MagicMock()
    mock_resp_non_wav.status_code = 200
    mock_resp_non_wav.content = b"A" * 60

    with patch("requests.post", return_value=mock_resp_non_wav):
        with pytest.raises(TimbreInvalidAudioError, match="invalid non-WAV"):
            service.synthesize_speech("Test invalid audio")


# ------------------------------------------------------------------------------
# 4. Successful Mocked Synthesis Tests
# ------------------------------------------------------------------------------
def test_timbre_successful_synthesis_mocked():
    """Verify successful synthesis produces valid TimbreSynthesisResult with base64 and data URL."""
    service = TimbreService()
    if not service.is_configured:
        service._api_key = "test_key"

    mock_wav = create_mock_wav_bytes(num_frames=2000, sample_rate=48000)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = mock_wav

    with patch("requests.post", return_value=mock_resp):
        result = service.synthesize_speech(
            text="Why did the AI cross the road?",
            mode="always_funny",
        )

        assert isinstance(result, TimbreSynthesisResult)
        assert result.voice == "Yashvi"
        assert result.format == "audio/wav"
        assert result.sample_rate == 48000
        assert len(result.audio_bytes) == len(mock_wav)
        assert result.data_url.startswith("data:audio/wav;base64,")
        assert len(result.audio_base64) > 0


# ------------------------------------------------------------------------------
# 5. FastAPI Endpoints Tests
# ------------------------------------------------------------------------------
def test_tts_health_endpoint():
    """Verify GET /api/v1/tts/health returns Timbre readiness status."""
    res = client.get("/api/v1/tts/health")
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "Gnani Timbre"
    assert data["model"] == "timbre-v2.5"
    assert "default_voice" in data
    assert "voice_funny" in data
    assert "voice_roast" in data


def test_tts_endpoint_success_mocked():
    """Verify POST /api/v1/tts returns 200 with audio payload."""
    mock_wav = create_mock_wav_bytes()
    mock_result = TimbreSynthesisResult(
        audio_bytes=mock_wav,
        voice="Yashvi",
        model="timbre-v2.5",
        language="en-IN",
        sample_rate=48000,
        format="audio/wav",
    )

    with patch.object(TimbreService, "is_configured", new_callable=PropertyMock, return_value=True):
        with patch.object(TimbreService, "async_synthesize_speech", return_value=mock_result):
            res = client.post(
                "/api/v1/tts",
                json={
                    "text": "Hello! Welcome to LaughBot. Your daily dose of laughter starts now!",
                    "mode": "always_funny",
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["status"] == "completed"
            assert data["voice"] == "Yashvi"
            assert data["format"] == "audio/wav"
            assert data["model"] == "timbre-v2.5"
            assert len(data["audio_base64"]) > 0
            assert data["audio_url"].startswith("data:audio/wav;base64,")


def test_tts_endpoint_unconfigured_503():
    """Verify POST /api/v1/tts returns 503 when credentials are not configured."""
    with patch.object(TimbreService, "is_configured", new_callable=PropertyMock, return_value=False):
        res = client.post(
            "/api/v1/tts",
            json={"text": "Hello", "mode": "always_funny"},
        )
        assert res.status_code == 503
        assert "not configured" in res.json()["error"].lower()


def test_api_status_reports_timbre():
    """Verify GET /api/v1/status reports Timbre status correctly."""
    res = client.get("/api/v1/status")
    assert res.status_code == 200
    data = res.json()
    assert "timbre" in data["services"]


def test_timbre_voice_validation_supported_and_unsupported():
    """Verify validate_voice accepts supported voices and rejects untrusted inputs."""
    service = TimbreService()
    # Case-insensitive valid voices
    assert service.validate_voice("Yashvi") == "Yashvi"
    assert service.validate_voice("yashvi") == "Yashvi"
    assert service.validate_voice("Zayan") == "Zayan"
    assert service.validate_voice("zayan") == "Zayan"

    # Unsupported or empty voices raise TimbreValidationError
    with pytest.raises(TimbreValidationError, match="Unsupported voice"):
        service.validate_voice("UnknownVoice")

    with pytest.raises(TimbreValidationError, match="Unsupported voice"):
        service.validate_voice("Alexa")

    with pytest.raises(TimbreValidationError, match="cannot be empty"):
        service.validate_voice("")


def test_tts_voices_catalog_endpoint():
    """Verify GET /api/v1/tts/voices returns allowed voices and metadata."""
    res = client.get("/api/v1/tts/voices")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "Yashvi" in data["voices"]
    assert "Zayan" in data["voices"]
    assert data["voices"]["Yashvi"]["gender"] == "Female"
    assert data["voices"]["Zayan"]["gender"] == "Male"


def test_tts_endpoint_rejects_untrusted_voice():
    """Verify POST /api/v1/tts returns 400 when an untrusted voice name is supplied."""
    with patch.object(TimbreService, "is_configured", new_callable=PropertyMock, return_value=True):
        res = client.post(
            "/api/v1/tts",
            json={
                "text": "Hello world",
                "mode": "always_funny",
                "voice": "HackerVoice123",
            },
        )
        assert res.status_code == 400
        assert "unsupported voice" in res.json()["error"].lower()


@pytest.mark.parametrize("mode,voice", [
    ("always_funny", "Yashvi"),
    ("always_funny", "Zayan"),
    ("roast_companion", "Yashvi"),
    ("roast_companion", "Zayan"),
])
def test_all_four_voice_combinations_mocked(mode, voice):
    """Verify all 4 combinations (Funny/Roast x Yashvi/Zayan) pass selected voice cleanly."""
    mock_wav = create_mock_wav_bytes()
    mock_result = TimbreSynthesisResult(
        audio_bytes=mock_wav,
        voice=voice,
        model="timbre-v2.5",
        language="en-IN",
        sample_rate=48000,
        format="audio/wav",
    )

    with patch.object(TimbreService, "is_configured", new_callable=PropertyMock, return_value=True):
        with patch.object(TimbreService, "async_synthesize_speech", return_value=mock_result) as mock_synth:
            res = client.post(
                "/api/v1/tts",
                json={
                    "text": "Your brain has officially entered power-saving mode.",
                    "mode": mode,
                    "voice": voice,
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["voice"] == voice
            # Verify the service received the explicit voice selection
            mock_synth.assert_called_once()
            call_kwargs = mock_synth.call_args.kwargs
            assert call_kwargs["voice"] == voice
            assert call_kwargs["mode"] == mode


def test_live_all_four_voice_combinations():
    """
    Live test against Gnani Timbre API testing all four combinations:
    1. Funny + Yashvi (Female)
    2. Funny + Zayan (Male)
    3. Roast + Yashvi (Female)
    4. Roast + Zayan (Male)
    """
    service = TimbreService()
    if not service.is_configured:
        pytest.skip("Gnani API credentials not configured in environment.")

    combinations = [
        ("always_funny", "Yashvi", "Rise and shine, you professional snooze button addict."),
        ("always_funny", "Zayan", "Your internal clock just filed for divorce."),
        ("roast_companion", "Yashvi", "Your closet is a crime scene waiting for a trial."),
        ("roast_companion", "Zayan", "Tomorrow called; it blocked your number."),
    ]

    for mode, voice, text in combinations:
        result = service.synthesize_speech(
            text=text,
            mode=mode,
            voice=voice,
        )
        assert result is not None
        assert result.voice == voice
        assert result.model == "timbre-v2.5"
        assert result.format == "audio/wav"
        assert len(result.audio_bytes) > 1000
        assert result.audio_bytes.startswith(b"RIFF")
        assert b"WAVE" in result.audio_bytes[:16]
        print(f"\n[LIVE TTS COMBO SUCCESS] Mode: {mode}, Voice: {voice}, Bytes: {len(result.audio_bytes)}")

def test_live_gnani_timbre_standalone_call():
    """
    Test live speech synthesis against the Gnani Timbre API using real credentials.
    Uses documented sentence: 'Hello! Welcome to LaughBot. Your daily dose of laughter starts now!'
    Verifies HTTP 200, valid RIFF WAVE container, and sample rate.
    """
    service = TimbreService()
    if not service.is_configured:
        pytest.skip("Gnani API credentials not configured in environment.")

    text = "Hello! Welcome to LaughBot. Your daily dose of laughter starts now!"
    result = service.synthesize_speech(
        text=text,
        mode="always_funny",
    )

    assert result is not None
    assert isinstance(result.audio_bytes, bytes)
    assert len(result.audio_bytes) > 1000
    assert result.audio_bytes.startswith(b"RIFF")
    assert b"WAVE" in result.audio_bytes[:16]
    assert result.voice == "Yashvi"
    assert result.model == "timbre-v2.5"

    # Verify WAV audio validity via standard wave module
    with wave.open(io.BytesIO(result.audio_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 48000
        assert wf.getnframes() > 0

    print(
        f"\n[LIVE TIMBRE TTS SUCCESS] Voice: {result.voice}, "
        f"AudioBytes: {len(result.audio_bytes)}, Base64Length: {len(result.audio_base64)}"
    )

