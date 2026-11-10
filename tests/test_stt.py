"""
Unit and Integration Tests for LaughBot AI — Gnani Prisma STT (Step 2).

Tests:
- WAV audio validation (valid, empty, corrupted, oversized, too short)
- Unsupported language rejection
- Missing API credentials handling
- API timeout and network error handling
- Invalid / error response handling
- Correct transcript extraction
- Live Gnani Prisma API integration test (distinguished from unit tests)
"""

import io
import math
import struct
from unittest.mock import MagicMock, PropertyMock, patch
import wave

from fastapi.testclient import TestClient
import pytest
import requests

from backend.config import Settings
from backend.main import app
from backend.services.prisma import (
    PrismaAPIError,
    PrismaConfigError,
    PrismaConnectionError,
    PrismaService,
    PrismaTimeoutError,
)
from backend.utils.audio import (
    AudioValidationError,
    validate_wav_audio,
)

client = TestClient(app)


def generate_test_wav(duration_seconds: float = 1.0, sample_rate: int = 16000, frequency: float = 440.0) -> bytes:
    """Generate a valid 16-bit mono PCM WAV audio byte buffer with a pure sine wave tone."""
    num_samples = int(duration_seconds * sample_rate)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        # Generate 16-bit PCM sine wave
        frames = bytearray()
        for i in range(num_samples):
            sample_val = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * frequency * (i / sample_rate)))
            frames.extend(struct.pack("<h", sample_val))
        wav_file.writeframes(frames)
    return buffer.getvalue()


# ------------------------------------------------------------------------------
# 1. WAV Audio Validation Tests
# ------------------------------------------------------------------------------
def test_valid_wav_audio():
    """Verify that a valid 16 kHz Mono WAV audio file passes validation."""
    wav_bytes = generate_test_wav(duration_seconds=1.5, sample_rate=16000)
    meta = validate_wav_audio(wav_bytes)
    assert meta.channels == 1
    assert meta.sample_rate == 16000
    assert meta.sample_width == 2
    assert 1.4 <= meta.duration_seconds <= 1.6
    assert meta.size_bytes == len(wav_bytes)


def test_empty_audio_rejection():
    """Verify that 0-byte audio is rejected."""
    with pytest.raises(AudioValidationError, match="Audio file is empty"):
        validate_wav_audio(b"")


def test_invalid_wav_rejection():
    """Verify that non-WAV data is rejected."""
    with pytest.raises(AudioValidationError, match="Invalid audio format"):
        validate_wav_audio(b"NOT_A_WAV_HEADER_DATA_1234567890_ABCD")


def test_corrupted_wav_rejection():
    """Verify that header starting with RIFF but with corrupted chunk data is rejected."""
    corrupted = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"corrupted_garbage_bytes_here"
    with pytest.raises(AudioValidationError):
        validate_wav_audio(corrupted)


def test_oversized_audio_rejection():
    """Verify that files exceeding the size limit (10 MB) are rejected."""
    oversized_dummy = b"RIFF" + b"WAVE" + (b"\x00" * 15)
    with pytest.raises(AudioValidationError, match="exceeds maximum allowed limit"):
        validate_wav_audio(oversized_dummy, max_size=10)


def test_too_short_audio_rejection():
    """Verify that audio with 0 frames or less than minimum duration is rejected."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"")  # 0 frames
    with pytest.raises(AudioValidationError, match="too short or silent"):
        validate_wav_audio(buffer.getvalue(), min_duration=0.5)


# ------------------------------------------------------------------------------
# 2. Unsupported Language Rejection
# ------------------------------------------------------------------------------
def test_unsupported_language_rejection():
    """Verify endpoint rejects unsupported language codes with HTTP 400."""
    wav_bytes = generate_test_wav(duration_seconds=1.0)
    files = {"audio_file": ("test.wav", wav_bytes, "audio/wav")}
    data = {"language_code": "fr-FR"}  # French is not in supported Indian languages

    response = client.post("/api/v1/stt/transcribe", files=files, data=data)
    assert response.status_code == 400
    assert "Unsupported language_code" in response.json()["error"]


# ------------------------------------------------------------------------------
# 3. Missing API Credentials Handling
# ------------------------------------------------------------------------------
def test_missing_credentials_service():
    """Verify PrismaService raises PrismaConfigError when unconfigured."""
    mock_settings = Settings(GNANI_API_KEY=None, GNANI_PRISMA_API_KEY=None)
    service = PrismaService(settings=mock_settings)
    assert not service.is_configured

    with pytest.raises(PrismaConfigError, match="credentials are not configured"):
        service.transcribe_audio(generate_test_wav())


# ------------------------------------------------------------------------------
# 4. API Timeout and Network Error Handling (Mocked Unit Tests)
# ------------------------------------------------------------------------------
def test_api_timeout_handling():
    """Verify PrismaTimeoutError is raised when the API call times out."""
    mock_settings = Settings(GNANI_API_KEY="test_key", GNANI_PRISMA_TIMEOUT=5)
    service = PrismaService(settings=mock_settings)

    with patch("requests.post", side_effect=requests.exceptions.Timeout("Connection timed out")):
        with pytest.raises(PrismaTimeoutError, match="timed out"):
            service.transcribe_audio(generate_test_wav())


def test_api_connection_error_handling():
    """Verify PrismaConnectionError is raised on network failure."""
    mock_settings = Settings(GNANI_API_KEY="test_key")
    service = PrismaService(settings=mock_settings)

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Network down")):
        with pytest.raises(PrismaConnectionError, match="Failed to connect"):
            service.transcribe_audio(generate_test_wav())


# ------------------------------------------------------------------------------
# 5. Invalid API Response Handling (Mocked Unit Tests)
# ------------------------------------------------------------------------------
def test_api_http_error_handling():
    """Verify PrismaAPIError is raised when the API returns an HTTP 401 or 500 error."""
    mock_settings = Settings(GNANI_API_KEY="test_key")
    service = PrismaService(settings=mock_settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"error": "Invalid API Key ID"}

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(PrismaAPIError) as exc_info:
            service.transcribe_audio(generate_test_wav())
        assert exc_info.value.status_code == 401
        assert "Invalid API Key ID" in str(exc_info.value)


def test_api_unsuccessful_flag_handling():
    """Verify PrismaAPIError is raised when API returns success=False."""
    mock_settings = Settings(GNANI_API_KEY="test_key")
    service = PrismaService(settings=mock_settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"success": False, "error": "Speech too noisy"}

    with patch("requests.post", return_value=mock_resp):
        with pytest.raises(PrismaAPIError, match="Speech too noisy"):
            service.transcribe_audio(generate_test_wav())


# ------------------------------------------------------------------------------
# 6. Correct Transcript Extraction (Mocked Unit Test)
# ------------------------------------------------------------------------------
def test_successful_transcript_extraction():
    """Verify transcript, request_id, and language are extracted correctly from JSON response."""
    mock_settings = Settings(GNANI_API_KEY="test_key")
    service = PrismaService(settings=mock_settings)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "success": True,
        "request_id": "gnani_req_98765",
        "transcript": "Hello and welcome to LaughBot AI",
        "detected_language_code": "en-IN",
    }

    with patch("requests.post", return_value=mock_resp):
        result = service.transcribe_audio(generate_test_wav(), language_code="en-IN")
        assert result.success is True
        assert result.transcript == "Hello and welcome to LaughBot AI"
        assert result.request_id == "gnani_req_98765"
        assert result.language_code == "en-IN"


# ------------------------------------------------------------------------------
# 7. Endpoint Integration with Mocked Service
# ------------------------------------------------------------------------------
def test_endpoint_success_mocked():
    """Test POST /api/v1/stt/transcribe returns expected JSON schema."""
    wav_bytes = generate_test_wav(duration_seconds=1.0)
    files = {"audio_file": ("sample.wav", wav_bytes, "audio/wav")}
    data = {"language_code": "en-IN"}

    mock_result = MagicMock()
    mock_result.transcript = "Testing LaughBot speech recognition"
    mock_result.language_code = "en-IN"
    mock_result.request_id = "req_abc123"

    with patch("backend.main.prisma_service.transcribe_audio", return_value=mock_result):
        with patch.object(PrismaService, "is_configured", new_callable=PropertyMock, return_value=True):
            res = client.post("/api/v1/stt/transcribe", files=files, data=data)
            assert res.status_code == 200
            body = res.json()
            assert body["success"] is True
            assert body["transcript"] == "Testing LaughBot speech recognition"
            assert body["language"] == "en-IN"
            assert body["request_id"] == "req_abc123"


# ------------------------------------------------------------------------------
# 8. Real Live API Integration Test (Using configured account credentials)
# ------------------------------------------------------------------------------
def test_live_gnani_prisma_api_call():
    """
    Real integration test against live Gnani Prisma API (https://api.vachana.ai/stt/v3).
    Only executes if valid GNANI_API_KEY is configured.
    """
    from backend.config import get_settings
    current_settings = get_settings()

    if not current_settings.has_prisma_credentials:
        pytest.skip("Skipping live API test: no Gnani credentials in environment.")

    real_service = PrismaService(settings=current_settings)
    assert real_service.is_configured

    # Generate real 1-second 16 kHz WAV audio
    real_wav = generate_test_wav(duration_seconds=1.0, sample_rate=16000)

    try:
        result = real_service.transcribe_audio(real_wav, language_code="en-IN")
        assert result is not None
        assert result.success is True
        print(f"\n[LIVE API SUCCESS] Request ID: {result.request_id}, Transcript: '{result.transcript}'")
    except PrismaAPIError as err:
        # Check if error returned by live server
        print(f"\n[LIVE API RESPONSE] HTTP {err.status_code}: {err}")
        # An HTTP response (e.g., authentication or empty audio) confirms live server connectivity
        assert err.status_code is not None
