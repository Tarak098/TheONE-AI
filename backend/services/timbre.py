"""
LaughBot AI - Gnani Timbre Text-to-Speech (TTS) Service Adapter.

Provides the service integration for Gnani Timbre (timbre-v2.5) to synthesize
comedic and roast responses into high-quality speech audio across Indian English
and regional languages.
"""

import asyncio
import base64
from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, Optional, Set

import requests

from backend.config import Settings, get_settings

logger = logging.getLogger("laughbot.timbre")

# Maximum permitted character length for a single speech synthesis request
MAX_TTS_TEXT_LENGTH = 4000

# Supported voices for Gnani Timbre (timbre-v2.5)
SUPPORTED_VOICES: Dict[str, Dict[str, str]] = {
    "Yashvi": {
        "gender": "Female",
        "model": "timbre-v2.5",
        "description": "Energetic, expressive, natural comedic timing",
    },
    "Zayan": {
        "gender": "Male",
        "model": "timbre-v2.5",
        "description": "Crisp, sarcastic, deadpan expressive delivery",
    },
}

# Voices officially supported by Gnani Timbre v3.0 (timbre-v3.0)
SUPPORTED_V3_VOICES: Set[str] = {
    "Avinash", "Shivani", "Lakshmi", "Kalpana", "Shubham", "Prakash",
    "Devansh", "Yashraj", "Chetana", "Vasudha", "Harshad", "Riyaa", "Tarika",
}

# Official audio tags supported ONLY on Timbre v3.0
SUPPORTED_V3_AUDIO_TAGS: Set[str] = {
    "<laugh>", "<giggle>", "<sigh>", "<exhales>", "<sniffles>",
    "<yawns>", "<coughs>", "<clears throat>", "<whispers>",
    "<shouting>", "<excited>", "<slow>", "<pauses>",
}


def sanitize_tts_text(text: str, allow_tags: bool = False) -> str:
    """
    Sanitize text for natural, expressive TTS synthesis.

    - Strips angle-bracket tags if allow_tags is False so they are NEVER spoken aloud.
    - Preserves ellipses (...) and punctuation for natural pauses and pitch inflection.
    - Strips markdown formatting (*, _, #, `) that could degrade neural speech.
    - Normalizes spacing.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # If tags are not allowed by the model, strip all <...> tags cleanly
    if not allow_tags:
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
    else:
        # If tags are allowed, only keep officially supported v3 tags
        for match in re.findall(r"<[^>]+>", cleaned):
            if match.lower() not in SUPPORTED_V3_AUDIO_TAGS:
                cleaned = cleaned.replace(match, "")

    # Strip markdown emphasis and quotes
    cleaned = re.sub(r"[*_#`~]", "", cleaned)
    cleaned = cleaned.strip("\"'“”` ")

    # Normalize ellipses with space for optimal dramatic pause timing
    cleaned = re.sub(r"\.{3,}", "... ", cleaned)

    # Normalize multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned and not cleaned.endswith((".", "!", "?")):
        cleaned += "."

    return cleaned


class TimbreError(Exception):
    """Base exception for all Gnani Timbre operations."""
    pass


class TimbreConfigError(TimbreError):
    """Raised when Timbre configuration is missing or invalid."""
    pass


class TimbreServiceUnavailableError(TimbreError):
    """Raised when Timbre API credentials are not configured."""
    pass


class TimbreValidationError(TimbreError):
    """Raised when input text or synthesis parameters fail validation."""
    pass


class TimbreTimeoutError(TimbreError):
    """Raised when a Timbre TTS inference request times out."""
    pass


class TimbreRateLimitError(TimbreError):
    """Raised when Timbre API rate limits or quota are exceeded."""
    pass


class TimbreAPIError(TimbreError):
    """Raised when the Gnani Timbre endpoint returns an error response."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class TimbreInvalidAudioError(TimbreError):
    """Raised when Timbre returns empty or invalid audio data."""
    pass


@dataclass
class TimbreSynthesisResult:
    """Structured result of a speech synthesis operation."""
    audio_bytes: bytes
    voice: str
    model: str
    language: str
    sample_rate: int
    format: str = "audio/wav"

    @property
    def audio_base64(self) -> str:
        """Return base64-encoded audio string."""
        return base64.b64encode(self.audio_bytes).decode("ascii")

    @property
    def data_url(self) -> str:
        """Return standard data URI for instant browser playback."""
        return f"data:{self.format};base64,{self.audio_base64}"

    def to_dict(self, include_audio: bool = True) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        result: Dict[str, Any] = {
            "success": True,
            "voice": self.voice,
            "model": self.model,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "format": self.format,
            "audio_bytes_length": len(self.audio_bytes),
        }
        if include_audio:
            result["audio_base64"] = self.audio_base64
            result["audio_url"] = self.data_url
        return result


class TimbreService:
    """
    Service client for Gnani Timbre Text-to-Speech (timbre-v2.5).

    Supports:
    - Energetic, punchy comedic delivery for Always Funny mode (e.g., Yashvi)
    - Sarcastic, expressive delivery for Roast Companion mode (e.g., Zayan)
    - 48 kHz Linear PCM WAV container generation
    - Comprehensive error handling and exponential backoff
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._api_key = api_key
        self._api_url = api_url

    @property
    def api_key(self) -> Optional[str]:
        """Resolve Timbre API key securely."""
        return self._api_key or self._settings.timbre_api_key

    @property
    def api_url(self) -> str:
        """Endpoint URL for Gnani Timbre TTS."""
        return self._api_url or self._settings.GNANI_TIMBRE_API_URL

    @property
    def model(self) -> str:
        """Normalized Timbre model identifier (timbre-v2.5)."""
        return self._settings.normalized_tts_model

    @property
    def language(self) -> str:
        """Default TTS language code."""
        return self._settings.GNANI_TTS_LANGUAGE

    @property
    def default_voice(self) -> str:
        """Default TTS voice."""
        return self._settings.GNANI_TTS_VOICE

    @property
    def voice_funny(self) -> str:
        """Voice configured for Always Funny mode."""
        return self._settings.GNANI_TTS_VOICE_FUNNY

    @property
    def voice_roast(self) -> str:
        """Voice configured for Roast Companion mode."""
        return self._settings.GNANI_TTS_VOICE_ROAST

    @property
    def sample_rate(self) -> int:
        """Configured output audio sample rate in Hz."""
        return self._settings.GNANI_TTS_SAMPLE_RATE

    @property
    def timeout(self) -> int:
        """Request timeout in seconds."""
        return self._settings.GNANI_TIMBRE_TIMEOUT

    @property
    def is_configured(self) -> bool:
        """Check whether Timbre has credentials configured."""
        return bool(self.api_key and self.api_key.strip())

    def validate_configuration(self) -> bool:
        """
        Validate that Timbre API credentials and URL are configured.

        Raises:
            TimbreServiceUnavailableError: If credentials are missing.
        """
        if not self.is_configured:
            raise TimbreServiceUnavailableError(
                "Gnani Timbre TTS credentials are not configured. "
                "Please configure GNANI_API_KEY or GNANI_TIMBRE_API_KEY in your .env file."
            )
        return True

    def get_voice_for_mode(self, mode: Optional[str] = None) -> str:
        """
        Select an appropriate voice based on personality mode.

        Always Funny -> Yashvi (energetic, friendly, expressive)
        Roast Companion -> Zayan (witty, confident, sarcastic edge)
        """
        if not mode:
            return self.default_voice

        normalized_mode = mode.lower().strip()
        if normalized_mode in ("always_funny", "funny"):
            return self.voice_funny
        elif normalized_mode in ("roast_companion", "roast"):
            return self.voice_roast

        return self.default_voice

    @classmethod
    def get_supported_voices(cls) -> Dict[str, Dict[str, str]]:
        """Return the dictionary of allowed voices and their metadata."""
        return SUPPORTED_VOICES.copy()

    def validate_voice(self, voice: Optional[str]) -> str:
        """
        Validate requested voice against allowed catalog.

        Args:
            voice: Requested voice name (e.g. 'Yashvi', 'Zayan').

        Returns:
            str: Canonical voice name.

        Raises:
            TimbreValidationError: If voice is not in the allowed catalog.
        """
        if not voice or not voice.strip():
            raise TimbreValidationError("Voice name cannot be empty.")

        clean_voice = voice.strip()
        for canonical_name in SUPPORTED_VOICES:
            if canonical_name.lower() == clean_voice.lower():
                return canonical_name

        allowed = ", ".join(SUPPORTED_VOICES.keys())
        raise TimbreValidationError(
            f"Unsupported voice '{voice}'. Allowed voices: {allowed}"
        )

    def health_check(self) -> Dict[str, Any]:
        """Return readiness status without leaking credentials."""
        configured = self.is_configured
        return {
            "service": "Gnani Timbre",
            "model": self.model,
            "configured": configured,
            "endpoint": self.api_url,
            "default_voice": self.default_voice,
            "voice_funny": self.voice_funny,
            "voice_roast": self.voice_roast,
            "supported_voices": list(SUPPORTED_VOICES.keys()),
            "voices_detail": SUPPORTED_VOICES,
            "expressive_tags_supported": False,  # v2.5 with Yashvi/Zayan uses prosody/speed
            "sample_rate": self.sample_rate,
            "status": "ready" if configured else "not_configured",
        }

    def synthesize_speech(
        self,
        text: str,
        mode: Optional[str] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> TimbreSynthesisResult:
        """
        Synchronously synthesize text into WAV speech audio using Gnani Timbre.

        Args:
            text: Text to synthesize into speech.
            mode: Optional personality mode ('always_funny' or 'roast_companion').
            voice: Explicit voice name (overrides mode-based voice).
            language: Language code (defaults to en-IN).
            sample_rate: Audio sample rate in Hz (defaults to 48000).
            speed: Speech speed multiplier (defaults to mode-tailored prosodic speed).

        Returns:
            TimbreSynthesisResult: Contains raw WAV audio bytes, base64 data, and metadata.

        Raises:
            TimbreServiceUnavailableError: If API credentials are not configured.
            TimbreValidationError: If text is empty or invalid.
            TimbreRateLimitError: If API quota or rate limits are exceeded.
            TimbreTimeoutError: If the request times out.
            TimbreAPIError: If the upstream API returns an error.
            TimbreInvalidAudioError: If returned audio is empty or corrupted.
        """
        self.validate_configuration()

        # 1. Validate Input Text
        if not text or not text.strip():
            raise TimbreValidationError("Speech synthesis text cannot be empty or whitespace.")

        # 2. Resolve Parameters (Validate user-selected voice if specified)
        if voice and voice.strip():
            selected_voice = self.validate_voice(voice)
        else:
            selected_voice = self.get_voice_for_mode(mode)

        selected_language = language or self.language
        selected_sample_rate = sample_rate or self.sample_rate

        # 3. Model Compatibility & Audio Tag Gate
        # Yashvi and Zayan are official v2.5 voices (v3.0 rejects them)
        if selected_voice in SUPPORTED_VOICES:
            effective_model = "timbre-v2.5"
            allow_tags = False
        elif selected_voice in SUPPORTED_V3_VOICES and self.model == "timbre-v3.0":
            effective_model = "timbre-v3.0"
            allow_tags = True
        else:
            effective_model = self.model
            allow_tags = False

        # 4. Text Sanitization & Prosody Optimization
        clean_text = sanitize_tts_text(text, allow_tags=allow_tags)
        if not clean_text:
            raise TimbreValidationError("Speech text became empty after sanitization.")

        if len(clean_text) > MAX_TTS_TEXT_LENGTH:
            raise TimbreValidationError(
                f"Text length ({len(clean_text)} characters) exceeds maximum limit of {MAX_TTS_TEXT_LENGTH}."
            )

        # 5. Mode-Tailored Expressive Speed
        if speed is not None:
            selected_speed = speed
        elif mode in ("always_funny", "funny") or selected_voice == "Yashvi":
            selected_speed = 1.04  # Upbeat, cheerful, lively comedic delivery
        elif mode in ("roast_companion", "roast") or selected_voice == "Zayan":
            selected_speed = 0.98  # Confident, deadpan, deliberate sarcastic delivery
        else:
            selected_speed = 1.0

        # Defensive sanitization of API key to prevent header syntax errors
        clean_key = (self.api_key or "").strip().strip("\r\n\t\"'")
        if "=" in clean_key:
            prefix, _, after = clean_key.partition("=")
            if prefix.strip().replace("_", "").isalnum():
                clean_key = after.strip().strip("\r\n\t\"'")

        headers = {
            "Authorization": f"Bearer {clean_key}",
            "X-API-Key-ID": clean_key,
            "Content-Type": "application/json",
        }

        payload = {
            "model": effective_model,
            "language": selected_language,
            "voice": selected_voice,
            "speed": selected_speed,
            "text": clean_text,
            "audio_config": {
                "encoding": "linear_pcm",
                "container": "wav",
                "num_channels": 1,
                "sample_rate": selected_sample_rate,
                "sample_width": 2,
            },
        }

        logger.info(
            "Calling Gnani Timbre TTS (Model: %s, Voice: %s, Speed: %.2f, Lang: %s, TextLength: %d)",
            effective_model,
            selected_voice,
            selected_speed,
            selected_language,
            len(clean_text),
        )

        max_attempts = 3
        backoff_delay = 1.0
        response = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                break
            except requests.exceptions.Timeout as err:
                if attempt < max_attempts:
                    logger.warning("Timbre TTS request timed out (attempt %d/%d). Retrying...", attempt, max_attempts)
                    import time
                    time.sleep(backoff_delay)
                    backoff_delay *= 2.0
                    continue
                logger.error("Timbre TTS inference request timed out after %ds", self.timeout)
                raise TimbreTimeoutError(f"Gnani Timbre TTS request timed out after {self.timeout}s.") from err
            except requests.exceptions.RequestException as err:
                if attempt < max_attempts:
                    logger.warning("Transient network error calling Timbre. Retrying...")
                    import time
                    time.sleep(backoff_delay)
                    backoff_delay *= 2.0
                    continue
                logger.error("Network error communicating with Timbre TTS: %s", err)
                err_msg = str(err)
                if clean_key and clean_key in err_msg:
                    err_msg = err_msg.replace(clean_key, "[REDACTED]")
                if "header value" in err_msg.lower():
                    err_msg = "Invalid API key format or header characters. Check GNANI_API_KEY environment variable."
                raise TimbreAPIError(f"Network error communicating with Gnani Timbre: {err_msg}") from err

        if response is None:
            raise TimbreAPIError("Failed to obtain response from Gnani Timbre.")

        status_code = response.status_code

        # Handle API Error Responses
        if status_code != 200:
            err_msg = f"Timbre error (HTTP {status_code})"
            try:
                err_data = response.json()
                err_msg = err_data.get("message") or err_data.get("error") or err_msg
            except Exception:
                err_msg = response.text[:200] if response.text else err_msg

            logger.error("Timbre TTS API error (HTTP %d): %s", status_code, err_msg)

            if status_code == 429:
                raise TimbreRateLimitError(f"Gnani Timbre quota or rate limit exceeded: {err_msg}")
            if status_code in (400, 422):
                raise TimbreValidationError(f"Invalid request parameters for Timbre: {err_msg}")
            if status_code in (401, 403):
                raise TimbreAPIError(f"Authentication error with Gnani Timbre (HTTP {status_code}): {err_msg}", status_code=status_code)

            raise TimbreAPIError(f"Gnani Timbre error (HTTP {status_code}): {err_msg}", status_code=status_code)

        # Validate Audio Content
        audio_bytes = response.content
        if not audio_bytes or len(audio_bytes) < 44:
            raise TimbreInvalidAudioError("Gnani Timbre returned an empty or truncated audio stream.")

        # Ensure valid WAV container header (starts with RIFF and contains WAVE)
        if not (audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:16]):
            raise TimbreInvalidAudioError("Gnani Timbre returned invalid non-WAV audio data.")

        logger.info(
            "Timbre TTS synthesis succeeded (Voice: %s, AudioSize: %d bytes)",
            selected_voice,
            len(audio_bytes),
        )

        return TimbreSynthesisResult(
            audio_bytes=audio_bytes,
            voice=selected_voice,
            model=effective_model,
            language=selected_language,
            sample_rate=selected_sample_rate,
            format="audio/wav",
        )

    async def async_synthesize_speech(
        self,
        text: str,
        mode: Optional[str] = None,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        sample_rate: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> TimbreSynthesisResult:
        """Asynchronous wrapper for synthesize_speech using asyncio.to_thread."""
        return await asyncio.to_thread(
            self.synthesize_speech,
            text=text,
            mode=mode,
            voice=voice,
            language=language,
            sample_rate=sample_rate,
            speed=speed,
        )
