"""
LaughBot AI - Gnani Prisma Speech-to-Text (STT) Service.

Implements the official Gnani Prisma v3 REST API integration
via https://api.vachana.ai/stt/v3 using multipart form-data
and X-API-Key-ID authentication.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

import requests

from backend.config import Settings, get_settings

logger = logging.getLogger("laughbot.prisma")


class PrismaError(Exception):
    """Base exception for all Gnani Prisma STT operations."""
    pass


class PrismaConfigError(PrismaError):
    """Raised when Prisma credentials are unconfigured or invalid."""
    pass


class PrismaTimeoutError(PrismaError):
    """Raised when a request to the Prisma API times out."""
    pass


class PrismaConnectionError(PrismaError):
    """Raised when connection to the Prisma API fails."""
    pass


class PrismaAPIError(PrismaError):
    """Raised when the Prisma API responds with an error status or failure payload."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


@dataclass
class TranscriptionResult:
    """Structured result returned by Gnani Prisma STT."""
    transcript: str
    language_code: str
    request_id: str
    success: bool
    raw_response: Dict[str, Any]


class PrismaService:
    """Service client for Gnani Prisma v3 Speech-to-Text API."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    @property
    def api_key(self) -> Optional[str]:
        """Retrieve the configured API key securely."""
        return self._settings.prisma_api_key

    @property
    def api_url(self) -> str:
        """Endpoint URL for Prisma v3 STT."""
        return self._settings.GNANI_PRISMA_API_URL

    @property
    def timeout(self) -> int:
        """Timeout in seconds for Prisma requests."""
        return self._settings.GNANI_PRISMA_TIMEOUT

    @property
    def is_configured(self) -> bool:
        """Check whether Prisma has valid API credentials configured."""
        return bool(self.api_key)

    def transcribe_audio(
        self,
        audio_bytes: bytes,
        language_code: str = "en-IN",
        filename: str = "recording.wav",
        format_type: str = "transcribe",
        itn_native_numerals: bool = True,
    ) -> TranscriptionResult:
        """
        Send WAV audio to Gnani Prisma v3 and return the recognized transcript.

        Args:
            audio_bytes: Validated raw WAV audio bytes.
            language_code: Target BCP-47 language tag (e.g., 'en-IN', 'hi-IN', 'te-IN').
            filename: Name of the audio file in the multipart payload.
            format_type: Transcription format ('transcribe').
            itn_native_numerals: Whether to format numbers into native numerals.

        Returns:
            TranscriptionResult: Containing transcript, request_id, and language.

        Raises:
            PrismaConfigError: If credentials are not configured.
            PrismaTimeoutError: If the API request times out.
            PrismaConnectionError: If network connection fails.
            PrismaAPIError: If the API returns an error response.
        """
        if not self.is_configured:
            logger.error("Attempted transcription without configured Prisma API credentials.")
            raise PrismaConfigError("Gnani Prisma API credentials are not configured.")

        # Defensive sanitization of API key to prevent header syntax errors
        clean_key = (self.api_key or "").strip().strip("\r\n\t\"'")
        if "=" in clean_key:
            prefix, _, after = clean_key.partition("=")
            if prefix.strip().replace("_", "").isalnum():
                clean_key = after.strip().strip("\r\n\t\"'")

        headers = {
            "X-API-Key-ID": clean_key,
        }

        # Form fields documented for Prisma v3 REST
        payload = {
            "language_code": language_code,
            "preferred_language": language_code,
            "format": format_type,
            "itn_native_numerals": "true" if itn_native_numerals else "false",
        }

        files = {
            "audio_file": (filename, audio_bytes, "audio/wav"),
        }

        logger.info(
            "Submitting audio to Gnani Prisma STT (URL: %s, Lang: %s, AudioSize: %d bytes)",
            self.api_url,
            language_code,
            len(audio_bytes),
        )

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=payload,
                files=files,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as err:
            logger.error("Prisma STT request timed out after %d seconds.", self.timeout)
            raise PrismaTimeoutError(
                f"Gnani Prisma STT request timed out after {self.timeout}s."
            ) from err
        except requests.exceptions.ConnectionError as err:
            logger.error("Failed to connect to Gnani Prisma STT: %s", err)
            raise PrismaConnectionError(
                "Failed to connect to Gnani Prisma speech service."
            ) from err
        except requests.exceptions.RequestException as err:
            logger.error("Network request error during Prisma STT: %s", err)
            err_msg = str(err)
            if clean_key and clean_key in err_msg:
                err_msg = err_msg.replace(clean_key, "[REDACTED]")
            if "header value" in err_msg.lower():
                err_msg = "Invalid API key format or header characters. Check GNANI_API_KEY environment variable."
            raise PrismaConnectionError(f"Network error during speech recognition: {err_msg}") from err

        # Parse response safely
        return self._parse_response(response, language_code)

    def _parse_response(self, response: requests.Response, language_code: str) -> TranscriptionResult:
        """Parse and validate the JSON response from Gnani Prisma."""
        status_code = response.status_code

        try:
            data = response.json()
        except Exception as err:
            logger.error("Prisma returned non-JSON response (HTTP %d): %s", status_code, response.text[:200])
            raise PrismaAPIError(
                f"Gnani Prisma returned an unexpected non-JSON response (HTTP {status_code}).",
                status_code=status_code,
            ) from err

        # Check HTTP error status
        if status_code >= 400:
            error_message = (
                data.get("error")
                or data.get("message")
                or data.get("detail")
                or f"Gnani Prisma API error (HTTP {status_code})"
            )
            logger.error("Prisma STT API error (HTTP %d): %s", status_code, error_message)
            raise PrismaAPIError(
                f"Gnani Prisma API error: {error_message}",
                status_code=status_code,
                details=data,
            )

        # Check API-level success flag
        api_success = data.get("success", True)
        if api_success is False:
            error_message = data.get("error") or data.get("message") or "Speech recognition failed."
            logger.error("Prisma reported unsuccessful transcription: %s", error_message)
            raise PrismaAPIError(
                f"Gnani Prisma transcription failed: {error_message}",
                status_code=status_code,
                details=data,
            )

        # Extract transcript field
        transcript = (
            data.get("transcript")
            if "transcript" in data
            else data.get("text", "")
        )
        if transcript is None:
            transcript = ""

        request_id = str(data.get("request_id") or data.get("requestId") or "")
        detected_lang = str(data.get("detected_language_code") or language_code)

        logger.info(
            "Prisma STT transcription succeeded (RequestID: %s, Length: %d chars)",
            request_id,
            len(transcript),
        )

        return TranscriptionResult(
            transcript=str(transcript).strip(),
            language_code=detected_lang,
            request_id=request_id,
            success=True,
            raw_response=data,
        )
