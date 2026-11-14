"""
LaughBot AI - Google Gemini Intelligence Service Adapter.

Provides the service integration for Google Gemini LLM using the official
google-genai SDK (v3.0.0+). Implements multi-turn conversation handling,
personality system prompt injection, and comprehensive error handling.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from google import genai
from google.genai import errors, types

from backend.config import Settings, get_settings

logger = logging.getLogger("laughbot.gemini")


class GeminiError(Exception):
    """Base exception for all Gemini operations."""
    pass


class GeminiConfigError(GeminiError):
    """Raised when Gemini configuration is invalid."""
    pass


class GeminiServiceUnavailableError(GeminiError):
    """Raised when Gemini API key is missing or service is unavailable."""
    pass


class GeminiTimeoutError(GeminiError):
    """Raised when a Gemini API request times out."""
    pass


class GeminiRateLimitError(GeminiError):
    """Raised when Gemini quota or rate limits are exceeded."""
    pass


class GeminiAPIError(GeminiError):
    """Raised when the Gemini API returns an error response."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class GeminiInvalidResponseError(GeminiError):
    """Raised when Gemini returns an empty or filtered response."""
    pass


class GeminiService:
    """
    Service adapter for Google Gemini LLM using the official google-genai SDK.

    Supports:
    - Asynchronous generation via client.aio.models.generate_content
    - System instruction injection via GenerateContentConfig
    - Multi-turn conversation history mapping ('user' and 'model' roles)
    - Temperature and token limits configuration
    - Robust exception handling (timeouts, quotas, API errors)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[genai.Client] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._injected_client: Optional[genai.Client] = client
        self._client: Optional[genai.Client] = client
        self._client_loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_client(self) -> genai.Client:
        """
        Return Google GenAI client instance.
        Ensures the client and its underlying httpx connection pool are bound
        to the active event loop, preventing 'Event loop is closed' errors
        across requests.
        """
        if self._injected_client is not None:
            return self._injected_client

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._client is None or self._client_loop != current_loop:
            try:
                clean_key = (self.api_key or "").strip().strip("\r\n\t\"'“”‘’ ")
                if "=" in clean_key:
                    prefix, _, after = clean_key.partition("=")
                    if prefix.strip().replace("_", "").isalnum():
                        clean_key = after.strip().strip("\r\n\t\"'“”‘’ ")
                clean_key = "".join(c for c in clean_key if 33 <= ord(c) <= 126)

                http_options = types.HttpOptions(api_version="v1beta")
                self._client = genai.Client(
                    api_key=clean_key,
                    http_options=http_options,
                )
                self._client_loop = current_loop
                logger.info("Initialized Google GenAI client (Model: %s)", self.model)
            except Exception as err:
                logger.error("Failed to initialize Google GenAI client: %s", err)
                self._client = None
                raise GeminiServiceUnavailableError(f"Failed to initialize GenAI client: {err}") from err

        return self._client

    @property
    def api_key(self) -> Optional[str]:
        """Configured Gemini API key."""
        return self._settings.GEMINI_API_KEY

    @property
    def model(self) -> str:
        """Configured Gemini model identifier."""
        return self._settings.GEMINI_MODEL

    @property
    def timeout(self) -> int:
        """Configured request timeout in seconds."""
        return self._settings.GEMINI_TIMEOUT

    @property
    def max_output_tokens(self) -> int:
        """Configured maximum output tokens."""
        return self._settings.GEMINI_MAX_OUTPUT_TOKENS

    @property
    def is_configured(self) -> bool:
        """Check whether Gemini API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def validate_configuration(self) -> bool:
        """
        Validate that Gemini is properly configured.

        Returns:
            bool: True if configured.

        Raises:
            GeminiServiceUnavailableError: If credentials are missing.
        """
        if not self.is_configured:
            raise GeminiServiceUnavailableError(
                "Google Gemini API key is not configured. "
                "Please configure GEMINI_API_KEY in your .env file."
            )
        return True

    def health_check(self) -> Dict[str, Any]:
        """
        Return configuration readiness status for Google Gemini / Gemma without exposing secrets.
        """
        configured = self.is_configured
        is_gemma = "gemma" in self.model.lower()
        service_name = "Google Gemma" if is_gemma else "Google Gemini"
        return {
            "service": service_name,
            "provider": "gemini",
            "model": self.model,
            "configured": configured,
            "status": "ready" if configured else "not_configured",
        }

    def _format_contents(
        self,
        messages: List[Union[Dict[str, str], types.Content]],
    ) -> List[types.Content]:
        """
        Format message sequence into official types.Content objects.

        Maps roles:
        - 'user' -> 'user'
        - 'assistant' or 'model' -> 'model'
        System messages are excluded here as they are supplied via system_instruction.
        """
        contents: List[types.Content] = []

        for msg in messages:
            if isinstance(msg, types.Content):
                contents.append(msg)
                continue

            role = msg.get("role", "user")
            content_text = msg.get("content", "")

            # Exclude system messages from content sequence
            if role == "system":
                continue

            # Map assistant role to Gemini's 'model' role
            gemini_role = "model" if role in ("assistant", "model") else "user"
            content_obj = types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=content_text)],
            )
            contents.append(content_obj)

        return contents

    async def generate_response(
        self,
        messages: List[Union[Dict[str, str], types.Content]],
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a conversational AI response using Google Gemini.

        Args:
            messages: Conversation history turns (role/content dicts or types.Content).
            system_instruction: Personality system prompt text.
            temperature: Creativity/sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate (defaults to GEMINI_MAX_OUTPUT_TOKENS).

        Returns:
            str: Generated comedy/roast response text.

        Raises:
            GeminiServiceUnavailableError: If API key is missing.
            GeminiRateLimitError: If API quota or rate limits are exceeded.
            GeminiTimeoutError: If the request times out.
            GeminiAPIError: If the Gemini API returns an error.
            GeminiInvalidResponseError: If the response is empty or blocked.
        """
        self.validate_configuration()

        client = self._get_client()

        formatted_contents = self._format_contents(messages)
        if not formatted_contents:
            raise GeminiAPIError("No conversation content provided for Gemini generation.")

        is_gemma = "gemma" in self.model.lower()
        # For Gemma models, max_output_tokens must be omitted (None) because passing
        # a token budget on Gemma via the Gemini API triggers a 500 INTERNAL error.
        tokens = None if is_gemma else (max_tokens or self.max_output_tokens)

        # For Gemma models, configure MINIMAL thinking to ensure fast, short, direct responses
        thinking_config = types.ThinkingConfig(thinking_level="MINIMAL") if is_gemma else None

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=tokens,
            thinking_config=thinking_config,
        )

        logger.info(
            "Calling Google GenAI API (Model: %s, Messages: %d, MaxTokens: %s)",
            self.model,
            len(formatted_contents),
            str(tokens),
        )

        max_attempts = 4
        backoff_delay = 1.5
        response = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=self.model,
                        contents=formatted_contents,
                        config=config,
                    ),
                    timeout=float(self.timeout),
                )
                break
            except asyncio.TimeoutError as err:
                if attempt < max_attempts:
                    logger.warning(
                        "GenAI request timed out (attempt %d/%d). Retrying in %.1fs...",
                        attempt,
                        max_attempts,
                        backoff_delay,
                    )
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= 2.0
                    continue
                logger.error("GenAI API request timed out after %ds", self.timeout)
                raise GeminiTimeoutError(f"Gemini API request timed out after {self.timeout}s.") from err
            except errors.APIError as err:
                code = getattr(err, "code", None)
                status_str = str(getattr(err, "status", "") or "")
                msg = str(err)
                logger.error(
                    "GenAI API error (Status: %s, Code: %s, Attempt: %d/%d): %s",
                    status_str,
                    code,
                    attempt,
                    max_attempts,
                    msg,
                )

                # HTTP 429 / Quota: If on gemma, attempt failover to high-quota flash models
                if code == 429 or "RESOURCE_EXHAUSTED" in status_str or "quota" in msg.lower():
                    if is_gemma:
                        for fallback_model in ("gemini-2.5-flash", "gemini-2.0-flash-lite"):
                            logger.warning(
                                "Gemma rate/token limit reached (429 RESOURCE_EXHAUSTED). "
                                "Failing over to %s for seamless continuity...",
                                fallback_model,
                            )
                            try:
                                fallback_config = types.GenerateContentConfig(
                                    system_instruction=system_instruction,
                                    temperature=temperature,
                                    max_output_tokens=60,
                                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                                )
                                fallback_resp = await asyncio.wait_for(
                                    client.aio.models.generate_content(
                                        model=fallback_model,
                                        contents=formatted_contents,
                                        config=fallback_config,
                                    ),
                                    timeout=float(self.timeout),
                                )
                                if fallback_resp and fallback_resp.text and fallback_resp.text.strip():
                                    logger.info("Successfully recovered from Gemma 429 limit using %s fallback.", fallback_model)
                                    return fallback_resp.text.strip()
                            except Exception as fallback_err:
                                logger.error("Failover on 429 to %s failed: %s", fallback_model, fallback_err)

                    raise GeminiRateLimitError(
                        "Google Gemini/Gemma API quota or rate limit exceeded. Please try again later."
                    ) from err

                # HTTP 401 / Authentication: Do NOT retry
                if code == 401 or "API_KEY_INVALID" in msg:
                    raise GeminiAPIError(
                        "Invalid Gemini API key (HTTP 401). Please check GEMINI_API_KEY in .env.",
                        status_code=401,
                    ) from err

                # HTTP 403 / Permission: Do NOT retry
                if code == 403 or "PERMISSION_DENIED" in status_str:
                    raise GeminiAPIError(
                        f"Permission denied for Gemini/Gemma API (HTTP 403): {msg}",
                        status_code=403,
                    ) from err

                # HTTP 500 / 502 / 503 Server Error: Retry with limited exponential backoff
                if (code in (500, 502, 503) or "INTERNAL" in status_str) and attempt < max_attempts:
                    logger.warning(
                        "Temporary GenAI server error (%s). Retrying in %.1fs...",
                        code,
                        backoff_delay,
                    )
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= 2.0
                    continue

                # If Gemma returns persistent 500 INTERNAL (upstream infrastructure outage), failover to high-quota flash models
                if is_gemma and (code in (500, 502, 503) or "INTERNAL" in status_str):
                    for fallback_model in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"):
                        logger.warning(
                            "Upstream Google Gemma endpoint returned persistent 500 INTERNAL (%s). "
                            "Failing over to %s to ensure continuous service...",
                            msg,
                            fallback_model,
                        )
                        try:
                            fallback_config = types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=temperature,
                                max_output_tokens=60,
                                thinking_config=types.ThinkingConfig(thinking_budget=0),
                            )
                            fallback_resp = await asyncio.wait_for(
                                client.aio.models.generate_content(
                                    model=fallback_model,
                                    contents=formatted_contents,
                                    config=fallback_config,
                                ),
                                timeout=float(self.timeout),
                            )
                            if fallback_resp and fallback_resp.text and fallback_resp.text.strip():
                                logger.info("Successfully recovered from upstream Gemma 500 error using %s fallback.", fallback_model)
                                return fallback_resp.text.strip()
                        except Exception as fallback_err:
                            logger.error("Failover to %s also encountered error: %s", fallback_model, fallback_err)

                if "deadline" in msg.lower() or "timeout" in msg.lower():
                    raise GeminiTimeoutError(f"Gemini request timed out: {msg}") from err

                raise GeminiAPIError(
                    f"Google Gemini API error: {msg}",
                    status_code=code,
                    details=getattr(err, "response_json", None),
                ) from err
            except Exception as err:
                if attempt < max_attempts:
                    logger.warning(
                        "Transient error during generation (%s). Retrying in %.1fs...",
                        err,
                        backoff_delay,
                    )
                    await asyncio.sleep(backoff_delay)
                    backoff_delay *= 2.0
                    continue
                logger.exception("Unexpected error during GenAI content generation: %s", err)
                raise GeminiAPIError(f"Unexpected error communicating with Gemini: {err}") from err

        # Validate response content
        if not response or not response.text or not response.text.strip():
            finish_reason = None
            if response and getattr(response, "candidates", None) and len(response.candidates) > 0:
                finish_reason = getattr(response.candidates[0], "finish_reason", "UNKNOWN")
            logger.warning("Gemini returned empty or filtered response. Finish reason: %s", finish_reason)
            raise GeminiInvalidResponseError(
                f"Gemini returned an empty or filtered response (Finish reason: {finish_reason})."
            )

        return response.text.strip()
