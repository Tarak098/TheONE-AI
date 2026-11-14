"""
LaughBot AI - Gnani Evon Intelligence Service Adapter.

Provides the service integration for Gnani Evon v3.3 (30B Mamba-Transformer MoE).
When a live remote inference endpoint is configured, sends conversational requests
and parses responses. When unconfigured, explicitly reports status without
substituting another LLM or synthesizing fake outputs.
"""

import logging
from typing import Any, Dict, List, Optional

import requests

from backend.config import Settings, get_settings

logger = logging.getLogger("laughbot.evon")


class EvonError(Exception):
    """Base exception for all Evon LLM operations."""
    pass


class EvonConfigError(EvonError):
    """Raised when Evon configuration or credentials are missing."""
    pass


class EvonServiceUnavailableError(EvonError):
    """Raised when the Evon inference endpoint is unconfigured or unreachable."""
    pass


class EvonTimeoutError(EvonError):
    """Raised when an Evon inference request times out."""
    pass


class EvonConnectionError(EvonError):
    """Raised when connecting to the Evon inference endpoint fails."""
    pass


class EvonAPIError(EvonError):
    """Raised when the remote Evon endpoint responds with an error."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.details = details


class EvonService:
    """
    Service adapter for Gnani Evon v3.3 LLM.

    Handles communication with a remote Evon inference endpoint
    (e.g., vLLM, TGI, or Hugging Face Inference Endpoint).
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    @property
    def api_key(self) -> Optional[str]:
        """Retrieve configured API key or Bearer token securely."""
        return self._settings.evon_api_key

    @property
    def api_url(self) -> Optional[str]:
        """Remote inference URL for Gnani Evon."""
        return self._settings.GNANI_EVON_API_URL

    @property
    def model(self) -> str:
        """Evon model identifier."""
        return self._settings.GNANI_EVON_MODEL

    @property
    def timeout(self) -> int:
        """Inference request timeout in seconds."""
        return self._settings.GNANI_EVON_TIMEOUT

    @property
    def is_configured(self) -> bool:
        """
        Check whether an active remote Evon inference endpoint and credentials
        are fully configured.
        """
        return bool(self.api_url and self.api_url.strip() and self.api_key)

    def health_check(self) -> Dict[str, Any]:
        """
        Return configuration readiness status for Gnani Evon without leaking keys.
        """
        configured = self.is_configured
        return {
            "service": "Gnani Evon",
            "model": self.model,
            "configured": configured,
            "endpoint_configured": bool(self.api_url and self.api_url.strip()),
            "credentials_configured": bool(self.api_key),
            "status": "ready" if configured else "not_configured",
        }

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """
        Send conversational messages to the remote Gnani Evon inference endpoint.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            temperature: Sampling temperature for comedy response creativity.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional inference hyperparameters.

        Returns:
            str: Generated response text from Gnani Evon.

        Raises:
            EvonServiceUnavailableError: If inference endpoint or credentials are not configured.
            EvonTimeoutError: If remote generation times out.
            EvonConnectionError: If network connection fails.
            EvonAPIError: If remote endpoint returns an error.
        """
        if not self.is_configured:
            logger.warning("generate_response requested but Evon inference endpoint is not configured.")
            raise EvonServiceUnavailableError(
                "Gnani Evon v3.3 inference endpoint is not configured. "
                "Please configure GNANI_EVON_API_URL and credentials in .env."
            )

        clean_key = (self.api_key or "").strip().strip("\r\n\t\"'“”‘’ ")
        if "=" in clean_key:
            prefix, _, after = clean_key.partition("=")
            if prefix.strip().replace("_", "").isalnum():
                clean_key = after.strip().strip("\r\n\t\"'“”‘’ ")
        clean_key = "".join(c for c in clean_key if 33 <= ord(c) <= 126)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {clean_key}",
            "X-API-Key-ID": clean_key,  # Compatible with Gnani API headers
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        logger.info("Submitting chat completion request to Evon endpoint (Model: %s)", self.model)

        try:
            # Synchronous requests call inside async endpoint is wrapped cleanly
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as err:
            logger.error("Evon inference request timed out after %ds", self.timeout)
            raise EvonTimeoutError(f"Evon inference request timed out after {self.timeout}s.") from err
        except requests.exceptions.ConnectionError as err:
            logger.error("Connection error to Evon endpoint: %s", err)
            raise EvonConnectionError("Failed to connect to Gnani Evon inference endpoint.") from err
        except requests.exceptions.RequestException as err:
            logger.error("Network error during Evon inference: %s", err)
            raise EvonConnectionError(f"Network error during Evon inference: {err}") from err

        return self._parse_response(response)

    def _parse_response(self, response: requests.Response) -> str:
        """Parse the remote inference response payload safely."""
        status_code = response.status_code

        try:
            data = response.json()
        except Exception as err:
            logger.error("Evon endpoint returned non-JSON response (HTTP %d)", status_code)
            raise EvonAPIError(
                f"Evon endpoint returned non-JSON payload (HTTP {status_code}).",
                status_code=status_code,
            ) from err

        if status_code >= 400:
            err_msg = (
                data.get("error")
                or data.get("message")
                or data.get("detail")
                or f"Evon inference error (HTTP {status_code})"
            )
            # If error is a dict (like OpenAI format {"error": {"message": ...}})
            if isinstance(err_msg, dict):
                err_msg = err_msg.get("message", str(err_msg))

            logger.error("Evon API error (HTTP %d): %s", status_code, err_msg)
            raise EvonAPIError(f"Gnani Evon API error: {err_msg}", status_code=status_code, details=data)

        # Standard OpenAI / vLLM format: choices[0].message.content
        if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return str(choice["message"]["content"]).strip()
            elif "text" in choice:
                return str(choice["text"]).strip()

        # Hugging Face TGI format: [{"generated_text": "..."}] or {"generated_text": "..."}
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return str(data[0]["generated_text"]).strip()
        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"]).strip()

        # Direct response key
        if "response" in data:
            return str(data["response"]).strip()

        logger.error("Unexpected Evon response format: %s", data)
        raise EvonAPIError("Unrecognized response structure from Evon inference endpoint.", status_code=status_code)
