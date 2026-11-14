"""
LaughBot AI - Centralized Configuration Management.

Loads and validates application settings from the root `.env` file.
All API credentials remain strictly optional and masked to prevent credential exposure.
"""

from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_core import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("laughbot.config")

# Project root path (directory containing .env and backend/)
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"
PROMPTS_DIR = BACKEND_DIR / "prompts"


class Settings(BaseSettings):
    """
    Application settings for LaughBot AI.

    Loads from environment variables and root .env file.
    All secrets are marked with repr=False to prevent accidental leakage in logs.
    """

    # Application settings
    APP_NAME: str = Field(default="TheONE AI", description="Application name")
    APP_ENV: str = Field(default="development", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=True, description="Debug mode flag")
    HOST: str = Field(default="0.0.0.0", description="Host address for backend binding")
    PORT: int = Field(default=8000, ge=1, le=65535, description="Port number for backend binding")
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"], description="Allowed CORS origins")

    # LLM Intelligence Engine Provider Selection ('gemini' or 'evon')
    LLM_PROVIDER: str = Field(default="gemini", description="Active LLM provider: 'gemini' or 'evon'")

    # Google Gemini / Gemma Configuration
    GEMINI_API_KEY: Optional[str] = Field(default=None, repr=False, description="Google Gemini API key")
    GEMINI_MODEL: str = Field(default="gemma-4-26b-a4b-it", description="Configured Gemini or Gemma model identifier")
    GEMINI_TIMEOUT: int = Field(default=60, ge=1, le=300, description="Gemini API timeout in seconds")
    GEMINI_MAX_OUTPUT_TOKENS: int = Field(default=60, ge=10, le=4096, description="Gemini max tokens to generate (short response optimization)")

    # Shared Gnani API Key (Optional)
    GNANI_API_KEY: Optional[str] = Field(default=None, repr=False, description="Shared Gnani API key")

    # Gnani Prisma (Speech-to-Text / ASR) Configuration
    GNANI_PRISMA_API_KEY: Optional[str] = Field(default=None, repr=False, description="Gnani Prisma STT API key")
    GNANI_PRISMA_API_URL: str = Field(
        default="https://api.vachana.ai/stt/v3",
        description="Endpoint URL for Gnani Prisma STT API",
    )
    GNANI_PRISMA_VERSION: str = Field(default="v3", description="Gnani Prisma API version")
    GNANI_PRISMA_TIMEOUT: int = Field(default=60, ge=1, le=300, description="Prisma request timeout in seconds")

    # Gnani Evon v3.3 (AI Response Generation / LLM) Configuration
    GNANI_EVON_ENABLED: bool = Field(default=False, description="Whether Gnani Evon integration is enabled")
    GNANI_EVON_API_KEY: Optional[str] = Field(default=None, repr=False, description="Gnani Evon LLM API key / Bearer Token")
    GNANI_EVON_API_URL: Optional[str] = Field(default=None, description="Remote inference endpoint URL for Gnani Evon")
    GNANI_EVON_MODEL: str = Field(default="evon-v3.3", description="Evon model identifier")
    GNANI_EVON_TIMEOUT: int = Field(default=60, ge=1, le=300, description="Evon inference timeout in seconds")
    MAX_CONVERSATION_TURNS: int = Field(default=20, ge=1, le=100, description="Maximum dialogue turns kept in memory per conversation")

    # Gnani Timbre (Text-to-Speech / TTS) Configuration
    GNANI_TIMBRE_API_KEY: Optional[str] = Field(default=None, repr=False, description="Gnani Timbre TTS API key")
    GNANI_TIMBRE_API_URL: str = Field(
        default="https://api.vachana.ai/api/v1/tts/inference",
        description="Endpoint URL for Gnani Timbre TTS API",
    )
    GNANI_TTS_MODEL: str = Field(default="timbre-2.5", description="Gnani Timbre model identifier")
    GNANI_TTS_LANGUAGE: str = Field(default="en-IN", description="Default TTS language code")
    GNANI_TTS_VOICE: str = Field(default="Zayan", description="Default TTS voice")
    GNANI_TTS_VOICE_FUNNY: str = Field(default="Yashvi", description="Voice for Always Funny mode")
    GNANI_TTS_VOICE_ROAST: str = Field(default="Zayan", description="Voice for Roast Companion mode")
    GNANI_TTS_SAMPLE_RATE: int = Field(default=48000, description="TTS output sample rate in Hz")
    GNANI_TIMBRE_TIMEOUT: int = Field(default=30, ge=1, le=120, description="Timbre TTS request timeout in seconds")

    # Paths
    PROMPTS_PATH: Path = PROMPTS_DIR

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Union[str, List[str], Any]) -> List[str]:
        """Parse CORS origins from JSON list, comma-separated string, or python list."""
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed]
                except json.JSONDecodeError:
                    pass
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        elif isinstance(value, (list, tuple)):
            return [str(origin).strip() for origin in value]
        return ["*"]

    @field_validator(
        "GEMINI_API_KEY",
        "GNANI_API_KEY",
        "GNANI_PRISMA_API_KEY",
        "GNANI_EVON_API_KEY",
        "GNANI_TIMBRE_API_KEY",
        mode="before",
    )
    @classmethod
    def clean_api_credential(cls, value: Any) -> Optional[str]:
        """
        Sanitize and normalize API credential values.
        Strips:
        - Accidental variable prefix (e.g., 'GNANI_API_KEY=xxx' -> 'xxx')
        - Surrounding double or single quotes ('"xxx"' -> 'xxx')
        - Leading/trailing whitespace, newlines, carriage returns, tabs
        - Non-printable or control characters to ensure HTTP header compliance
        """
        if value is None:
            return None
        val_str = str(value).strip().strip("\r\n\t")
        if not val_str:
            return None
        # Handle accidental 'KEY_NAME=value' paste in Render or .env
        if "=" in val_str:
            prefix, _, after = val_str.partition("=")
            if prefix.strip().replace("_", "").isalnum():
                val_str = after.strip()
        # Strip surrounding quotes and special punctuation quotes
        val_str = val_str.strip("\"'“”‘’ \t\r\n")
        # Keep only printable ASCII characters without whitespace
        val_str = "".join(c for c in val_str if 33 <= ord(c) <= 126)
        return val_str if val_str else None

    # Credential Resolution Properties
    @property
    def has_gemini_credentials(self) -> bool:
        """Check whether Google Gemini API key is configured (boolean flag only)."""
        return bool(self.GEMINI_API_KEY and self.GEMINI_API_KEY.strip())

    @property
    def active_llm_provider(self) -> str:
        """Return normalized active LLM provider name."""
        return (self.LLM_PROVIDER or "gemini").lower().strip()

    @property
    def prisma_api_key(self) -> Optional[str]:
        """Resolve Prisma API key from specific key or shared key."""
        return self.GNANI_PRISMA_API_KEY or self.GNANI_API_KEY

    @property
    def has_prisma_credentials(self) -> bool:
        """Check whether Prisma has credentials configured (boolean flag only)."""
        return bool(self.prisma_api_key)

    @property
    def evon_api_key(self) -> Optional[str]:
        """Resolve Evon API key from specific key or shared key."""
        return self.GNANI_EVON_API_KEY or self.GNANI_API_KEY

    @property
    def has_evon_credentials(self) -> bool:
        """Check whether Evon has remote endpoint and credentials configured."""
        return bool(self.GNANI_EVON_API_URL and (self.GNANI_EVON_API_KEY or self.GNANI_API_KEY))

    @property
    def timbre_api_key(self) -> Optional[str]:
        """Resolve Timbre API key from specific key or shared key."""
        return self.GNANI_TIMBRE_API_KEY or self.GNANI_API_KEY

    @property
    def has_timbre_credentials(self) -> bool:
        """Check whether Timbre has credentials configured (boolean flag only)."""
        return bool(self.timbre_api_key)

    @property
    def normalized_tts_model(self) -> str:
        """Normalize timbre-2.5 to timbre-v2.5 for official API compatibility."""
        model = (self.GNANI_TTS_MODEL or "timbre-v2.5").strip()
        if model in ("timbre-2.5", "timbre2.5"):
            return "timbre-v2.5"
        if model in ("timbre-3.0", "timbre3.0"):
            return "timbre-v3.0"
        return model

    def __repr__(self) -> str:
        """Safe representation masking sensitive credentials."""
        return (
            f"Settings(APP_NAME={self.APP_NAME!r}, APP_ENV={self.APP_ENV!r}, "
            f"DEBUG={self.DEBUG!r}, HOST={self.HOST!r}, PORT={self.PORT!r}, "
            f"CORS_ORIGINS={self.CORS_ORIGINS!r}, "
            f"LLM_PROVIDER={self.LLM_PROVIDER!r}, "
            f"GEMINI_MODEL={self.GEMINI_MODEL!r}, "
            f"has_gemini={self.has_gemini_credentials}, "
            f"GNANI_PRISMA_API_URL={self.GNANI_PRISMA_API_URL!r}, "
            f"GNANI_EVON_API_URL={self.GNANI_EVON_API_URL!r}, "
            f"GNANI_EVON_MODEL={self.GNANI_EVON_MODEL!r}, "
            f"has_prisma={self.has_prisma_credentials}, "
            f"has_evon={self.has_evon_credentials}, "
            f"has_timbre={self.has_timbre_credentials})"
        )

    def __str__(self) -> str:
        return self.__repr__()


def load_settings() -> Settings:
    """
    Load settings with clear validation error handling.

    Raises:
        RuntimeError: If configuration validation fails with explicit details.
    """
    try:
        return Settings()
    except ValidationError as err:
        errors = [f" - {e['loc']}: {e['msg']} (got {e.get('input', 'N/A')})" for e in err.errors()]
        error_msg = f"Configuration validation failed:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise RuntimeError(error_msg) from err


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return load_settings()
