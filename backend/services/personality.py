"""
LaughBot AI - Personality Management Service.

Manages conversational personalities, system prompt resolution,
and personality mode switching for Always Funny and Roast Companion modes.
"""

from enum import Enum
import logging
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import get_settings

logger = logging.getLogger("laughbot.personality")


class PersonalityMode(str, Enum):
    """Supported personality modes for LaughBot AI."""
    ALWAYS_FUNNY = "always_funny"
    ROAST_COMPANION = "roast_companion"


class PersonalityManager:
    """
    Manages loading, caching, and resolution of personality system prompts.
    Ensures safe mode switching while preserving conversational context.
    """

    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.prompts_dir = prompts_dir or settings.PROMPTS_PATH
        self._prompt_cache: Dict[str, str] = {}

    @classmethod
    def get_supported_modes(cls) -> List[str]:
        """Return a list of all supported personality mode keys."""
        return [mode.value for mode in PersonalityMode]

    def validate_mode(self, mode: str) -> str:
        """
        Validate and normalize a personality mode string.

        Args:
            mode: Candidate mode string.

        Returns:
            Normalized mode string ('always_funny' or 'roast_companion').

        Raises:
            ValueError: If mode is not supported.
        """
        normalized = mode.strip().lower() if mode else ""
        # Support aliases like 'funny' -> 'always_funny', 'roast' -> 'roast_companion'
        if normalized in ("funny", "always_funny"):
            return PersonalityMode.ALWAYS_FUNNY.value
        elif normalized in ("roast", "roast_companion"):
            return PersonalityMode.ROAST_COMPANION.value

        supported = ", ".join(self.get_supported_modes())
        raise ValueError(f"Unsupported personality mode '{mode}'. Supported modes: {supported}")

    def get_prompt_file_path(self, mode: str) -> Path:
        """Resolve filesystem path for a mode's prompt file."""
        validated_mode = self.validate_mode(mode)
        if validated_mode == PersonalityMode.ALWAYS_FUNNY.value:
            return self.prompts_dir / "funny_mode.txt"
        elif validated_mode == PersonalityMode.ROAST_COMPANION.value:
            return self.prompts_dir / "roast_mode.txt"
        else:
            raise ValueError(f"No prompt file mapped for mode: {mode}")

    def get_system_prompt(self, mode: str) -> str:
        """
        Retrieve the system prompt for the specified personality mode,
        using memory caching for efficient repeated retrieval.

        Args:
            mode: 'always_funny' or 'roast_companion'.

        Returns:
            str: Complete system prompt instructions.
        """
        validated_mode = self.validate_mode(mode)
        settings = get_settings()
        if not settings.DEBUG and validated_mode in self._prompt_cache:
            return self._prompt_cache[validated_mode]

        prompt_path = self.get_prompt_file_path(validated_mode)
        if not prompt_path.exists():
            logger.error("System prompt file not found at path: %s", prompt_path)
            raise FileNotFoundError(f"Personality prompt file not found: {prompt_path}")

        prompt_content = prompt_path.read_text(encoding="utf-8").strip()
        self._prompt_cache[validated_mode] = prompt_content
        logger.debug("Loaded and cached prompt for mode '%s' (%d chars)", validated_mode, len(prompt_content))
        return prompt_content

    def clear_cache(self) -> None:
        """Clear cached system prompts (useful in tests or when prompt files change)."""
        self._prompt_cache.clear()

