"""
TheONE AI - Final Unified Comedy Personality Management Service.

TheONE AI has ONE unified personality: ALWAYS funny, sarcastic, witty, and entertaining.
The conversational style adapts dynamically to the user's message, while the core
comedy-first identity never changes.
"""

from enum import Enum
import logging
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import get_settings

logger = logging.getLogger("theone.personality")


class PersonalityMode(str, Enum):
    """Unified comedy personality mode for TheONE AI."""
    UNIFIED_COMEDY = "unified_comedy"


class PersonalityManager:
    """
    Manages loading, caching, and resolution of TheONE AI's unified comedy system prompt.
    Adapts comedy style dynamically without separate mode divisions.
    """

    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        settings = get_settings()
        self.prompts_dir = prompts_dir or settings.PROMPTS_PATH
        self._prompt_cache: Dict[str, str] = {}

    @classmethod
    def get_supported_modes(cls) -> List[str]:
        """Return the unified personality identifier."""
        return [PersonalityMode.UNIFIED_COMEDY.value]

    def validate_mode(self, mode: Optional[str] = None) -> str:
        """
        Validate and normalize personality mode to the single unified comedy personality.
        Gracefully accepts legacy aliases ('always_funny', 'roast_companion') and normalizes
        them to 'unified_comedy'.

        Args:
            mode: Candidate mode string or None.

        Returns:
            str: 'unified_comedy'
        """
        # TheONE AI has ONE unified personality. All inputs resolve to unified_comedy.
        return PersonalityMode.UNIFIED_COMEDY.value

    def get_prompt_file_path(self, mode: Optional[str] = None) -> Path:
        """Resolve filesystem path for the unified comedy prompt file."""
        unified_path = self.prompts_dir / "unified_comedy.txt"
        if unified_path.exists():
            return unified_path
        # Fallback to funny_mode.txt if unified file is missing
        return self.prompts_dir / "funny_mode.txt"

    def get_system_prompt(self, mode: Optional[str] = None) -> str:
        """
        Retrieve TheONE AI's unified comedy system prompt using memory caching.

        Args:
            mode: Optional mode parameter (defaults to unified_comedy).

        Returns:
            str: Complete unified comedy system prompt.
        """
        normalized_mode = self.validate_mode(mode)
        settings = get_settings()
        if not settings.DEBUG and normalized_mode in self._prompt_cache:
            return self._prompt_cache[normalized_mode]

        prompt_path = self.get_prompt_file_path(normalized_mode)
        if not prompt_path.exists():
            logger.error("Unified comedy prompt file not found at path: %s", prompt_path)
            raise FileNotFoundError(f"Personality prompt file not found: {prompt_path}")

        prompt_content = prompt_path.read_text(encoding="utf-8").strip()
        self._prompt_cache[normalized_mode] = prompt_content
        logger.debug("Loaded unified comedy prompt (%d chars)", len(prompt_content))
        return prompt_content

    def clear_cache(self) -> None:
        """Clear cached system prompts (useful in tests or when prompt files change)."""
        self._prompt_cache.clear()
