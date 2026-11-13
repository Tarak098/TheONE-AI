"""
LaughBot AI - Intelligence Engine Orchestrator.

Coordinates conversation state, personality system prompts, and multi-provider
LLM response generation (Google Gemini API and Gnani Evon v3.3).
Preserves session context and ensures clean provider abstraction.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

from backend.config import Settings, get_settings
from backend.services.conversation import ConversationManager
from backend.services.evon import EvonService, EvonServiceUnavailableError
from backend.services.gemini import GeminiService, GeminiServiceUnavailableError
from backend.services.personality import PersonalityManager
from backend.services.response_validator import (
    count_sentences,
    count_words,
    enforce_strict_short_response,
    is_sensitive_message,
    is_valid_short_response,
    validate_response_quality,
)

logger = logging.getLogger("laughbot.intelligence")

# Maximum permitted character length for a single user message
MAX_MESSAGE_LENGTH = 2000


class IntelligenceValidationError(ValueError):
    """Raised when an incoming chat message fails validation."""
    pass


@dataclass
class ChatTurnResult:
    """Structured result of a processed intelligence turn."""
    conversation_id: str
    mode: str
    user_message: str
    assistant_response: str
    provider: str
    model: str
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "success": self.success,
            "conversation_id": self.conversation_id,
            "mode": self.mode,
            "user_message": self.user_message,
            "assistant_response": self.assistant_response,
            "provider": self.provider,
            "model": self.model,
        }


class IntelligenceEngine:
    """
    Core conversational intelligence coordinator for LaughBot AI.

    Connects:
    1. Input text (transcribed from Prisma STT or typed by user)
    2. Personality system prompts (Always Funny / Roast Companion)
    3. Conversation session memory
    4. Active LLM response generation (Google Gemini or Gnani Evon)
    """

    def __init__(
        self,
        gemini_service: Optional[GeminiService] = None,
        evon_service: Optional[EvonService] = None,
        personality_manager: Optional[PersonalityManager] = None,
        conversation_manager: Optional[ConversationManager] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self.gemini = gemini_service or GeminiService(settings=self._settings)
        self.evon = evon_service or EvonService(settings=self._settings)
        self.personality = personality_manager or PersonalityManager()
        self.conversations = conversation_manager or ConversationManager()

    @property
    def active_provider(self) -> str:
        """Return configured active LLM provider ('gemini' or 'evon')."""
        return self._settings.active_llm_provider

    @property
    def active_model(self) -> str:
        """Return model identifier of currently active provider."""
        if self.active_provider == "gemini":
            return self.gemini.model
        return self.evon.model

    async def generate_response(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        mode: str = "unified_comedy",
        language: str = "en",
    ) -> ChatTurnResult:
        """Alias for process_turn supporting standard method signatures."""
        return await self.process_turn(
            message=message,
            conversation_id=conversation_id,
            mode=mode,
            language=language,
        )

    async def process_turn(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        mode: str = "unified_comedy",
        language: str = "en",
    ) -> ChatTurnResult:
        """
        Execute an end-to-end intelligence turn.

        Workflow:
        1. Receive and validate user message.
        2. Validate and normalize personality mode (unified comedy).
        3. Retrieve or create isolated conversation session.
        4. Load unified comedy system prompt.
        5. Dispatch to active LLM provider (Gemini or Evon).
        6. Persist user and assistant turns in conversation history.
        7. Return structured ChatTurnResult.

        Raises:
            IntelligenceValidationError: If user message is empty or exceeds limits.
            ValueError: If personality mode is invalid.
            GeminiServiceUnavailableError: If Gemini API key is missing.
            EvonServiceUnavailableError: If Evon endpoint is unconfigured.
        """
        # 1. Validate Input Message
        if not message or not message.strip():
            raise IntelligenceValidationError("Message text cannot be empty or whitespace.")

        clean_message = message.strip()
        if len(clean_message) > MAX_MESSAGE_LENGTH:
            raise IntelligenceValidationError(
                f"Message length ({len(clean_message)} characters) exceeds maximum limit of {MAX_MESSAGE_LENGTH}."
            )

        # 2. Validate and Normalize Personality Mode (Unified Comedy)
        valid_mode = self.personality.validate_mode(mode)

        # 3. Retrieve or Create Conversation Session
        session = self.conversations.get_or_create_conversation(conversation_id, mode=valid_mode)
        if session.mode != valid_mode:
            session.switch_mode(valid_mode)

        # 4. Load Unified Comedy System Prompt (Adaptive, Mood-Independent Humour)
        system_prompt = self.personality.get_system_prompt(valid_mode)

        # 5. Route to Active LLM Provider
        provider_name = self.active_provider

        logger.info(
            "Executing intelligence turn (ConvID: %s, Mode: %s, Provider: %s, HistoryLength: %d turns)",
            session.conversation_id,
            valid_mode,
            provider_name,
            len(session.messages),
        )

        if provider_name == "gemini":
            # Pass chronological history turns + new candidate user message
            history_turns = self.conversations.get_history_turns(session.conversation_id)
            candidate_turns = history_turns + [{"role": "user", "content": clean_message}]

            assistant_reply = await self.gemini.generate_response(
                messages=candidate_turns,
                system_instruction=system_prompt,
            )
            response_model = self.gemini.model

        elif provider_name == "evon":
            # Evon format: system message at head, followed by dialogue turns
            history_messages = self.conversations.get_formatted_messages(
                session.conversation_id,
                system_prompt=system_prompt,
            )
            full_prompt_sequence = history_messages + [{"role": "user", "content": clean_message}]

            assistant_reply = await self.evon.generate_response(full_prompt_sequence)
            response_model = self.evon.model

        else:
            raise IntelligenceValidationError(
                f"Unsupported LLM provider '{provider_name}'. Supported providers: 'gemini', 'evon'."
            )

        # 6. Response Quality & Context Relevance Validation
        is_valid, failure_reason = validate_response_quality(
            user_message=clean_message,
            candidate_response=assistant_reply,
            mode=valid_mode,
        )

        if not is_valid:
            logger.warning(
                "Response failed quality/relevance validation: %s. Candidate: %r. Regenerating with context override...",
                failure_reason,
                assistant_reply,
            )
            relevance_instruction = (
                f"{system_prompt}\n\n"
                f"CRITICAL RELEVANCE CORRECTION: Your previous candidate response was rejected because: {failure_reason}. "
                f"User's exact message: {clean_message!r}. "
                f"You MUST respond directly to what the user said about their specific topic. "
                f"Never use generic off-topic insults or stock jokes. "
                f"Deliver EXACTLY ONE punchline in 3 to 10 words (maximum 15 words). No extra sentences."
            )
            try:
                if provider_name == "gemini":
                    assistant_reply = await self.gemini.generate_response(
                        messages=candidate_turns,
                        system_instruction=relevance_instruction,
                    )
                elif provider_name == "evon":
                    strict_prompt_seq = self.conversations.get_formatted_messages(
                        session.conversation_id,
                        system_prompt=relevance_instruction,
                    ) + [{"role": "user", "content": clean_message}]
                    assistant_reply = await self.evon.generate_response(strict_prompt_seq)
            except Exception as regen_err:
                logger.warning("Regeneration attempt failed (%s). Falling back to deterministic enforcement.", regen_err)

        # Enforce exact 1-sentence and max 15 words cleanly without mid-word cutting
        final_reply, was_modified = enforce_strict_short_response(assistant_reply, max_words=15)
        if was_modified:
            logger.info("Enforced strict 1-sentence rule: %r -> %r", assistant_reply, final_reply)
        assistant_reply = final_reply

        # 7. Persist Turns in Conversation History
        session.add_message(role="user", content=clean_message)
        session.add_message(role="assistant", content=assistant_reply)

        return ChatTurnResult(
            conversation_id=session.conversation_id,
            mode=valid_mode,
            user_message=clean_message,
            assistant_response=assistant_reply,
            provider=provider_name,
            model=response_model,
            success=True,
        )
