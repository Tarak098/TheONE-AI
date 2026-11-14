"""
LaughBot AI - Conversation Management Service.

Manages multi-turn conversation sessions, message history, user isolation,
personality switching, and memory truncation to prevent unbounded growth.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

from backend.config import get_settings

logger = logging.getLogger("laughbot.conversation")


@dataclass
class ChatMessage:
    """Represents a single message turn in a conversation."""
    role: str  # "user" | "assistant" | "system"
    content: str
    mode: str
    language: str = "en-IN"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary representation."""
        return {
            "role": self.role,
            "content": self.content,
            "mode": self.mode,
            "language": self.language,
            "timestamp": self.timestamp.isoformat(),
        }


class ConversationSession:
    """Represents an isolated user conversation session."""

    def __init__(self, conversation_id: str, mode: str = "unified_comedy", max_turns: int = 20) -> None:
        self.conversation_id = conversation_id
        self.mode = mode
        self.max_turns = max_turns
        self.active_voice: str = "Yashvi"
        self.last_language: str = "en-IN"
        self.messages: List[ChatMessage] = []
        self.recent_openers: List[str] = []
        self.recent_topics: List[str] = []
        self.running_entities: List[str] = []
        self.created_at = datetime.now(timezone.utc)
        self.last_updated_at = datetime.now(timezone.utc)

    def record_turn_metadata(
        self,
        topic: Optional[str] = None,
        assistant_opener: Optional[str] = None,
        user_entity: Optional[str] = None,
    ) -> None:
        """Record topic, opener phrase, and user subject for continuity & anti-repetition."""
        if topic:
            self.recent_topics.append(topic)
            if len(self.recent_topics) > 10:
                self.recent_topics = self.recent_topics[-10:]
        if assistant_opener:
            clean_op = assistant_opener.strip().lower()
            if clean_op:
                self.recent_openers.append(clean_op)
                if len(self.recent_openers) > 6:
                    self.recent_openers = self.recent_openers[-6:]
        if user_entity:
            clean_ent = user_entity.strip()
            if clean_ent and clean_ent not in self.running_entities:
                self.running_entities.append(clean_ent)
                if len(self.running_entities) > 8:
                    self.running_entities = self.running_entities[-8:]

    def get_recent_openers(self) -> List[str]:
        """Return list of recent assistant response openers (first 1-2 words)."""
        return list(self.recent_openers)

    def get_potential_callbacks(self) -> List[str]:
        """Return list of notable user subjects/entities from earlier turns."""
        return list(self.running_entities)

    def add_message(
        self,
        role: str,
        content: str,
        mode: Optional[str] = None,
        language: Optional[str] = None,
    ) -> ChatMessage:
        """
        Append a message to this session's history and enforce memory limit.
        Truncates the oldest user/assistant turns if the limit is exceeded.
        """
        active_mode = mode or self.mode
        active_lang = language or self.last_language or "en-IN"
        self.last_language = active_lang

        msg = ChatMessage(role=role, content=content, mode=active_mode, language=active_lang)
        self.messages.append(msg)
        self.last_updated_at = datetime.now(timezone.utc)

        # Enforce maximum message history (each turn is ~2 messages: user + assistant)
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            excess = len(self.messages) - max_messages
            self.messages = self.messages[excess:]
            logger.debug(
                "Truncated %d oldest messages in conversation '%s' to stay within limit (%d)",
                excess,
                self.conversation_id,
                max_messages,
            )

        return msg

    def clear(self) -> None:
        """Clear all messages in this conversation."""
        self.messages.clear()
        self.recent_openers.clear()
        self.recent_topics.clear()
        self.running_entities.clear()
        self.last_updated_at = datetime.now(timezone.utc)

    def switch_mode(self, new_mode: str) -> None:
        """
        Switch the active personality mode for future turns
        while preserving all previous conversation history.
        """
        self.mode = new_mode
        self.last_updated_at = datetime.now(timezone.utc)

    def get_history(self) -> List[Dict[str, Any]]:
        """Return history as list of message dictionaries."""
        return [m.to_dict() for m in self.messages]


class ConversationManager:
    """
    Manages in-memory conversation sessions, ensuring strict isolation
    between conversations and thread-safe session retrieval.
    """

    def __init__(self, max_turns: Optional[int] = None) -> None:
        settings = get_settings()
        self.max_turns = max_turns or settings.MAX_CONVERSATION_TURNS
        self._sessions: Dict[str, ConversationSession] = {}

    def create_conversation(self, conversation_id: Optional[str] = None, mode: str = "unified_comedy") -> ConversationSession:
        """Create a new unique conversation session."""
        cid = conversation_id.strip() if conversation_id and conversation_id.strip() else f"conv_{uuid.uuid4().hex[:12]}"
        session = ConversationSession(conversation_id=cid, mode=mode, max_turns=self.max_turns)
        self._sessions[cid] = session
        logger.info("Created new conversation session '%s' with mode '%s'", cid, mode)
        return session

    def get_conversation(self, conversation_id: str) -> Optional[ConversationSession]:
        """Retrieve an existing conversation by ID without creating a new one."""
        if not conversation_id:
            return None
        return self._sessions.get(conversation_id.strip())

    def get_or_create_conversation(self, conversation_id: Optional[str] = None, mode: str = "unified_comedy") -> ConversationSession:
        """Retrieve existing conversation or create a new one if not found."""
        if conversation_id and conversation_id.strip() in self._sessions:
            return self._sessions[conversation_id.strip()]
        return self.create_conversation(conversation_id=conversation_id, mode=mode)

    def add_user_message(self, conversation_id: str, content: str, language: Optional[str] = None) -> ChatMessage:
        """Add a user message turn."""
        session = self.get_or_create_conversation(conversation_id)
        return session.add_message(role="user", content=content, language=language)

    def add_assistant_message(self, conversation_id: str, content: str, language: Optional[str] = None) -> ChatMessage:
        """Add an assistant response turn."""
        session = self.get_or_create_conversation(conversation_id)
        return session.add_message(role="assistant", content=content, language=language)

    def switch_mode(self, conversation_id: str, mode: str) -> ConversationSession:
        """Switch active mode on a session while keeping past messages."""
        session = self.get_or_create_conversation(conversation_id, mode=mode)
        session.switch_mode(mode)
        return session

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear message history for a conversation."""
        session = self.get_conversation(conversation_id)
        if session:
            session.clear()
            logger.info("Cleared conversation history for '%s'", conversation_id)
            return True
        return False

    def get_history_turns(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Return chronological conversation turns without prepended system prompts.
        Each turn contains 'role' ('user' or 'assistant') and 'content'.
        """
        session = self.get_conversation(conversation_id)
        if not session:
            return []
        return [{"role": msg.role, "content": msg.content} for msg in session.messages]

    def get_formatted_messages(self, conversation_id: str, system_prompt: str) -> List[Dict[str, str]]:
        """
        Format the full conversational prompt sequence for LLM inference:
        System prompt at the head, followed by user/assistant chronological turns.
        """
        session = self.get_conversation(conversation_id)
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if session:
            for msg in session.messages:
                messages.append({"role": msg.role, "content": msg.content})
        return messages

    def get_session_count(self) -> int:
        """Return the number of active in-memory conversation sessions."""
        return len(self._sessions)
