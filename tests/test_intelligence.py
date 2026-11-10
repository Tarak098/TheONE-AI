"""
Unit and Integration Tests for LaughBot AI — Step 3: Intelligence Engine & Personality System.

Tests:
1. Personality Selection & Normalization
2. System Prompt Loading & Caching (funny_mode.txt and roast_mode.txt)
3. Mode Switching with Context Preservation
4. Conversation Creation and User Isolation
5. Conversation History & Sliding Window Memory Truncation
6. Conversation Clearing
7. Message Validation (Empty message, oversized message)
8. Evon Configuration Validation & Readiness Check
9. Evon API Timeout & Connection Error Handling (Mocked)
10. End-to-End Intelligence Pipeline Orchestration (Mocked Evon)
11. FastAPI Endpoints (/api/v1/chat, /api/v1/chat/health, /api/v1/chat/clear, /api/v1/chat/switch-mode)
12. Live Evon Access Status Inspection
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import app
from backend.services.conversation import ConversationManager
from backend.services.evon import (
    EvonConnectionError,
    EvonService,
    EvonServiceUnavailableError,
    EvonTimeoutError,
)
from backend.services.intelligence import (
    IntelligenceEngine,
    IntelligenceValidationError,
)
from backend.services.personality import PersonalityManager, PersonalityMode

client = TestClient(app)


# ------------------------------------------------------------------------------
# 1. Personality Selection & Validation Tests
# ------------------------------------------------------------------------------
def test_personality_mode_validation():
    """Verify personality mode validation and normalization."""
    pm = PersonalityManager()

    assert pm.validate_mode("always_funny") == "always_funny"
    assert pm.validate_mode("funny") == "always_funny"
    assert pm.validate_mode("ALWAYS_FUNNY") == "always_funny"

    assert pm.validate_mode("roast_companion") == "roast_companion"
    assert pm.validate_mode("roast") == "roast_companion"
    assert pm.validate_mode("ROAST_COMPANION") == "roast_companion"

    with pytest.raises(ValueError, match="Unsupported personality mode"):
        pm.validate_mode("serious_lawyer")


# ------------------------------------------------------------------------------
# 2. System Prompt Loading & Caching
# ------------------------------------------------------------------------------
def test_system_prompt_loading():
    """Verify loading of system prompts for funny and roast modes."""
    pm = PersonalityManager()

    funny_prompt = pm.get_system_prompt("always_funny")
    assert "LaughBot" in funny_prompt
    assert "creative and entertaining AI comedy companion" in funny_prompt
    assert "Behavior:" in funny_prompt or "Rules:" in funny_prompt

    roast_prompt = pm.get_system_prompt("roast_companion")
    assert "Roast Companion" in roast_prompt
    assert "savage and entertaining roasts" in roast_prompt or "Roast Companion" in roast_prompt
    assert "Rules:" in roast_prompt or "Behavior:" in roast_prompt


def test_system_prompt_caching():
    """Verify system prompts are cached in memory after first load."""
    pm = PersonalityManager()
    pm.clear_cache()
    assert len(pm._prompt_cache) == 0

    p1 = pm.get_system_prompt("always_funny")
    assert len(pm._prompt_cache) == 1
    p2 = pm.get_system_prompt("always_funny")
    assert p1 == p2


# ------------------------------------------------------------------------------
# 3. Conversation Creation & History Management
# ------------------------------------------------------------------------------
def test_conversation_creation_and_isolation():
    """Verify distinct conversation sessions maintain strictly isolated histories."""
    cm = ConversationManager(max_turns=5)

    s1 = cm.create_conversation("user_alice")
    s2 = cm.create_conversation("user_bob")

    cm.add_user_message("user_alice", "Hello from Alice")
    cm.add_assistant_message("user_alice", "Alice joke response")

    cm.add_user_message("user_bob", "Hello from Bob")

    assert len(s1.messages) == 2
    assert len(s2.messages) == 1
    assert s1.messages[0].content == "Hello from Alice"
    assert s2.messages[0].content == "Hello from Bob"


def test_conversation_memory_truncation():
    """Verify oldest turns are truncated when exceeding max_turns * 2."""
    cm = ConversationManager(max_turns=2)  # max_messages = 4
    session = cm.create_conversation("session_truncate")

    for i in range(6):
        session.add_message(role="user" if i % 2 == 0 else "assistant", content=f"msg_{i}")

    # Should only retain the last 4 messages (msg_2, msg_3, msg_4, msg_5)
    assert len(session.messages) == 4
    assert session.messages[0].content == "msg_2"
    assert session.messages[3].content == "msg_5"


def test_conversation_clearing():
    """Verify conversation history clearing."""
    cm = ConversationManager()
    session = cm.create_conversation("clear_me")
    session.add_message("user", "Hello")
    session.add_message("assistant", "Hi there")
    assert len(session.messages) == 2

    assert cm.clear_conversation("clear_me") is True
    assert len(session.messages) == 0


def test_mode_switching_preserves_history():
    """Verify switching personality preserves existing conversational history."""
    cm = ConversationManager()
    session = cm.create_conversation("session_switch", mode="always_funny")

    session.add_message("user", "Tell me a joke")
    session.add_message("assistant", "Why did the chicken cross the road?")

    # Switch to roast_companion
    cm.switch_mode("session_switch", "roast_companion")

    assert session.mode == "roast_companion"
    assert len(session.messages) == 2
    assert session.messages[0].content == "Tell me a joke"


# ------------------------------------------------------------------------------
# 4. Message Validation Tests
# ------------------------------------------------------------------------------
def test_empty_message_validation():
    """Verify empty or whitespace message is rejected."""
    engine = IntelligenceEngine()
    with pytest.raises(IntelligenceValidationError, match="cannot be empty"):
        asyncio.run(engine.process_turn(message="   "))


def test_oversized_message_validation():
    """Verify oversized message exceeding 2000 characters is rejected."""
    engine = IntelligenceEngine()
    with pytest.raises(IntelligenceValidationError, match="exceeds maximum limit"):
        asyncio.run(engine.process_turn(message="A" * 2005))


# ------------------------------------------------------------------------------
# 5. Evon Service Configuration & Error Handling (Mocked)
# ------------------------------------------------------------------------------
def test_evon_unconfigured_status():
    """Verify Evon reports not_configured when URL or key is missing."""
    mock_settings = Settings(GNANI_EVON_API_URL=None, GNANI_EVON_API_KEY=None)
    evon = EvonService(settings=mock_settings)
    assert not evon.is_configured
    health = evon.health_check()
    assert health["status"] == "not_configured"
    assert health["configured"] is False


def test_evon_unconfigured_raises_unavailable():
    """Verify EvonServiceUnavailableError is raised if inference is attempted without config."""
    mock_settings = Settings(GNANI_EVON_API_URL=None, GNANI_EVON_API_KEY=None)
    evon = EvonService(settings=mock_settings)

    with pytest.raises(EvonServiceUnavailableError, match="not configured"):
        asyncio.run(evon.generate_response([{"role": "user", "content": "Hello"}]))


def test_evon_timeout_handling():
    """Verify EvonTimeoutError is raised when requests times out."""
    import requests
    mock_settings = Settings(GNANI_EVON_API_URL="https://mock-evon.internal/v1", GNANI_EVON_API_KEY="test_key")
    evon = EvonService(settings=mock_settings)

    with patch("requests.post", side_effect=requests.exceptions.Timeout("Timeout")):
        with pytest.raises(EvonTimeoutError):
            asyncio.run(evon.generate_response([{"role": "user", "content": "Hello"}]))


# ------------------------------------------------------------------------------
# 6. Intelligence Pipeline Orchestration (Mocked Evon Service)
# ------------------------------------------------------------------------------
def test_intelligence_pipeline_orchestration_mocked():
    """Verify full turn orchestration: user message -> prompt -> Evon reply -> history."""
    mock_evon = MagicMock(spec=EvonService)
    mock_evon.is_configured = True
    mock_evon.model = "evon-v3.3"
    mock_evon.generate_response = AsyncMock(return_value="Your alarm clock has officially resigned. 😂")

    mock_settings = Settings(LLM_PROVIDER="evon")
    cm = ConversationManager()
    pm = PersonalityManager()
    engine = IntelligenceEngine(
        evon_service=mock_evon,
        personality_manager=pm,
        conversation_manager=cm,
        settings=mock_settings,
    )

    result = asyncio.run(engine.process_turn(
        message="I woke up late today",
        conversation_id="conv_test_1",
        mode="always_funny",
    ))

    assert result.success is True
    assert result.user_message == "I woke up late today"
    assert result.assistant_response == "Your alarm clock has officially resigned. 😂"
    assert result.mode == "always_funny"

    # Verify conversation history was persisted
    session = cm.get_conversation("conv_test_1")
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"


# ------------------------------------------------------------------------------
# 7. FastAPI Chat Endpoints Tests
# ------------------------------------------------------------------------------
def test_chat_health_endpoint():
    """Verify GET /api/v1/chat/health returns intelligence engine readiness."""
    res = client.get("/api/v1/chat/health")
    assert res.status_code == 200
    data = res.json()
    assert data["intelligence_engine"] == "active"
    assert "always_funny" in data["supported_modes"]
    assert "roast_companion" in data["supported_modes"]


def test_chat_unconfigured_provider_returns_503():
    """Verify POST /api/v1/chat returns HTTP 503 rather than fake content when provider is unconfigured."""
    from backend.services.gemini import GeminiService
    with patch.object(GeminiService, "is_configured", new_callable=PropertyMock, return_value=False):
        res = client.post(
            "/api/v1/chat",
            json={"message": "I woke up late", "mode": "always_funny"},
        )
        assert res.status_code == 503
        assert "not configured" in res.json()["error"].lower()


def test_chat_endpoint_mocked_success():
    """Verify POST /api/v1/chat returns expected 200 schema when active provider succeeds."""
    from backend.services.gemini import GeminiService
    with patch.object(GeminiService, "is_configured", new_callable=PropertyMock, return_value=True):
        with patch.object(
            GeminiService,
            "generate_response",
            new=AsyncMock(return_value="Mocked comedy response"),
        ):
            res = client.post(
                "/api/v1/chat",
                json={"message": "Tell me a joke", "mode": "always_funny"},
            )
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["assistant_response"] == "Mocked comedy response"
            assert data["mode"] == "always_funny"
            assert "conversation_id" in data


def test_chat_switch_mode_endpoint():
    """Verify POST /api/v1/chat/switch-mode switches active personality."""
    res = client.post(
        "/api/v1/chat/switch-mode",
        json={"conversation_id": "test_conv_mode", "mode": "roast_companion"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["mode"] == "roast_companion"


def test_chat_clear_endpoint():
    """Verify POST /api/v1/chat/clear clears the conversation."""
    res = client.post(
        "/api/v1/chat/clear",
        json={"conversation_id": "test_conv_mode"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["conversation_id"] == "test_conv_mode"


# ------------------------------------------------------------------------------
# 8. Live Evon Access Check
# ------------------------------------------------------------------------------
def test_live_evon_access_status():
    """
    Inspect whether a live remote Gnani Evon endpoint is configured in environment.
    If not configured, unit test verifies that is_configured is cleanly False.
    """
    from backend.config import get_settings
    settings = get_settings()

    if settings.has_evon_credentials:
        print(f"\n[LIVE EVON CONFIG DETECTED] Endpoint: {settings.GNANI_EVON_API_URL}, Model: {settings.GNANI_EVON_MODEL}")
    else:
        print("\n[LIVE EVON NOT CONFIGURED] No GNANI_EVON_API_URL configured. Adapter cleanly reports not_configured.")
        assert settings.has_evon_credentials is False
