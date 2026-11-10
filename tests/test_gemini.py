"""
Unit and Integration Tests for LaughBot AI — Step 3: Google Gemini API Integration.

Tests:
1. Gemini Configuration Loading & Validation
2. Missing API Key Handling (GeminiServiceUnavailableError)
3. Valid Provider Selection (LLM_PROVIDER='gemini')
4. System Prompt Loading & Verification (Always Funny & Roast Companion)
5. Always Funny Mode Generation (Mocked GenAI Client)
6. Roast Companion Mode Generation (Mocked GenAI Client)
7. Multi-Turn Conversation History Mapping ('user' -> 'user', 'assistant' -> 'model')
8. Personality Switching with Context Preservation
9. Conversation Memory Clearing
10. Gemini API Error Handling (APIError -> GeminiAPIError)
11. Gemini Rate Limit / Quota Handling (429 / RESOURCE_EXHAUSTED -> GeminiRateLimitError)
12. Gemini Timeout Handling (asyncio.TimeoutError -> GeminiTimeoutError)
13. Invalid / Empty Gemini Response Handling (GeminiInvalidResponseError)
14. FastAPI /api/v1/chat Endpoint with Gemini Provider (Mocked)
15. Live Google Gemini API Verification (Live API Key)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
import pytest
from fastapi.testclient import TestClient
from google.genai import errors, types

from backend.config import Settings, get_settings
from backend.main import app
from backend.services.conversation import ConversationManager
from backend.services.gemini import (
    GeminiAPIError,
    GeminiConfigError,
    GeminiInvalidResponseError,
    GeminiRateLimitError,
    GeminiService,
    GeminiServiceUnavailableError,
    GeminiTimeoutError,
)
from backend.services.intelligence import (
    IntelligenceEngine,
    IntelligenceValidationError,
)
from backend.services.personality import PersonalityManager

client = TestClient(app)


# ------------------------------------------------------------------------------
# 1. Configuration & Validation Tests
# ------------------------------------------------------------------------------
def test_gemini_config_loading():
    """Verify Gemini settings are loaded from configuration."""
    settings = get_settings()
    assert settings.LLM_PROVIDER == "gemini"
    assert settings.active_llm_provider == "gemini"
    assert settings.GEMINI_MODEL == "gemma-4-26b-a4b-it"
    assert settings.GEMINI_TIMEOUT >= 1
    assert settings.GEMINI_MAX_OUTPUT_TOKENS >= 10


def test_gemini_missing_api_key_raises_unavailable():
    """Verify GeminiServiceUnavailableError is raised when API key is missing."""
    mock_settings = Settings(GEMINI_API_KEY=None, LLM_PROVIDER="gemini", GEMINI_MODEL="gemma-4-26b-a4b-it")
    service = GeminiService(settings=mock_settings)

    assert not service.is_configured
    health = service.health_check()
    assert health["configured"] is False
    assert health["status"] == "not_configured"

    with pytest.raises(GeminiServiceUnavailableError, match="not configured"):
        service.validate_configuration()

    with pytest.raises(GeminiServiceUnavailableError, match="not configured"):
        asyncio.run(service.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
            system_instruction="Be funny",
        ))


def test_gemini_provider_selection():
    """Verify IntelligenceEngine routes to Gemini when LLM_PROVIDER='gemini'."""
    mock_settings = Settings(LLM_PROVIDER="gemini", GEMINI_API_KEY="test_key", GEMINI_MODEL="gemma-4-26b-a4b-it")
    mock_gemini = MagicMock(spec=GeminiService)
    mock_gemini.model = "gemma-4-26b-a4b-it"
    mock_gemini.generate_response = AsyncMock(return_value="Gemini funny response")

    engine = IntelligenceEngine(
        gemini_service=mock_gemini,
        settings=mock_settings,
    )

    assert engine.active_provider == "gemini"
    assert engine.active_model == "gemma-4-26b-a4b-it"

    result = asyncio.run(engine.process_turn(
        message="I spilled coffee",
        mode="always_funny",
    ))

    assert result.provider == "gemini"
    assert result.model == "gemma-4-26b-a4b-it"
    assert result.assistant_response == "Gemini funny response"
    assert mock_gemini.generate_response.called


# ------------------------------------------------------------------------------
# 2. Personality System Prompt & Generation Tests (Mocked SDK)
# ------------------------------------------------------------------------------
def test_gemini_always_funny_generation_mocked():
    """Verify Always Funny mode passes correct system prompt and returns comedy text."""
    mock_client = MagicMock()
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Your alarm clock just filed for retirement! 😂"
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    service = GeminiService(client=mock_client)
    pm = PersonalityManager()
    system_prompt = pm.get_system_prompt("always_funny")

    reply = asyncio.run(service.generate_response(
        messages=[{"role": "user", "content": "I woke up late."}],
        system_instruction=system_prompt,
    ))

    assert reply == "Your alarm clock just filed for retirement! 😂"
    assert mock_client.aio.models.generate_content.called

    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].system_instruction == system_prompt
    assert len(call_kwargs["contents"]) == 1
    assert call_kwargs["contents"][0].role == "user"


def test_gemini_roast_companion_generation_mocked():
    """Verify Roast Companion mode passes roast system prompt and returns burn."""
    mock_client = MagicMock()
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Your code has more bugs than an entomology convention. 🔥"
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    service = GeminiService(client=mock_client)
    pm = PersonalityManager()
    system_prompt = pm.get_system_prompt("roast_companion")

    reply = asyncio.run(service.generate_response(
        messages=[{"role": "user", "content": "I code in Python."}],
        system_instruction=system_prompt,
    ))

    assert reply == "Your code has more bugs than an entomology convention. 🔥"
    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].system_instruction == system_prompt


# ------------------------------------------------------------------------------
# 3. Conversation Memory & Multi-Turn History
# ------------------------------------------------------------------------------
def test_gemini_multi_turn_history_mapping():
    """Verify conversation history maps user -> 'user' and assistant -> 'model'."""
    mock_client = MagicMock()
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Yes, you like neon green!"
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    service = GeminiService(client=mock_client)
    cm = ConversationManager()
    session = cm.create_conversation("conv_history_test", mode="always_funny")
    session.add_message("user", "My favorite color is neon green.")
    session.add_message("assistant", "Neon green is so bright it illuminates space.")

    history_turns = cm.get_history_turns(session.conversation_id)
    candidate_turns = history_turns + [{"role": "user", "content": "What color do I like?"}]

    reply = asyncio.run(service.generate_response(
        messages=candidate_turns,
        system_instruction="Be funny",
    ))

    assert reply == "Yes, you like neon green!"
    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    contents = call_kwargs["contents"]

    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "My favorite color is neon green."
    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "Neon green is so bright it illuminates space."
    assert contents[2].role == "user"
    assert contents[2].parts[0].text == "What color do I like?"


def test_gemini_personality_switching_preserves_history():
    """Verify switching personality maintains multi-turn history."""
    cm = ConversationManager()
    session = cm.create_conversation("conv_switch_test", mode="always_funny")
    session.add_message("user", "I am tired.")
    session.add_message("assistant", "Coffee is waiting for you.")

    # Switch personality
    session.switch_mode("roast_companion")
    assert session.mode == "roast_companion"

    # Verify history is preserved intact
    history = cm.get_history_turns(session.conversation_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_gemini_conversation_clearing():
    """Verify conversation history clearing."""
    cm = ConversationManager()
    session = cm.create_conversation("conv_clear_test", mode="always_funny")
    session.add_message("user", "Hello")
    session.add_message("assistant", "Hi there")

    assert len(session.messages) == 2
    assert cm.clear_conversation("conv_clear_test") is True
    assert len(session.messages) == 0


# ------------------------------------------------------------------------------
# 4. Error Handling Tests (Mocked API Exceptions)
# ------------------------------------------------------------------------------
def test_gemini_rate_limit_error_handling():
    """Verify HTTP 429 / RESOURCE_EXHAUSTED is mapped to GeminiRateLimitError."""
    mock_client = MagicMock()
    api_error = errors.APIError(
        429,
        {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
    )
    mock_client.aio.models.generate_content = AsyncMock(side_effect=api_error)

    service = GeminiService(client=mock_client)

    with pytest.raises(GeminiRateLimitError, match="quota or rate limit exceeded"):
        asyncio.run(service.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
            system_instruction="Be funny",
        ))


def test_gemini_timeout_error_handling():
    """Verify timeout translates to GeminiTimeoutError."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=asyncio.TimeoutError("Timeout"))

    service = GeminiService(client=mock_client)

    with pytest.raises(GeminiTimeoutError, match="timed out"):
        asyncio.run(service.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
            system_instruction="Be funny",
        ))


def test_gemini_api_error_handling():
    """Verify general APIError translates to GeminiAPIError."""
    mock_client = MagicMock()
    api_error = errors.APIError(
        500,
        {"error": {"code": 500, "message": "Internal Server Error in Gemini backend"}},
    )
    mock_client.aio.models.generate_content = AsyncMock(side_effect=api_error)

    service = GeminiService(client=mock_client)

    with pytest.raises(GeminiAPIError, match="Google Gemini API error"):
        asyncio.run(service.generate_response(
            messages=[{"role": "user", "content": "Hello"}],
            system_instruction="Be funny",
        ))


def test_gemini_empty_response_handling():
    """Verify empty or blocked text raises GeminiInvalidResponseError."""
    mock_client = MagicMock()
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = None
    mock_response.candidates = [MagicMock(finish_reason="SAFETY")]
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    service = GeminiService(client=mock_client)

    with pytest.raises(GeminiInvalidResponseError, match="empty or filtered"):
        asyncio.run(service.generate_response(
            messages=[{"role": "user", "content": "Inappropriate query"}],
            system_instruction="Be funny",
        ))


# ------------------------------------------------------------------------------
# 5. FastAPI Endpoints Integration Tests (Mocked)
# ------------------------------------------------------------------------------
def test_chat_endpoint_gemini_response_structure():
    """Verify POST /api/v1/chat returns provider='gemini' and model name."""
    with patch.object(GeminiService, "is_configured", new_callable=PropertyMock, return_value=True):
        with patch.object(
            GeminiService,
            "generate_response",
            new=AsyncMock(return_value="Why did the AI cross the road? To optimize the path! 😂"),
        ):
            res = client.post(
                "/api/v1/chat",
                json={
                    "message": "Tell me a joke",
                    "mode": "always_funny",
                    "language": "en",
                },
            )
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert data["provider"] == "gemini"
            assert data["model"] == "gemma-4-26b-a4b-it"
            assert data["mode"] == "always_funny"
            assert "Why did the AI" in data["assistant_response"]
            assert "conversation_id" in data


def test_api_status_reports_gemini():
    """Verify GET /api/v1/status reports Gemini service status and active provider."""
    res = client.get("/api/v1/status")
    assert res.status_code == 200
    data = res.json()
    assert data["active_provider"] == "gemini"
    assert "gemini" in data["services"]
    assert "prisma" in data["services"]
    assert "evon" in data["services"]


# ------------------------------------------------------------------------------
# 6. Live Google Gemma API Test (Real API Credentials)
# ------------------------------------------------------------------------------
def test_live_gemma_api_call_success():
    """
    Test live communication with Google's Gemma 4 26B A4B model using configured GEMINI_API_KEY.
    Verifies that real API responses are returned directly from gemma-4-26b-a4b-it.
    """
    service = GeminiService()
    if not service.is_configured:
        pytest.skip("GEMINI_API_KEY is not configured in environment.")

    reply = asyncio.run(service.generate_response(
        messages=[{"role": "user", "content": "Tell a short one-liner joke about computers."}],
        system_instruction="You are LaughBot in Always Funny mode.",
    ))
    assert reply is not None
    assert len(reply.strip()) > 0
    assert service.model == "gemma-4-26b-a4b-it"
    print(f"\n[LIVE GEMMA SUCCESS] Model: {service.model}, Response: {reply!r}")
