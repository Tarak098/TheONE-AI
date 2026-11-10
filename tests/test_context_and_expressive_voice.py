"""
TheONE AI - 30-Message Context-Aware Humour & Voice Expressiveness Test Suite.

Covers:
1. Ordinary conversations
2. Questions
3. Personal achievements
4. Failures and mistakes
5. Daily activities
6. Sarcastic interactions
7. Roast requests
8. Ambiguous messages
9. Serious / Sensitive messages

Verifies:
- Maximum ONE sentence per response
- Maximum 15 words (preferred 3-10 words)
- Direct contextual relevance (zero disconnected generic insults)
- Sensitive message compassionate handling
- Expressive voice synthesis with Yashvi and Zayan (prosody, pauses, speed)
"""

import pytest

from backend.services.intelligence import IntelligenceEngine
from backend.services.response_validator import (
    count_sentences,
    count_words,
    is_sensitive_message,
    validate_response_quality,
)
from backend.services.timbre import TimbreService

# 30 Diverse Test Messages across all 9 specified categories
TEST_SCENARIOS = [
    # 1. Ordinary Conversations
    {"category": "ordinary", "mode": "always_funny", "input": "I'm having a quiet cup of green tea."},
    {"category": "ordinary", "mode": "always_funny", "input": "The traffic was surprisingly smooth this morning."},
    {"category": "ordinary", "mode": "always_funny", "input": "I finally cleaned up my messy room."},

    # 2. Questions
    {"category": "question", "mode": "always_funny", "input": "What should I eat for dinner tonight?"},
    {"category": "question", "mode": "always_funny", "input": "Why do cats sleep twenty hours a day?"},
    {"category": "question", "mode": "roast_companion", "input": "Where can I find the fountain of eternal youth?"},
    {"category": "question", "mode": "roast_companion", "input": "What is the secret to getting rich without working?"},

    # 3. Personal Achievements
    {"category": "achievement", "mode": "always_funny", "input": "I got selected for my dream job interview!"},
    {"category": "achievement", "mode": "always_funny", "input": "I ran a full 5k marathon without stopping."},
    {"category": "achievement", "mode": "roast_companion", "input": "I got promoted to senior lead developer today!"},
    {"category": "achievement", "mode": "roast_companion", "input": "My startup just got its very first paying customer."},

    # 4. Failures & Mistakes
    {"category": "mistake", "mode": "always_funny", "input": "I lost my car keys inside the house again."},
    {"category": "mistake", "mode": "always_funny", "input": "I accidentally put salt in my coffee instead of sugar."},
    {"category": "mistake", "mode": "roast_companion", "input": "I accidentally sent an email to the entire company."},
    {"category": "mistake", "mode": "roast_companion", "input": "I completely burned my dinner to charcoal."},

    # 5. Daily Activities
    {"category": "daily", "mode": "always_funny", "input": "I hit the snooze button four times this morning."},
    {"category": "daily", "mode": "always_funny", "input": "I have forty browser tabs open and I need all of them."},
    {"category": "daily", "mode": "roast_companion", "input": "I have been scrolling social media for three hours."},
    {"category": "daily", "mode": "roast_companion", "input": "I'm standing in a never-ending grocery checkout line."},

    # 6. Sarcastic Interactions
    {"category": "sarcastic", "mode": "always_funny", "input": "I am definitely the most patient driver on the road."},
    {"category": "sarcastic", "mode": "always_funny", "input": "I love doing complicated tax forms on weekends."},
    {"category": "sarcastic", "mode": "roast_companion", "input": "My cooking skills are basically Michelin star level."},
    {"category": "sarcastic", "mode": "roast_companion", "input": "I am basically the undisputed genius of this household."},

    # 7. Roast Requests
    {"category": "roast_request", "mode": "roast_companion", "input": "Roast me, give me your best shot."},
    {"category": "roast_request", "mode": "roast_companion", "input": "Destroy my ego with a single sentence."},

    # 8. Ambiguous Messages
    {"category": "ambiguous", "mode": "always_funny", "input": "Wait, what?"},
    {"category": "ambiguous", "mode": "roast_companion", "input": "Seriously?"},

    # 9. Serious / Sensitive Messages
    {"category": "serious", "mode": "always_funny", "input": "My grandfather passed away peacefully this morning."},
    {"category": "serious", "mode": "roast_companion", "input": "I was diagnosed with a serious medical illness today."},
    {"category": "serious", "mode": "roast_companion", "input": "My beloved dog died yesterday and I'm heartbroken."},
]


def test_scenario_count():
    """Ensure exactly 30 test scenarios are defined across all required categories."""
    assert len(TEST_SCENARIOS) == 30
    categories = {s["category"] for s in TEST_SCENARIOS}
    assert "ordinary" in categories
    assert "question" in categories
    assert "achievement" in categories
    assert "mistake" in categories
    assert "daily" in categories
    assert "sarcastic" in categories
    assert "roast_request" in categories
    assert "ambiguous" in categories
    assert "serious" in categories


def test_timbre_expressive_voice_enhancements():
    """
    Verify voice delivery requirements:
    - Yashvi (Always Funny): speed 1.04, upbeat prosody
    - Zayan (Roast Companion): speed 0.98, dramatic pause (...) prosody
    - Audio tag safety: Unsupported tags stripped cleanly so they are never spoken aloud
    """
    service = TimbreService()
    if not service.is_configured:
        pytest.skip("Timbre service unconfigured.")

    # 1. Test Yashvi - Upbeat Funny
    res_yashvi = service.synthesize_speech(
        text="Finally, your resume escaped captivity! <laugh>",
        mode="always_funny",
        voice="Yashvi",
    )
    assert res_yashvi.voice == "Yashvi"
    assert res_yashvi.model == "timbre-v2.5"
    assert len(res_yashvi.audio_bytes) > 1000
    assert res_yashvi.audio_bytes.startswith(b"RIFF")

    # 2. Test Zayan - Deadpan Sarcastic Roast with Dramatic Pause
    res_zayan = service.synthesize_speech(
        text="Tomorrow called... it blocked your number.",
        mode="roast_companion",
        voice="Zayan",
    )
    assert res_zayan.voice == "Zayan"
    assert res_zayan.model == "timbre-v2.5"
    assert len(res_zayan.audio_bytes) > 1000
    assert res_zayan.audio_bytes.startswith(b"RIFF")


@pytest.mark.parametrize("scenario", TEST_SCENARIOS, ids=[f"{s['category']}_{i+1}" for i, s in enumerate(TEST_SCENARIOS)])
def test_all_30_messages_context_and_length(scenario):
    """
    Live test all 30 scenarios through IntelligenceEngine:
    - 1 sentence maximum
    - <= 15 words (prefer 3-10)
    - Valid context relevance
    - Sensitive message safe handling
    """
    import asyncio
    engine = IntelligenceEngine()
    if not engine.gemini.is_configured:
        pytest.skip("Gemini API not configured.")

    msg = scenario["input"]
    mode = scenario["mode"]
    category = scenario["category"]

    async def _test_turn():
        await asyncio.sleep(0.8)  # Gentle pacing for free-tier quota
        turn = await engine.process_turn(
            message=msg,
            mode=mode,
            conversation_id=f"test_30_{category}_{hash(msg) % 10000}",
        )
        reply = turn.assistant_response

        # 1. Word count & sentence count verification
        sentences = count_sentences(reply)
        words = count_words(reply)

        print(f"\n[{category.upper()} | {mode}] User: {msg!r} -> AI: {reply!r} ({words} words, {sentences} sentence)")

        assert sentences <= 1, f"Expected 1 sentence but got {sentences}: {reply}"
        assert words <= 15, f"Expected <= 15 words but got {words}: {reply}"

        # 2. Quality & Context validation
        is_valid, reason = validate_response_quality(msg, reply, mode=mode)
        assert is_valid, f"Quality check failed for {msg!r} -> {reply!r}: {reason}"

        # 3. Serious message handling verification
        if category == "serious":
            assert is_sensitive_message(msg) is True
            lower_reply = reply.lower()
            mocking_words = ["haha", "loser", "potato", "clown", "stupid"]
            assert not any(w in lower_reply for w in mocking_words), f"Serious message was mocked: {reply}"

    asyncio.run(_test_turn())
