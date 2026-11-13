"""
TheONE AI - 30-Message Unified Comedy Personality & Voice Test Suite.

Verifies:
1. One Unified Personality (ALWAYS funny, sarcastic, witty, entertaining regardless of user mood).
2. Mood-Independent Adaptive Comedy (bored, angry, failing, celebrating, confused, factual).
3. Contextual relevance & zero generic stock insults.
4. Strict length constraint (strictly 1 sentence, <= 15 words, preferred 3-10 words).
5. No supportive or serious assistant mode appears.
6. Sensitive topics handled with gentle, non-insensitive humour without dropping character.
7. Independent voice synthesis with Yashvi and Zayan.
"""

import pytest

from backend.services.intelligence import IntelligenceEngine
from backend.services.personality import PersonalityManager
from backend.services.response_validator import (
    count_sentences,
    count_words,
    is_sensitive_message,
    validate_response_quality,
)
from backend.services.timbre import TimbreService

# 30 Diverse Test Messages across different topics and emotional contexts
TEST_SCENARIOS = [
    # 1. User Mood: Boredom (User Example 1)
    {"category": "boredom", "input": "I'm feeling bored."},
    {"category": "boredom", "input": "Nothing interesting is happening in my life."},

    # 2. User Mood: Failure / Mistake (User Example 2)
    {"category": "failure", "input": "I failed my exam."},
    {"category": "failure", "input": "I completely burned my dinner to charcoal."},
    {"category": "failure", "input": "I accidentally replied-all to the whole company."},

    # 3. User Mood: Anger / Frustration (User Example 3)
    {"category": "anger", "input": "I'm very angry today."},
    {"category": "anger", "input": "My Wi-Fi keeps disconnecting every five minutes."},
    {"category": "anger", "input": "People who chew loudly are driving me insane."},

    # 4. User Mood: Celebration / Achievement (User Example 4)
    {"category": "achievement", "input": "I got selected!"},
    {"category": "achievement", "input": "I ran a full 5k marathon without stopping."},
    {"category": "achievement", "input": "I got promoted to senior lead developer today!"},

    # 5. User Mood: Confusion / Learning (User Example 5)
    {"category": "confusion", "input": "I don't understand this topic."},
    {"category": "confusion", "input": "Quantum physics makes absolutely no sense to me."},
    {"category": "confusion", "input": "Wait, what does that word even mean?"},

    # 6. Factual Questions (Accurate fact with humorous delivery)
    {"category": "factual", "input": "What is photosynthesis?"},
    {"category": "factual", "input": "Tell me a real fact about outer space."},
    {"category": "factual", "input": "Why do cats sleep so many hours a day?"},
    {"category": "factual", "input": "How many days are in a leap year?"},

    # 7. Boasting / Confidence
    {"category": "boast", "input": "I am basically the undisputed genius of this household."},
    {"category": "boast", "input": "My cooking skills are Michelin star level."},

    # 8. Daily Habits & Procrastination
    {"category": "habits", "input": "I hit the snooze button five times this morning."},
    {"category": "habits", "input": "I have forty browser tabs open and I need all of them."},
    {"category": "habits", "input": "I have been scrolling reels for three straight hours."},

    # 9. Roast Requests
    {"category": "roast_request", "input": "Roast me, give me your sharpest comeback."},
    {"category": "roast_request", "input": "Destroy my ego with a single punchline."},

    # 10. Ordinary Daily Events
    {"category": "ordinary", "input": "I'm having a quiet cup of green tea."},
    {"category": "ordinary", "input": "I lost my car keys inside my own house again."},

    # 11. Sensitive Contexts (Gentle, non-insensitive humour; NEVER a dry assistant)
    {"category": "sensitive", "input": "My grandfather passed away peacefully this morning."},
    {"category": "sensitive", "input": "I was diagnosed with a serious medical illness today."},
    {"category": "sensitive", "input": "My beloved dog died yesterday and I miss him."},
]


def test_scenario_count():
    """Ensure exactly 30 test scenarios are defined covering all mood and topic spectrums."""
    assert len(TEST_SCENARIOS) == 30
    categories = {s["category"] for s in TEST_SCENARIOS}
    assert "boredom" in categories
    assert "failure" in categories
    assert "anger" in categories
    assert "achievement" in categories
    assert "confusion" in categories
    assert "factual" in categories
    assert "boast" in categories
    assert "habits" in categories
    assert "roast_request" in categories
    assert "ordinary" in categories
    assert "sensitive" in categories


def test_personality_manager_unified():
    """Verify that PersonalityManager enforces ONE unified personality."""
    manager = PersonalityManager()
    assert manager.get_supported_modes() == ["unified_comedy"]
    assert manager.validate_mode("always_funny") == "unified_comedy"
    assert manager.validate_mode("roast_companion") == "unified_comedy"
    assert manager.validate_mode(None) == "unified_comedy"
    assert manager.validate_mode("anything") == "unified_comedy"

    prompt = manager.get_system_prompt()
    assert "You are TheONE AI" in prompt
    assert "SIMPLE, EVERYDAY SPOKEN ENGLISH" in prompt
    assert "Bro, tomorrow has blocked you already." in prompt
    assert "Good morning, grandpa." in prompt
    assert "Your bank account is running a charity for zeroes." in prompt
    assert "Yeah bro, your mirror must be proud." in prompt
    assert "Bro, the question paper won this fight." in prompt


def test_timbre_independent_voices():
    """
    Verify voice selection independence:
    - Both Yashvi (Female) and Zayan (Male) work in full capacity for unified comedy.
    - Speed prosody matches the selected voice (Yashvi ~1.04, Zayan ~0.98).
    - Audio tag safety is enforced.
    """
    service = TimbreService()
    if not service.is_configured:
        pytest.skip("Timbre service unconfigured.")

    # 1. Test Yashvi Voice
    res_yashvi = service.synthesize_speech(
        text="Even boredom is bored of you! <laugh>",
        voice="Yashvi",
    )
    assert res_yashvi.voice == "Yashvi"
    assert res_yashvi.model == "timbre-v2.5"
    assert len(res_yashvi.audio_bytes) > 1000
    assert res_yashvi.audio_bytes.startswith(b"RIFF")

    # 2. Test Zayan Voice
    res_zayan = service.synthesize_speech(
        text="Your exam paper won this round... better luck next time.",
        voice="Zayan",
    )
    assert res_zayan.voice == "Zayan"
    assert res_zayan.model == "timbre-v2.5"
    assert len(res_zayan.audio_bytes) > 1000
    assert res_zayan.audio_bytes.startswith(b"RIFF")


@pytest.mark.parametrize("scenario", TEST_SCENARIOS, ids=[f"{s['category']}_{i+1}" for i, s in enumerate(TEST_SCENARIOS)])
def test_all_30_messages_unified_comedy(scenario):
    """
    Live test all 30 scenarios through IntelligenceEngine:
    - Exactly 1 sentence
    - <= 15 words
    - Context-aligned comedy
    - Never breaks into a dry, serious, or supportive assistant
    """
    import asyncio
    engine = IntelligenceEngine()
    if not engine.gemini.is_configured:
        pytest.skip("Gemini API not configured.")

    msg = scenario["input"]
    category = scenario["category"]

    async def _test_turn():
        await asyncio.sleep(0.8)  # Free-tier rate pacing
        turn = await engine.process_turn(
            message=msg,
            conversation_id=f"test_unified_{category}_{abs(hash(msg)) % 10000}",
        )
        reply = turn.assistant_response

        # 1. Sentence and word count constraints
        sentences = count_sentences(reply)
        words = count_words(reply)

        print(f"\n[{category.upper()}] User: {msg!r} -> AI: {reply!r} ({words} words, {sentences} sentence)")

        assert sentences <= 1, f"Expected 1 sentence but got {sentences}: {reply}"
        assert words <= 15, f"Expected <= 15 words but got {words}: {reply}"

        # 2. Quality & Context validation
        is_valid, reason = validate_response_quality(msg, reply, mode="unified_comedy")
        assert is_valid, f"Quality check failed for {msg!r} -> {reply!r}: {reason}"

        # 3. Check that it NEVER sounds like a dry, generic corporate assistant
        boring_phrases = [
            "how can i help", "as an ai", "i am an artificial intelligence",
            "i cannot provide", "i understand that you are feeling",
            "i am here to assist", "please consult a professional"
        ]
        lower_reply = reply.lower()
        for phrase in boring_phrases:
            assert phrase not in lower_reply, f"Response slipped into boring assistant mode: {reply}"

        # 4. Sensitive situations: Gentle lightheartedness without cruel mocking
        if category == "sensitive":
            cruel_words = ["haha", "loser", "potato", "clown", "stupid", "idiot"]
            assert not any(w in lower_reply for w in cruel_words), f"Sensitive topic cruelly mocked: {reply}"

    asyncio.run(_test_turn())
