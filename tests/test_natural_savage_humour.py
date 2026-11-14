"""
TheONE AI - 50+ Natural, Savage & Human-Like Humour Test Suite.

Evaluates:
1. Simple Everyday English (no complicated, formal, or robotic words).
2. Natural Human Humour (spontaneous comebacks, relatable jokes, everyday sarcasm).
3. Savage Comedy Personality (playful roasting, bold and cheeky, never boring/polite).
4. Direct Contextual Relevance (anchored strictly to the user's specific situation).
5. Strict Length Constraint (strictly 1 sentence, <= 15 words, preferred 3-10 words).
6. Nine Distinct Life & Conversational Categories (54 total inputs).
"""

import asyncio
import re
from typing import Dict, List
import pytest

from backend.services.intelligence import IntelligenceEngine
from backend.services.personality import PersonalityManager
from backend.services.response_validator import (
    count_sentences,
    count_words,
    validate_response_quality,
)

# 54 Conversational Test Inputs across 9 specific categories
TEST_INPUTS_54: List[Dict[str, str]] = [
    # 1. Casual Conversations
    {"category": "casual", "input": "Hey, what are you doing right now?"},
    {"category": "casual", "input": "I'm feeling bored today."},
    {"category": "casual", "input": "It's way too hot outside."},
    {"category": "casual", "input": "I'm having iced coffee."},
    {"category": "casual", "input": "Just chilling on the couch doing nothing."},
    {"category": "casual", "input": "Tell me what is going on."},

    # 2. Daily Life Situations
    {"category": "daily_life", "input": "I woke up at 12."},
    {"category": "daily_life", "input": "I burned my dinner."},
    {"category": "daily_life", "input": "I lost my car keys inside my house."},
    {"category": "daily_life", "input": "I tripped over my own feet in public."},
    {"category": "daily_life", "input": "My phone battery died at 1% right when I needed it."},
    {"category": "daily_life", "input": "I stubbed my toe on the sofa."},

    # 3. Studies and Exams
    {"category": "studies", "input": "I'm going to study tomorrow."},
    {"category": "studies", "input": "I failed my exam."},
    {"category": "studies", "input": "My exam starts in one hour and I opened the book now."},
    {"category": "studies", "input": "Math homework is giving me a headache."},
    {"category": "studies", "input": "I fell asleep during the lecture."},
    {"category": "studies", "input": "I got a C on my assignment."},

    # 4. Friends and Relationships
    {"category": "relationships", "input": "Nobody is replying to my messages."},
    {"category": "relationships", "input": "My best friend left me on read."},
    {"category": "relationships", "input": "I sent a risky text to my crush and regret it."},
    {"category": "relationships", "input": "My friends went on a trip without inviting me."},
    {"category": "relationships", "input": "I'm third wheeling on a dinner date."},
    {"category": "relationships", "input": "My friend told me a terrible secret."},

    # 5. Procrastination
    {"category": "procrastination", "input": "I spent the whole day scrolling Instagram."},
    {"category": "procrastination", "input": "I will clean my room later this evening."},
    {"category": "procrastination", "input": "I've been binge watching Netflix for five hours."},
    {"category": "procrastination", "input": "I watched random shorts until 4 AM."},
    {"category": "procrastination", "input": "I opened my laptop to work and watched anime instead."},
    {"category": "procrastination", "input": "I hit the snooze button four times today."},

    # 6. Money
    {"category": "money", "input": "I have no money."},
    {"category": "money", "input": "I spent my entire salary on online shopping."},
    {"category": "money", "input": "I ordered expensive takeout again."},
    {"category": "money", "input": "I checked my bank balance and felt dizzy."},
    {"category": "money", "input": "Can you loan me five thousand rupees?"},
    {"category": "money", "input": "I bought sneakers I cannot afford."},

    # 7. Work and Productivity
    {"category": "work", "input": "I will start going to the gym."},
    {"category": "work", "input": "My boss scheduled another useless meeting."},
    {"category": "work", "input": "I accidentally sent an email with typos to my manager."},
    {"category": "work", "input": "I have Monday blues on a Wednesday."},
    {"category": "work", "input": "I've been staring at a blank Excel sheet for an hour."},
    {"category": "work", "input": "I got tired after working for twenty minutes."},

    # 8. User Challenges and Roast Requests
    {"category": "roast", "input": "I am very intelligent."},
    {"category": "roast", "input": "Roast me, give me your best shot."},
    {"category": "roast", "input": "I'm the coolest person in my office."},
    {"category": "roast", "input": "Destroy my confidence with one sentence."},
    {"category": "roast", "input": "You cannot say anything funny about me."},
    {"category": "roast", "input": "I think I am a 10 out of 10."},

    # 9. Unexpected Questions
    {"category": "questions", "input": "Why do cats sleep so much?"},
    {"category": "questions", "input": "What is the speed of light?"},
    {"category": "questions", "input": "Do aliens exist?"},
    {"category": "questions", "input": "Why is pizza round but the box is square?"},
    {"category": "questions", "input": "What came first, the chicken or the egg?"},
    {"category": "questions", "input": "Why do people wear sunglasses indoors?"},
]

# Unnatural, academic, or corporate buzzwords that a funny friend would NEVER use
FORBIDDEN_COMPLEX_WORDS = {
    "captivity", "buffering", "independence", "custody", "department",
    "photosynthesis", "carbon", "patron", "dimension", "utilize",
    "furthermore", "nonetheless", "henceforth", "facilitate", "optimal",
    "suboptimal", "algorithm", "consequently", "paramount", "patience department",
}

# Generic corporate assistant filler phrases that must NEVER appear
FORBIDDEN_ASSISTANT_PHRASES = [
    "how can i help", "as an ai", "i am an artificial intelligence",
    "i cannot provide", "i understand that you are feeling",
    "i am here to assist", "please consult a professional",
    "as a language model", "it is important to remember",
]


def test_suite_coverage():
    """Verify test suite contains at least 50 inputs spanning all 9 required categories."""
    assert len(TEST_INPUTS_54) >= 50
    assert len(TEST_INPUTS_54) == 54

    categories = {item["category"] for item in TEST_INPUTS_54}
    assert categories == {
        "casual", "daily_life", "studies", "relationships",
        "procrastination", "money", "work", "roast", "questions"
    }


def test_prompt_contains_simple_english_directives():
    """Verify prompt explicitly instructs the LLM to use simple spoken English and savage humour."""
    manager = PersonalityManager()
    prompt = manager.get_system_prompt()

    assert "SIMPLE, EVERYDAY SPOKEN ENGLISH" in prompt
    assert "CRITICAL RULE: SIMPLE EVERYDAY VOCABULARY ONLY" in prompt
    assert "NATURAL HUMAN HUMOUR STYLE" in prompt
    assert "Bro, tomorrow has blocked you already." in prompt
    assert "Good morning, grandpa." in prompt
    assert "Your bank account is running a charity for zeroes." in prompt
    assert "Yeah bro, your mirror must be proud." in prompt
    assert "Bro, the question paper won this fight." in prompt


@pytest.mark.parametrize(
    "item",
    TEST_INPUTS_54,
    ids=[f"{item['category']}_{i+1}" for i, item in enumerate(TEST_INPUTS_54)],
)
def test_natural_savage_turn(item):
    """
    Test live generation for each of the 54 inputs:
    - Simple everyday English (no big words)
    - 1 sentence (<= 15 words)
    - Natural savage tone
    - Contextual relevance
    """
    engine = IntelligenceEngine()
    if not engine.gemini.is_configured:
        pytest.skip("Gemini API not configured.")

    user_msg = item["input"]
    category = item["category"]

    async def _run_turn():
        await asyncio.sleep(0.8)  # Free-tier rate pacing
        turn = await engine.process_turn(
            message=user_msg,
            conversation_id=f"test_savage_{category}_{abs(hash(user_msg)) % 10000}",
        )
        reply = turn.assistant_response
        lower_reply = reply.lower()
        words = count_words(reply)
        sentences = count_sentences(reply)

        print(f"\n[{category.upper()}] User: {user_msg!r} -> AI: {reply!r} ({words} words, {sentences} sentence)")

        # 1. Strict length constraints
        assert sentences <= 1, f"Expected 1 sentence but got {sentences}: {reply}"
        assert words <= 15, f"Expected <= 15 words but got {words}: {reply}"

        # 2. Contextual relevance & Quality
        is_valid, reason = validate_response_quality(user_msg, reply, mode="unified_comedy")
        assert is_valid, f"Quality check failed for {user_msg!r} -> {reply!r}: {reason}"

        # 3. Simple vocabulary check (no fancy/robotic words)
        for bad_word in FORBIDDEN_COMPLEX_WORDS:
            assert bad_word not in lower_reply, f"Used complicated/robotic word '{bad_word}' in response: {reply}"

        # 4. Zero corporate assistant tone
        for phrase in FORBIDDEN_ASSISTANT_PHRASES:
            assert phrase not in lower_reply, f"Sounded like a corporate assistant: {reply}"

        # 5. No robotic stage directions or asterisks
        assert not re.search(r"[*<>()\[\]]", reply), f"Found robotic stage directions or formatting: {reply}"

    asyncio.run(_run_turn())

