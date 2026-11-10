"""
Tests for LaughBot AI Response Length & Humour Validator.
"""

from backend.services.response_validator import (
    clean_candidate_text,
    count_sentences,
    count_words,
    enforce_strict_short_response,
    extract_single_sentence,
    is_sensitive_message,
    is_valid_short_response,
    validate_response_quality,
)


def test_clean_candidate_text_prefixes():
    raw = "LaughBot: Your brain has officially resigned."
    assert clean_candidate_text(raw) == "Your brain has officially resigned."

    raw2 = "AI: Surviving, unlike your productivity."
    assert clean_candidate_text(raw2) == "Surviving, unlike your productivity."

    raw3 = "Response: *giggles* That outfit is brave."
    assert clean_candidate_text(raw3) == "That outfit is brave."


def test_clean_candidate_text_fillers():
    raw = "Haha, you actually think you did something there."
    assert clean_candidate_text(raw) == "You actually think you did something there."

    raw2 = "Well! Your alarm clock called it quits."
    assert clean_candidate_text(raw2) == "Your alarm clock called it quits."


def test_clean_candidate_text_quotes():
    raw = '"Your stomach is filing a formal complaint."'
    assert clean_candidate_text(raw) == "Your stomach is filing a formal complaint."

    raw2 = "“Tomorrow called; it blocked your number.”"
    assert clean_candidate_text(raw2) == "Tomorrow called; it blocked your number."


def test_extract_single_sentence_multi():
    raw = "Your brain has officially resigned. Go take a nap. You need it."
    result = extract_single_sentence(raw)
    assert result == "Your brain has officially resigned."
    assert count_sentences(result) == 1


def test_is_valid_short_response():
    assert is_valid_short_response("Your brain has officially resigned.") is True
    assert is_valid_short_response("One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen.") is False
    assert is_valid_short_response("First joke. Second joke.") is False


def test_enforce_strict_short_response_removes_extra_sentences():
    raw = "Your brain has officially resigned. You should sleep now."
    enforced, modified = enforce_strict_short_response(raw)
    assert enforced == "Your brain has officially resigned."
    assert modified is True
    assert count_sentences(enforced) == 1
    assert count_words(enforced) <= 15


def test_enforce_strict_short_response_word_boundary_never_cuts_words():
    raw = "Your confidence is so absurdly and remarkably oversized that all local satellites are losing their signals."
    enforced, modified = enforce_strict_short_response(raw, max_words=10)
    words = enforced.split()
    assert len(words) <= 10
    # Check each word is a full word without broken letters
    for w in words:
        clean_w = w.strip(".,!?:;")
        assert clean_w in raw.split() or clean_w.lower() in [x.lower() for x in raw.split()]
    assert enforced.endswith(".")


def test_is_sensitive_message_detection():
    assert is_sensitive_message("My grandfather passed away today.") is True
    assert is_sensitive_message("I am feeling very suicidal.") is True
    assert is_sensitive_message("I was diagnosed with cancer.") is True
    assert is_sensitive_message("I'm thinking of starting gym.") is False
    assert is_sensitive_message("I woke up late today.") is False


def test_validate_response_quality_good_vs_bad():
    # Example 1: Exam vs Weather disconnect
    valid, reason = validate_response_quality("I have an exam tomorrow.", "The weather is beautiful today, just like your dreams.")
    assert valid is False
    assert "weather" in reason.lower()

    valid, reason = validate_response_quality("I have an exam tomorrow.", "Your sleep schedule is officially in danger.")
    assert valid is True

    # Example 2: Procrastination vs Wi-Fi disconnect
    valid, reason = validate_response_quality("I have been procrastinating for a week.", "Your Wi-Fi is slower than a turtle.")
    assert valid is False
    assert "wi-fi" in reason.lower()

    valid, reason = validate_response_quality("I have been procrastinating for a week.", "Even your excuses need a vacation.")
    assert valid is True

    # Example 3: Studying vs Potato disconnect
    valid, reason = validate_response_quality("I studied for ten hours.", "You're basically a potato.")
    assert valid is False
    assert "potato" in reason.lower()

    valid, reason = validate_response_quality("I studied for ten hours.", "Your books finally recognize your existence.")
    assert valid is True

    # Example 4: Keys
    valid, reason = validate_response_quality("I lost my keys.", "Your keys chose independence.")
    assert valid is True

    # Example 5: Sensitive message protection
    valid, reason = validate_response_quality("My mother died today.", "Haha, what a loser!")
    assert valid is False

    valid, reason = validate_response_quality("My mother died today.", "I am so deeply sorry for your loss.")
    assert valid is True


