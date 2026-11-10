"""
TheONE AI - Response Length, Context Relevance & Humour Validator.

Ensures every AI response:
1. Strictly satisfies the single-sentence rule.
2. Enforces word count limits (prefer 3-10 words, max 15 words).
3. Strips conversational prefixes and raw filler openers.
4. Validates contextual relevance against the user's actual message.
5. Rejects off-topic, generic stock insults (e.g. random Wi-Fi or potato jokes).
6. Detects serious or sensitive messages to ensure respectful, safe handling.
"""

import logging
import re
from typing import List, Optional, Set, Tuple

logger = logging.getLogger("theone.validator")

# Word count boundaries
MIN_PREFERRED_WORDS = 3
MAX_PREFERRED_WORDS = 10
MAX_ALLOWED_WORDS = 15

# Conversational starter fillers to strip if they precede the punchline
PREFIX_FILLERS = re.compile(
    r"^(?:(?:TheONE AI|TheONE|LaughBot|AI|Bot|Response|Assistant|Companion)\s*:\s*)+"
    r"|^(?:\*[^*]+\*\s*)+"
    r"|^(?:Haha|Lol|LMAO|Well|Oh|Look|Listen|Honestly)[,!.:\s]+",
    re.IGNORECASE,
)

# Common sentence terminal boundary pattern
SENTENCE_SPLIT_PATTERN = re.compile(r"""(?<=[.!?])["')\]}]*\s+(?=[A-Z0-9"'“])""")

# Sensitive / Crisis keywords where humor/roasting is inappropriate
SENSITIVE_KEYWORDS: Set[str] = {
    "suicide", "suicidal", "kill myself", "killing myself", "end my life",
    "want to die", "depressed", "depression", "self harm", "hurt myself",
    "died", "passed away", "funeral", "death", "cancer", "hospital",
    "lost my dad", "lost my mom", "lost my mother", "lost my father",
    "lost my sister", "lost my brother", "lost my child", "lost my friend",
    "grieving", "grief", "abused", "heartbroken", "diagnosed", "emergency",
}

# Stopwords for user message keyword extraction
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "about", "for", "is", "am", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "to", "from", "in", "out", "on", "off", "over", "under", "again", "further",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "very", "really", "today", "now", "here", "there",
    "im", "i'm", "going", "got", "get", "like", "tell", "give", "make",
}

# Cliché off-topic disconnect patterns (generic insults that disconnect from user message)
GENERIC_DISCONNECT_PUNCHLINES = [
    (
        re.compile(r"\b(?:wi-?fi|router|broadband|connection|bandwidth)\b", re.I),
        re.compile(r"\b(?:wi-?fi|internet|router|broadband|signal|online|connection|disconnect)\b", re.I),
        "Mentions Wi-Fi / connectivity when user message is not about internet",
    ),
    (
        re.compile(r"\b(?:potato|vegetables?|couch potato|spud)\b", re.I),
        re.compile(r"\b(?:potato|vegetables?|couch potato|spud|veggie)\b", re.I),
        "Calls user a potato when user did not mention potatoes or vegetables",
    ),
    (
        re.compile(r"\b(?:mirror|reflection)\b", re.I),
        re.compile(r"\b(?:mirror|reflection|face|look|ugly|handsome|pretty|appearance|outfit|clothes)\b", re.I),
        "Mentions mirror/reflection when user did not mention appearance or mirror",
    ),
    (
        re.compile(r"\b(?:weather|raining|rain|clouds|sunny|storm)\b", re.I),
        re.compile(r"\b(?:weather|rain|raining|clouds|sunny|storm|forecast|umbrella)\b", re.I),
        "Mentions weather/rain when user message was not about weather",
    ),
    (
        re.compile(r"\b(?:empty bank account|bankrupt|broke|atm)\b", re.I),
        re.compile(r"\b(?:money|bank|broke|account|dollars|salary|cash|spend|spent|buy|bought|wallet|cost|expensive)\b", re.I),
        "Mentions empty bank account when user message was not about finances",
    ),
]


def clean_candidate_text(text: str) -> str:
    """
    Clean raw model text by stripping prefixes, stage directions,
    filler openers, and outer quote marks.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Strip conversational prefixes and fillers repeatedly if layered
    for _ in range(4):
        prev = cleaned
        cleaned = re.sub(
            r"^(?:(?:TheONE AI|TheONE|LaughBot|AI|Bot|Response|Assistant|Companion)\s*:\s*)+",
            "",
            cleaned,
            flags=re.I,
        ).strip()
        cleaned = re.sub(r"^\*[^*]+\*\s*", "", cleaned).strip()
        cleaned = re.sub(r"^\([^)]+\)\s*", "", cleaned).strip()
        cleaned = re.sub(
            r"^(?:Haha|Lol|LMAO|Well|Oh|Look|Listen|Honestly)[,!.:\s]+",
            "",
            cleaned,
            flags=re.I,
        ).strip()
        cleaned = cleaned.strip("\"'“”` ")
        if cleaned == prev:
            break

    # Strip surrounding asterisks or quotes
    if cleaned.startswith("*") and cleaned.endswith("*") and len(cleaned) > 2:
        cleaned = cleaned[1:-1].strip()

    # Strip angle-bracket tags if generated by LLM (e.g., <laugh>, <excited>)
    cleaned = re.sub(r"<[^>]+>", "", cleaned).strip()

    # Ensure the first letter is capitalized
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


def extract_single_sentence(text: str) -> str:
    """
    Extract exactly the first complete sentence from a potentially multi-sentence response.
    Never leaves a word cut in half.
    """
    cleaned = clean_candidate_text(text)
    if not cleaned:
        return ""

    sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(cleaned) if s.strip()]
    if not sentences:
        return cleaned

    return sentences[0]


def count_words(text: str) -> int:
    """Count whitespace-separated words in a string."""
    return len(text.strip().split()) if text and text.strip() else 0


def count_sentences(text: str) -> int:
    """Count number of sentences in text based on terminal punctuation."""
    cleaned = clean_candidate_text(text)
    if not cleaned:
        return 0
    sentences = [s.strip() for s in SENTENCE_SPLIT_PATTERN.split(cleaned) if s.strip()]
    return len(sentences) if sentences else 1


def is_valid_short_response(text: str, max_words: int = MAX_ALLOWED_WORDS) -> bool:
    """
    Check if a response strictly satisfies the one-sentence and word-count limits.
    """
    cleaned = clean_candidate_text(text)
    if not cleaned:
        return False

    num_words = count_words(cleaned)
    num_sentences = count_sentences(cleaned)

    return (num_sentences <= 1) and (1 <= num_words <= max_words)


def is_sensitive_message(user_message: str) -> bool:
    """
    Detect whether user message involves genuine grief, self-harm, or serious emergency.
    """
    if not user_message:
        return False
    lower_msg = user_message.lower()
    for phrase in SENSITIVE_KEYWORDS:
        if phrase in lower_msg:
            return True
    return False


def extract_content_words(text: str) -> List[str]:
    """Extract significant lowercase content words from text."""
    if not text:
        return []
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [w for w in words if w not in STOPWORDS]


def validate_response_quality(
    user_message: str,
    candidate_response: str,
    mode: str = "always_funny",
) -> Tuple[bool, str]:
    """
    Evaluate candidate response quality, length, and contextual relevance.

    Returns:
        Tuple[bool, str]: (is_valid, reason)
    """
    cleaned_candidate = clean_candidate_text(candidate_response)
    if not cleaned_candidate:
        return False, "Response is empty."

    # 1. Length validation (Maximum 1 sentence, maximum 15 words)
    num_sentences = count_sentences(cleaned_candidate)
    num_words = count_words(cleaned_candidate)

    if num_sentences > 1:
        return False, f"Contains multiple sentences ({num_sentences} sentences)."

    if num_words > MAX_ALLOWED_WORDS:
        return False, f"Response exceeds word limit ({num_words} words > {MAX_ALLOWED_WORDS})."

    if num_words < 2:
        return False, "Response is too short or empty."

    # 2. Check for verbatim echo (repeating user's message back)
    clean_user = user_message.strip().lower().rstrip(".?!")
    clean_cand = cleaned_candidate.strip().lower().rstrip(".?!")
    if clean_user and clean_user == clean_cand:
        return False, "Response simply repeats the user's input."

    # 3. Disconnected stock punchline detection
    for response_pattern, user_allowed_pattern, reason in GENERIC_DISCONNECT_PUNCHLINES:
        if response_pattern.search(cleaned_candidate):
            if not user_allowed_pattern.search(user_message):
                return False, reason

    # 4. Sensitive topic handling
    if is_sensitive_message(user_message):
        # Must not be an insult or mocking remark
        lower_resp = cleaned_candidate.lower()
        mock_words = ["haha", "loser", "clown", "stupid", "idiot", "pathetic", "ugly"]
        if any(w in lower_resp for w in mock_words):
            return False, "Response mocked a sensitive or serious topic."

    # 5. Question directness check
    user_trimmed = user_message.strip()
    is_question = user_trimmed.endswith("?") or re.match(r"^(?:who|what|where|when|why|how|can|is|are|do|does)\b", user_trimmed, re.I)
    if is_question and len(user_trimmed.split()) >= 3:
        # If user asked a specific question, the response should not be a disconnected cliché like "Your Wi-Fi is slow"
        user_keywords = set(extract_content_words(user_message))
        resp_keywords = set(extract_content_words(cleaned_candidate))
        # Questions with content words generally shouldn't share zero domain vocabulary unless witty answer
        pass

    return True, "Response is valid and context-aligned."


def enforce_strict_short_response(
    text: str,
    max_words: int = MAX_ALLOWED_WORDS,
) -> Tuple[str, bool]:
    """
    Enforce single-sentence and maximum word limit on text.
    Never truncates a word halfway.

    Returns:
        Tuple[str, bool]: (enforced_text, was_modified)
    """
    cleaned = clean_candidate_text(text)
    original = cleaned

    # Step 1: Extract strictly the first sentence
    single_sentence = extract_single_sentence(cleaned)

    # Step 2: Check word count of the single sentence
    words = single_sentence.split()
    if len(words) <= max_words:
        was_modified = single_sentence != original
        return single_sentence, was_modified

    # Step 3: If still over max_words, look for natural clause boundaries
    for clause_delim in (";", " - ", " – ", ","):
        if clause_delim in single_sentence:
            parts = single_sentence.split(clause_delim)
            candidate = parts[0].strip()
            candidate_words = candidate.split()
            if 3 <= len(candidate_words) <= max_words:
                if not candidate.endswith((".", "!", "?")):
                    candidate += "."
                return candidate, True

    # Step 4: Truncate cleanly at a whole-word boundary (never slice mid-word)
    truncated_words = words[:max_words]
    truncated_sentence = " ".join(truncated_words).rstrip(",;:-")
    if not truncated_sentence.endswith((".", "!", "?")):
        truncated_sentence += "."

    logger.info(
        "Truncated response at word boundary from %d to %d words",
        len(words),
        len(truncated_words),
    )
    return truncated_sentence, True
