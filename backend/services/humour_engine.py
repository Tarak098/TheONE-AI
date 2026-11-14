"""
TheONE AI - Humour Context Engine & Conversation Intelligence Layer.

Implements deep situation understanding, conversational continuity, repetition detection,
and comedic angle selection.

Architecture flow:
USER MESSAGE
    |
    v
CONVERSATION UNDERSTANDING (Topic, Intent, Situational Stakes)
    |
    v
HUMOUR CONTEXT ENGINE
    |-- Situation understanding
    |-- Conversation continuity (Multi-turn tracking & callbacks)
    |-- Meme, movie & cultural inspiration
    |-- Comedic angle selection (Deadpan, Reversal, Savage, Irony, etc.)
    |-- Repetition detection (Anti-cliché & opener diversity)
    |-- Language adaptation (Regional idioms & spoken phrasing)
    |
    v
GEMMA COMEDY GENERATION GUIDANCE
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.services.conversation import ConversationSession
from backend.services.humour_knowledge import (
    ComedicTechnique,
    ConversationTopic,
    HUMOUR_KNOWLEDGE_BASE,
    HumourConcept,
    get_humour_concept,
    get_regional_flavour,
)

logger = logging.getLogger("theone.humour_engine")


@dataclass
class HumourContext:
    """Structured context and constraints for generating an original comedic response."""
    topic: ConversationTopic
    intent: str
    technique: ComedicTechnique
    banned_openers: List[str]
    banned_phrases: List[str]
    creative_angle: str
    callback_note: Optional[str] = None
    cultural_note: Optional[str] = None
    language_code: str = "en-IN"

    def format_prompt_injection(self) -> str:
        """Format guidance into an actionable instruction block for Gemma/Gemini."""
        lines = [
            "--- CONTEXT-FIRST HUMOUR INTELLIGENCE DIRECTIVE ---",
            f"1. TOPIC & SITUATION: {self.topic.value.replace('_', ' ').title()}",
            f"2. COMEDIC ANGLE: {self.technique.value.replace('_', ' ').title()} — {self.creative_angle}",
        ]
        if self.callback_note:
            lines.append(f"3. MULTI-TURN CONTINUITY: {self.callback_note}")
        else:
            lines.append("3. MULTI-TURN CONTINUITY: Respond specifically to the user's situation; do not tell a generic disconnected joke.")

        if self.banned_openers:
            banned_str = ", ".join(f'"{op}"' for op in self.banned_openers)
            lines.append(f"4. ANTI-REPETITION (STRICT): DO NOT start your response with any of: {banned_str}.")

        if self.banned_phrases:
            banned_p_str = ", ".join(f'"{p}"' for p in self.banned_phrases)
            lines.append(f"5. FORBIDDEN CLICHÉS: NEVER use any of: {banned_p_str}.")

        if self.cultural_note:
            lines.append(f"6. LANGUAGE & NATURAL CONVERSATIONAL TONE: {self.cultural_note}")

        lines.append("7. LENGTH & FORMAT: EXACTLY ONE natural spoken sentence. 3 to 10 words (maximum 15 words). No extra sentences.")
        lines.append("-----------------------------------------------------")
        return "\n".join(lines)


# Regex-based topic keywords for lightning-fast, zero-overhead classification
TOPIC_KEYWORDS: List[Tuple[ConversationTopic, re.Pattern]] = [
    (
        ConversationTopic.SENSITIVE_CRISIS,
        re.compile(
            r"\b(?:suicide|kill myself|want to die|depressed|depression|self harm|"
            r"passed away|died|death|funeral|cancer|hospital|lost my dad|lost my mom|"
            r"lost my brother|lost my sister|diagnosed|heartbroken|emergency)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.EXAMS_STUDIES,
        re.compile(
            r"\b(?:exam|exams|study|studying|studied|syllabus|chapter|chapters|test|tests|grade|grades|assignment|"
            r"lecture|homework|math|maths|physics|chemistry|biology|school|college|degree|"
            r"topper|pass|passed|fail|failed|flunked|questions? paper|textbook|revision|"
            r"పరీక్ష|చదువు|పాస్|ఫెయిల్|पढ़ाई|परीक्षा|पढ़ा|பரீட்சை|ಓದು|ಪರೀಕ್ಷೆ|ಪരീക്ഷ|পড়া|পরীক্ষা|अभ्यास|વાંચન|ਪੜ੍ਹਾਈ)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.MONEY_FINANCES,
        re.compile(
            r"\b(?:money|salary|broke|bank|account|wallet|buy|bought|buying|spend|spent|"
            r"spending|expensive|cash|dollars?|rupees?|rupee|shopping|cart|discount|bill|"
            r"emi|credit card|luxury|rich|poor|savings|wealth|డబ్బు|పైసలు|पैसा|पैसे|பணம்|ದುಡ್ಡು|പണം|টাকা|पैसे|રૂપિયા|ਪੈਸੇ)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.WORK_CAREER,
        re.compile(
            r"\b(?:work|working|job|boss|office|meeting|meetings|email|emails|manager|"
            r"interview|resignation|resign|promoted|promotion|corporate|client|clients|"
            r"shift|deadlines?|standup|zoom call|colleague|coworker|cubicle)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.RELATIONSHIPS_DATING,
        re.compile(
            r"\b(?:crush|date|dating|girlfriend|boyfriend|gf|bf|ex|relationship|relationships|"
            r"ghosted|left on read|texted|flirt|breakup|broken up|"
            r"(?:single\s+(?:life|forever|status)|i'?m\s+single|still\s+single)|"
            r"marriage|married|tinder|bumble|hinge|propose|love)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.FITNESS_HEALTH,
        re.compile(
            r"\b(?:gym|workout|working out|cardio|running|run|ran|marathon|protein|muscle|"
            r"muscles|biceps|calories|weights|dumbbell|pushups|diet|cheat day|fat|abs|fitness)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.FOOD_COOKING,
        re.compile(
            r"\b(?:cook|cooking|cooked|burned|burnt|dinner|lunch|breakfast|food|recipe|"
            r"starving|hungry|pizza|burger|biryani|snack|snacks|tea|chai|coffee|kitchen|"
            r"toast|meal|swiggy|zomato|delivery|tasty|eating|ate)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.SLEEP_ROUTINES,
        re.compile(
            r"\b(?:sleep|sleepy|sleeping|slept|woke up|wake up|alarm|snooze|morning|tired|"
            r"exhausted|insomnia|bed|nap|napping|6 am|7 am|12 pm|midnight)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.TECHNOLOGY_GAMING,
        re.compile(
            r"\b(?:game|gaming|gamer|valorant|csgo|pubg|fortnite|ping|lag|fps|bugs?|errors?|"
            r"code|coding|python|javascript|laptop|pc|computer|wifi|wi-fi|router|battery|"
            r"phone|crash|crashed|steam|discord|console|playstation|xbox)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.PROCRASTINATION_HABITS,
        re.compile(
            r"\b(?:tomorrow|later|scroll|scrolling|reels?|shorts?|tiktok|binge|binging|"
            r"procrastinate|procrastinating|procrastination|netflix|youtube|couch|doing nothing|"
            r"lazy|laziness|snooze|delay|delayed|putting off)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.BOASTING_EGO,
        re.compile(
            r"\b(?:genius|smart|intelligent|undisputed|boss|best|look amazing|handsome|beautiful|"
            r"attractive|pro|talented|gifted|champion|superior|ego|masterpiece|michelin star)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.EMBARRASSMENT,
        re.compile(
            r"\b(?:awkward|embarrassing|embarrassed|tripped|stumbled|waved|stranger|"
            r"voice crack|cringe|cringed|public|stared|noticed|regret it)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.FAILURE_MISTAKES,
        re.compile(
            r"\b(?:failed|ruined|broke|broken|dropped|burned|burn|accidental|replied-all|"
            r"mistake|mess|messed up|disaster|lost|screwed up|wrong)\b",
            re.I,
        ),
    ),
    (
        ConversationTopic.FRIENDS_SOCIAL,
        re.compile(
            r"\b(?:friends?|bestie|gang|trip|hangout|hanging out|party|plans|group chat|"
            r"left me out|ignore|ignored|third wheel|social battery|introvert)\b",
            re.I,
        ),
    ),
]


class HumourEngine:
    """
    Core conversation intelligence layer for TheONE AI.

    Analyzes incoming user messages in the context of recent turns,
    identifies topic, intent, and continuity, selects a suitable comedic angle,
    and enforces strict non-repetition rules.
    """

    def __init__(self) -> None:
        pass

    def classify_topic(self, text: str) -> ConversationTopic:
        """Classify user text into a primary ConversationTopic with sub-millisecond latency."""
        if not text or not text.strip():
            return ConversationTopic.EVERYDAY_CASUAL

        for topic, pattern in TOPIC_KEYWORDS:
            if pattern.search(text):
                return topic

        return ConversationTopic.EVERYDAY_CASUAL

    def detect_intent(self, text: str) -> str:
        """Detect the conversational posture or intent of the user."""
        t = text.lower().strip()
        if re.search(r"\b(?:roast me|destroy me|give me your best|comeback)\b", t):
            return "roast_request"
        if re.search(r"\b(?:why|how|what|when|where|who)\b", t) or t.endswith("?"):
            return "question"
        if re.search(r"\b(?:i am|i'm the|my|i look|genius|best|undisputed)\b", t):
            return "boast"
        if re.search(r"\b(?:will|going to|tomorrow|later|next week|starting)\b", t):
            return "excuse_delay"
        if re.search(r"\b(?:hate|annoying|tired of|drives me crazy|angry|frustrated|insane)\b", t):
            return "complaint"
        if re.search(r"\b(?:failed|ruined|broke|burned|lost|dropped)\b", t):
            return "failure"
        return "sharing"

    def extract_salient_entity(self, text: str) -> Optional[str]:
        """Extract a key noun or phrase that can be stored for future conversational callbacks."""
        matches = re.findall(r"\b[A-Za-z]{4,}\b", text)
        stopwords = {
            "that", "this", "what", "with", "have", "from", "they", "will",
            "would", "there", "their", "about", "which", "could", "other",
            "into", "more", "some", "time", "very", "when", "come", "here",
            "just", "like", "your", "know", "take", "people", "year", "good",
            "them", "some", "could", "them", "then", "than", "look", "only",
            "today", "yesterday", "tomorrow", "night", "morning",
        }
        candidates = [m.lower() for m in matches if m.lower() not in stopwords]
        return candidates[0] if candidates else None

    def analyze_continuity_and_callbacks(
        self,
        current_topic: ConversationTopic,
        session: Optional[ConversationSession],
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Analyze conversation continuity from recent turns:
        - Detect ongoing topic progression (multi-turn conversation)
        - Spot intelligent callback opportunities from earlier user mentions.
        """
        if not session or not session.messages:
            return None, None

        recent_topics = session.recent_topics[-3:] if hasattr(session, "recent_topics") else []
        is_topic_continuation = recent_topics and recent_topics[-1] == current_topic.value

        callback_note = None
        if is_topic_continuation:
            callback_note = (
                "The user is continuing the same topic as the previous message. "
                "Build on this dialogue naturally; do NOT restart with a disconnected joke."
            )
        else:
            # Check for possible cross-turn callback
            callbacks = session.get_potential_callbacks() if hasattr(session, "get_potential_callbacks") else []
            if callbacks and len(session.messages) >= 4:
                earlier_entity = callbacks[0]
                callback_note = (
                    f"Notice: User previously mentioned '{earlier_entity}'. "
                    f"If a natural, hilarious connection exists, use a quick callback. Otherwise focus on the current message."
                )

        return callback_note, is_topic_continuation

    def select_comedic_technique(
        self,
        topic: ConversationTopic,
        intent: str,
        session: Optional[ConversationSession] = None,
    ) -> ComedicTechnique:
        """Select a fresh comedic technique, rotating away from recently used styles."""
        concept = get_humour_concept(topic)
        preferred = concept.preferred_techniques

        # Tailor technique to intent
        if intent == "boast":
            return ComedicTechnique.SAVAGE_ROAST
        elif intent == "excuse_delay":
            return ComedicTechnique.CINEMATIC_REVERSAL
        elif intent == "complaint":
            return ComedicTechnique.DEADPAN
        elif intent == "failure":
            return ComedicTechnique.DARK_IRONY
        elif intent == "roast_request":
            return ComedicTechnique.SAVAGE_ROAST

        return preferred[0] if preferred else ComedicTechnique.OBSERVATIONAL_SARCASM

    def determine_banned_openers(self, session: Optional[ConversationSession]) -> List[str]:
        """
        Determine which opening words are strictly forbidden on this turn
        to guarantee rich stylistic diversity and prevent predictable repetition.
        """
        banned = []
        if not session:
            # Default anti-cliché openers to rotate out
            return ["Bro, your", "Your brain", "At this point"]

        recent_openers = session.get_recent_openers() if hasattr(session, "get_recent_openers") else []
        last_opener = recent_openers[-1].lower() if recent_openers else ""

        # If previous assistant reply started with "bro", strictly forbid "bro" or "yeah bro"
        if "bro" in last_opener:
            banned.extend(["Bro", "Yeah bro", "Dude"])
        # If previous assistant reply started with "your", strictly forbid "your"
        if "your" in last_opener:
            banned.extend(["Your", "Your whole"])
        # If previous assistant reply started with "even", strictly forbid "even"
        if "even" in last_opener:
            banned.extend(["Even", "Even your"])
        # If previous assistant reply started with "congratulations", forbid it
        if "congratulations" in last_opener or "congrats" in last_opener:
            banned.extend(["Congratulations", "Congrats"])

        # Dynamically ban the immediate first word used in the last turn
        if last_opener:
            first_word = last_opener.split()[0].capitalize()
            if first_word and first_word not in banned:
                banned.append(first_word)

        # Always ban overused canned openers
        banned.extend(["Imagine", "At this point", "Honestly", "Well well"])

        # Deduplicate while preserving order
        seen = set()
        unique_banned = []
        for b in banned:
            lower_b = b.lower()
            if lower_b not in seen:
                seen.add(lower_b)
                unique_banned.append(b)

        return unique_banned

    def build_humour_context(
        self,
        user_message: str,
        session: Optional[ConversationSession] = None,
        language_code: str = "en-IN",
    ) -> HumourContext:
        """
        Generate a complete, structured HumourContext for the turn.
        Executes in < 1ms in-memory.
        """
        topic = self.classify_topic(user_message)
        intent = self.detect_intent(user_message)
        concept = get_humour_concept(topic)
        technique = self.select_comedic_technique(topic, intent, session)
        callback_note, _ = self.analyze_continuity_and_callbacks(topic, session)
        banned_openers = self.determine_banned_openers(session)
        banned_phrases = list(concept.banned_cliches)

        # Retrieve regional cultural flavour
        regional = get_regional_flavour(language_code)
        cultural_note = (
            f"Target dialect: {language_code}. {regional.get('style_notes', '')} "
            f"Deliver an authentic spoken comeback, NOT a translation."
        )

        # Salient entity extraction for session recording
        entity = self.extract_salient_entity(user_message)
        if session and hasattr(session, "record_turn_metadata"):
            session.record_turn_metadata(topic=topic.value, user_entity=entity)

        return HumourContext(
            topic=topic,
            intent=intent,
            technique=technique,
            banned_openers=banned_openers,
            banned_phrases=banned_phrases,
            creative_angle=concept.creative_angle,
            callback_note=callback_note,
            cultural_note=cultural_note,
            language_code=language_code,
        )
