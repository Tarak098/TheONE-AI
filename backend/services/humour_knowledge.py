"""
TheONE AI - Curated Humour, Meme & Comedy Reference Knowledge Base.

Provides an in-memory, zero-latency reference architecture of comedic techniques,
internet meme concepts, cultural comedy tropes, and regional comedic patterns.

Principles:
1. The user's actual conversation is ALWAYS the source of humour.
2. Memes, movies, and comedic techniques are creative inspirations, NOT fixed scripts.
3. No stock jokes, no copyrighted material, no repeated cliché templates.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ComedicTechnique(str, Enum):
    """Core comedic delivery mechanisms."""
    DEADPAN = "deadpan"                         # Flat, unbothered statement of stark truth
    CINEMATIC_REVERSAL = "cinematic_reversal"   # Dramatic expectation flipped into hilarious anti-climax
    SAVAGE_ROAST = "savage_roast"               # Sharp, surgical punchline targeting an excuse or boast
    DARK_IRONY = "dark_irony"                   # Witty bleakness, ironic consequences of bad decisions
    OBSERVATIONAL_SARCASM = "observational_sarcasm"  # Calling out the absurd irrationality of normal habits
    ABSURDIST_EXAGGERATION = "absurdist_exaggeration" # Surreal visual or conceptual escalation
    UNDERSTATEMENT = "understatement"           # Treating a disaster like a minor tea spill
    INTELLIGENT_CALLBACK = "intelligent_callback" # Funny cross-reference to earlier conversation details


class ConversationTopic(str, Enum):
    """Categorized conversation domains."""
    EXAMS_STUDIES = "exams_studies"
    MONEY_FINANCES = "money_finances"
    WORK_CAREER = "work_career"
    RELATIONSHIPS_DATING = "relationships_dating"
    FRIENDS_SOCIAL = "friends_social"
    FITNESS_HEALTH = "fitness_health"
    PROCRASTINATION_HABITS = "procrastination_habits"
    FOOD_COOKING = "food_cooking"
    SLEEP_ROUTINES = "sleep_routines"
    TECHNOLOGY_GAMING = "technology_gaming"
    BOASTING_EGO = "boasting_ego"
    FAILURE_MISTAKES = "failure_mistakes"
    EMBARRASSMENT = "embarrassment"
    EVERYDAY_CASUAL = "everyday_casual"
    SENSITIVE_CRISIS = "sensitive_crisis"


@dataclass(frozen=True)
class HumourConcept:
    """A structured creative inspiration pattern for humour generation."""
    concept_id: str
    topic: ConversationTopic
    preferred_techniques: List[ComedicTechnique]
    cultural_context: str
    creative_angle: str
    banned_cliches: List[str]
    sample_contextual_directions: List[str]


# Curated Knowledge Base of Humour Concepts across Topics
HUMOUR_KNOWLEDGE_BASE: Dict[ConversationTopic, List[HumourConcept]] = {
    ConversationTopic.EXAMS_STUDIES: [
        HumourConcept(
            concept_id="exam_plot_twist",
            topic=ConversationTopic.EXAMS_STUDIES,
            preferred_techniques=[ComedicTechnique.CINEMATIC_REVERSAL, ComedicTechnique.DEADPAN],
            cultural_context="Exam hall panic, starting syllabus on the last night, syllabus that never ends",
            creative_angle="Treat studying late as entering a suspense thriller or betting against destiny",
            banned_cliches=["bro, tomorrow has blocked you", "your brain is buffering", "question paper won"],
            sample_contextual_directions=[
                "Contrast confidence with the actual number of pages unread",
                "Frame exam hall silence as the suspense before a tragedy",
                "Highlight how suddenly cleaning the room feels more urgent than revising",
            ],
        ),
        HumourConcept(
            concept_id="exam_topper_contrast",
            topic=ConversationTopic.EXAMS_STUDIES,
            preferred_techniques=[ComedicTechnique.OBSERVATIONAL_SARCASM, ComedicTechnique.SAVAGE_ROAST],
            cultural_context="Topper friend claiming they studied nothing while you actually studied nothing",
            creative_angle="Expose the gap between study plans and YouTube scrolling",
            banned_cliches=["bro", "your brain"],
            sample_contextual_directions=[
                "Point out how opening page one counts as hard work in their mind",
                "Treat their syllabus completion as science fiction",
            ],
        ),
    ],

    ConversationTopic.MONEY_FINANCES: [
        HumourConcept(
            concept_id="wallet_minimalism",
            topic=ConversationTopic.MONEY_FINANCES,
            preferred_techniques=[ComedicTechnique.DARK_IRONY, ComedicTechnique.UNDERSTATEMENT],
            cultural_context="Salary gone in 3 days, online cart addiction, broke after impulse purchase",
            creative_angle="Frame spending as an extreme sport or voluntary financial destruction",
            banned_cliches=["charity for zeroes", "bank account crying", "empty bank account"],
            sample_contextual_directions=[
                "Highlight how their wallet is on a strict calorie-free diet",
                "Frame checking the account balance as opening a horror novel",
                "Point out how buying coffee somehow costs more than their life savings",
            ],
        ),
    ],

    ConversationTopic.WORK_CAREER: [
        HumourConcept(
            concept_id="corporate_existentialism",
            topic=ConversationTopic.WORK_CAREER,
            preferred_techniques=[ComedicTechnique.DEADPAN, ComedicTechnique.OBSERVATIONAL_SARCASM],
            cultural_context="Endless Zoom meetings that could have been emails, Monday blues, fake productivity",
            creative_angle="Contrast corporate jargon with the urge to take a three-hour nap",
            banned_cliches=["employee of the month", "bro"],
            sample_contextual_directions=[
                "Frame typing on the keyboard as an elaborate acting performance",
                "Treat nodding in a meeting as Olympic level endurance",
                "Observe how replying to one email exhausted their weekly enthusiasm",
            ],
        ),
    ],

    ConversationTopic.RELATIONSHIPS_DATING: [
        HumourConcept(
            concept_id="modern_dating_tragedy",
            topic=ConversationTopic.RELATIONSHIPS_DATING,
            preferred_techniques=[ComedicTechnique.CINEMATIC_REVERSAL, ComedicTechnique.SAVAGE_ROAST],
            cultural_context="Left on read, ghosting, overthinking a two-word text, talking stages",
            creative_angle="Highlight how the imaginary wedding was planned before the first reply arrived",
            banned_cliches=["notifications gave up", "bro"],
            sample_contextual_directions=[
                "Treat waiting for a reply as an archaeological expedition",
                "Frame double texting as an act of diplomatic desperation",
            ],
        ),
    ],

    ConversationTopic.FRIENDS_SOCIAL: [
        HumourConcept(
            concept_id="social_battery_drain",
            topic=ConversationTopic.FRIENDS_SOCIAL,
            preferred_techniques=[ComedicTechnique.DEADPAN, ComedicTechnique.ABSURDIST_EXAGGERATION],
            cultural_context="Cancelling plans last minute, social awkwardness, group chat drama",
            creative_angle="Celebrate staying indoors like winning an international award",
            banned_cliches=["bro"],
            sample_contextual_directions=[
                "Treat stepping outside as entering an active warzone",
                "Frame cancelling plans as achieving inner peace",
            ],
        ),
    ],

    ConversationTopic.FITNESS_HEALTH: [
        HumourConcept(
            concept_id="gym_delusion",
            topic=ConversationTopic.FITNESS_HEALTH,
            preferred_techniques=[ComedicTechnique.SAVAGE_ROAST, ComedicTechnique.OBSERVATIONAL_SARCASM],
            cultural_context="Buying gym shoes, working out for one day and checking the mirror, diet cheat days",
            creative_angle="Expose the heroic narrative behind lifting one light dumbbell",
            banned_cliches=["gym membership crying", "mirror must be proud"],
            sample_contextual_directions=[
                "Point out how buying gym clothes was the entire workout",
                "Treat walking to the fridge as their daily cardio",
            ],
        ),
    ],

    ConversationTopic.PROCRASTINATION_HABITS: [
        HumourConcept(
            concept_id="procrastination_masterclass",
            topic=ConversationTopic.PROCRASTINATION_HABITS,
            preferred_techniques=[ComedicTechnique.CINEMATIC_REVERSAL, ComedicTechnique.OBSERVATIONAL_SARCASM],
            cultural_context="3 AM reel rabbit hole, tomorrow plans that never come, deep cleaning to avoid work",
            creative_angle="Glorify delay tactics as sophisticated strategic planning",
            banned_cliches=["bro, tomorrow has blocked you", "thumb deserves award"],
            sample_contextual_directions=[
                "Observe how researching 14th century castles became urgent at 2 AM",
                "Frame tomorrow as a mythical land where all work magically gets done",
            ],
        ),
    ],

    ConversationTopic.FOOD_COOKING: [
        HumourConcept(
            concept_id="kitchen_disaster",
            topic=ConversationTopic.FOOD_COOKING,
            preferred_techniques=[ComedicTechnique.UNDERSTATEMENT, ComedicTechnique.DEADPAN],
            cultural_context="Burned meals, microwave gourmet, midnight snacking, delivery tracking",
            creative_angle="Treat simple burnt food like an ambitious art installation",
            banned_cliches=["smoke detector", "couch potato"],
            sample_contextual_directions=[
                "Treat the burnt crust as a bold textural choice",
                "Frame tracking the delivery driver as staring into the promised land",
            ],
        ),
    ],

    ConversationTopic.SLEEP_ROUTINES: [
        HumourConcept(
            concept_id="sleep_negotiation",
            topic=ConversationTopic.SLEEP_ROUTINES,
            preferred_techniques=[ComedicTechnique.DARK_IRONY, ComedicTechnique.DEADPAN],
            cultural_context="Snooze button warfare, 5 more minutes turning into 2 hours, waking up tired",
            creative_angle="Frame the alarm clock as an arch-nemesis fighting an unwinnable war",
            banned_cliches=["good morning grandpa", "buffering"],
            sample_contextual_directions=[
                "Observe how five snooze alarms simply created a dramatic soundtrack",
                "Treat opening one eye as an act of immense sacrifice",
            ],
        ),
    ],

    ConversationTopic.TECHNOLOGY_GAMING: [
        HumourConcept(
            concept_id="tech_rage_and_ragequit",
            topic=ConversationTopic.TECHNOLOGY_GAMING,
            preferred_techniques=[ComedicTechnique.SAVAGE_ROAST, ComedicTechnique.ABSURDIST_EXAGGERATION],
            cultural_context="Blaming ping for losses, code breaking on line 1, infinite loading wheels",
            creative_angle="Expose how the hardware is completely innocent of their gameplay mistakes",
            banned_cliches=["your wi-fi is slow", "potato router"],
            sample_contextual_directions=[
                "Point out how the enemy team won simply by standing still",
                "Frame 47 compiler errors as an emotional cry from the computer",
            ],
        ),
    ],

    ConversationTopic.BOASTING_EGO: [
        HumourConcept(
            concept_id="ego_puncture",
            topic=ConversationTopic.BOASTING_EGO,
            preferred_techniques=[ComedicTechnique.SAVAGE_ROAST, ComedicTechnique.CINEMATIC_REVERSAL],
            cultural_context="Self-proclaimed genius, claiming to look amazing, bragging about talent",
            creative_angle="Deflate supreme confidence with a swift, pinpoint reality check",
            banned_cliches=["mirror must be proud", "talent escaped captivity"],
            sample_contextual_directions=[
                "Let reality introduce itself politely to their fantasy",
                "Point out that confidence without evidence is just comedy",
            ],
        ),
    ],

    ConversationTopic.FAILURE_MISTAKES: [
        HumourConcept(
            concept_id="failure_celebration",
            topic=ConversationTopic.FAILURE_MISTAKES,
            preferred_techniques=[ComedicTechnique.DARK_IRONY, ComedicTechnique.UNDERSTATEMENT],
            cultural_context="Accidental reply-all, dropping food, awkward stumbles, blown opportunities",
            creative_angle="Treat catastrophic failure with calm, hilarious admiration",
            banned_cliches=["exam paper won", "better luck"],
            sample_contextual_directions=[
                "Applaud the consistency of their bad decisions",
                "Frame disaster as an unforgettable masterclass in what not to do",
            ],
        ),
    ],

    ConversationTopic.EMBARRASSMENT: [
        HumourConcept(
            concept_id="public_awkwardness",
            topic=ConversationTopic.EMBARRASSMENT,
            preferred_techniques=[ComedicTechnique.DEADPAN, ComedicTechnique.CINEMATIC_REVERSAL],
            cultural_context="Waving at stranger who wasn't waving, tripping in public, voice cracking",
            creative_angle="Treat moving to another continent as the only logical solution",
            banned_cliches=["bro"],
            sample_contextual_directions=[
                "Suggest changing their identity and fleeing the country",
                "Treat the ground as an accomplice in their public performance",
            ],
        ),
    ],

    ConversationTopic.EVERYDAY_CASUAL: [
        HumourConcept(
            concept_id="ordinary_absurdity",
            topic=ConversationTopic.EVERYDAY_CASUAL,
            preferred_techniques=[ComedicTechnique.OBSERVATIONAL_SARCASM, ComedicTechnique.DEADPAN],
            cultural_context="Boredom, bad weather, drinking tea, doing nothing, everyday chatter",
            creative_angle="Find the quiet existential comedy inside ordinary moments",
            banned_cliches=["even boredom is bored of you", "weather"],
            sample_contextual_directions=[
                "Point out that doing nothing takes tremendous dedication",
                "Observe how sitting completely still became their full-time occupation",
            ],
        ),
    ],

    ConversationTopic.SENSITIVE_CRISIS: [
        HumourConcept(
            concept_id="gentle_warmth_care",
            topic=ConversationTopic.SENSITIVE_CRISIS,
            preferred_techniques=[ComedicTechnique.UNDERSTATEMENT],
            cultural_context="Loss, grief, severe illness, genuine sadness",
            creative_angle="Warm, gentle, non-mocking camaraderie; zero cruel insults",
            banned_cliches=["loser", "pathetic", "clown"],
            sample_contextual_directions=[
                "Provide brief, quiet support without sounding like a corporate chatbot",
                "Acknowledge the heavy moment with genuine warmth",
            ],
        ),
    ],
}


# Regional and Indian Internet Culture Flavours for Multilingual Delivery
REGIONAL_HUMOUR_FLAVOURS: Dict[str, Dict[str, Any]] = {
    "te-IN": {
        "culture_markers": ["Biryani obsession", "Cinema dialogue exaggeration", "Tenglish slang", "Hostel / BTech realities", "RTC bus rush"],
        "slang_particles": ["bro", "mama", "guru", "thammudu", "enti idi"],
        "style_notes": "Punchy punchlines, hyper-expressive cinematic contrast, natural spoken Telugu without literary Telugu.",
    },
    "hi-IN": {
        "culture_markers": ["Chai tapri debates", "Jugaad philosophy", "Sharma ji ka beta comparison", "Metro rush", "Hostel life"],
        "slang_particles": ["bhai", "yaar", "guru", "boss", "arre"],
        "style_notes": "Everyday witty banter, conversational sarcasm, crisp punchlines, natural Delhi/Mumbai spoken Hindi.",
    },
    "ta-IN": {
        "culture_markers": ["Filter coffee ritual", "Cinema mass punchlines", "Chennai rain & auto drivers", "Tanglish banter"],
        "slang_particles": ["thalaiva", "machi", "pa", "dosthu", "ennada"],
        "style_notes": "Sharp, witty sarcasm, deadpan mass comebacks, natural spoken Tamil without formal Chennai textbook grammar.",
    },
    "kn-IN": {
        "culture_markers": ["Bangalore traffic eternity", "Filter kaapi", "Silk Board signals", "Weekend plan illusions"],
        "slang_particles": ["macha", "guru", "bro", "saar"],
        "style_notes": "Casual conversational Kannada, witty understatement, relatable Bangalore & regional youth humour.",
    },
    "ml-IN": {
        "culture_markers": ["Chai & parotta", "Cinema satire traditions", "Monsoon drama", "Gulf uncle returns"],
        "slang_particles": ["aliya", "bro", "chettan", "mwone"],
        "style_notes": "Dry, intelligent satire, natural spoken Malayalam, subtle sarcastic wit.",
    },
    "bn-IN": {
        "culture_markers": ["Adda debates", "Fish market bargaining", "Literary pretension vs lazy naps", "Durga Puja excitement"],
        "slang_particles": ["bhai", "dada", "guru", "babu"],
        "style_notes": "Witty observational intellect, playful sarcasm, natural Kolkata spoken Bengali.",
    },
    "mr-IN": {
        "culture_markers": ["Vada pav supremacy", "Mumbai local train struggles", "Puneri deadpan sarcasm", "Ganpati energy"],
        "slang_particles": ["bhawa", "mitra", "dost", "arre"],
        "style_notes": "Sharp, dry Puneri-style sarcastic comebacks, rich everyday spoken Marathi.",
    },
    "gu-IN": {
        "culture_markers": ["Business mindset", "Fafda-jalebi mornings", "Discount negotiation", "Navratri stamina"],
        "slang_particles": ["bhai", "yaar", "dost", "kaka"],
        "style_notes": "Good-natured practical roasting, business irony, casual Gujarati speech.",
    },
    "pa-IN": {
        "culture_markers": ["Gaddi & swag boasting", "Lassi & butter feast", "Canada visa dreams", "Bhangra energy"],
        "slang_particles": ["veer", "bai", "prava", "yaara"],
        "style_notes": "High-energy bold comebacks, friendly boast deflation, earthy Punjabi wit.",
    },
    "en-IN": {
        "culture_markers": ["Urban metro life", "LinkedIn corporate cringe", "Swiggy/Zomato addiction", "Dating app fatigue"],
        "slang_particles": ["bro", "dude", "man"],
        "style_notes": "Snappy, razor-sharp everyday English, modern meme-literate delivery, zero corporate jargon.",
    },
}


def get_humour_concept(topic: ConversationTopic) -> HumourConcept:
    """Retrieve the primary creative humour concept for a given conversation topic."""
    concepts = HUMOUR_KNOWLEDGE_BASE.get(topic)
    if concepts:
        return concepts[0]
    return HUMOUR_KNOWLEDGE_BASE[ConversationTopic.EVERYDAY_CASUAL][0]


def get_regional_flavour(language_code: str) -> Dict[str, Any]:
    """Retrieve regional cultural markers and style guidance for a language."""
    from backend.services.languages import normalize_language_code
    normalized = normalize_language_code(language_code)
    return REGIONAL_HUMOUR_FLAVOURS.get(normalized, REGIONAL_HUMOUR_FLAVOURS["en-IN"])
