"""
TheONE AI - Complete Multilingual Service & Language Detection Engine.

Provides unified support for:
1. English (en-IN)
2. Telugu (te-IN)
3. Hindi (hi-IN)
4. Tamil (ta-IN)
5. Kannada (kn-IN)
6. Malayalam (ml-IN)
7. Bengali (bn-IN)
8. Marathi (mr-IN)
9. Gujarati (gu-IN)
10. Punjabi (pa-IN)

Includes automatic language detection via Unicode script analysis,
code-mixed marker detection, Gnani Prisma STT language mapping,
and Gnani Timbre TTS voice-model compatibility resolution.
"""

from dataclasses import dataclass
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("theone.languages")

INDIC_AUTO_CODE = "indic-auto"
DEFAULT_LANGUAGE = "en-IN"


@dataclass(frozen=True)
class LanguageMeta:
    code: str
    iso: str
    name: str
    native_name: str
    script_family: str
    default_female_voice: str
    default_male_voice: str


SUPPORTED_LANGUAGES: Dict[str, LanguageMeta] = {
    "en-IN": LanguageMeta(
        code="en-IN",
        iso="en",
        name="English (India)",
        native_name="English",
        script_family="latin",
        default_female_voice="Yashvi",
        default_male_voice="Zayan",
    ),
    "te-IN": LanguageMeta(
        code="te-IN",
        iso="te",
        name="Telugu",
        native_name="తెలుగు",
        script_family="telugu",
        default_female_voice="Yashvi",
        default_male_voice="Zayan",
    ),
    "hi-IN": LanguageMeta(
        code="hi-IN",
        iso="hi",
        name="Hindi",
        native_name="हिन्दी",
        script_family="devanagari",
        default_female_voice="Yashvi",
        default_male_voice="Zayan",
    ),
    "ta-IN": LanguageMeta(
        code="ta-IN",
        iso="ta",
        name="Tamil",
        native_name="தமிழ்",
        script_family="tamil",
        default_female_voice="Yashvi",
        default_male_voice="Zayan",
    ),
    "kn-IN": LanguageMeta(
        code="kn-IN",
        iso="kn",
        name="Kannada",
        native_name="ಕನ್ನಡ",
        script_family="kannada",
        default_female_voice="Yashvi",
        default_male_voice="Zayan",
    ),
    "ml-IN": LanguageMeta(
        code="ml-IN",
        iso="ml",
        name="Malayalam",
        native_name="മലയാളം",
        script_family="malayalam",
        default_female_voice="Yashvi",
        default_male_voice="Zayan",
    ),
    "bn-IN": LanguageMeta(
        code="bn-IN",
        iso="bn",
        name="Bengali",
        native_name="বাংলা",
        script_family="bengali",
        default_female_voice="Yashvi",
        default_male_voice="Zayan",
    ),
    "mr-IN": LanguageMeta(
        code="mr-IN",
        iso="mr",
        name="Marathi",
        native_name="मराठी",
        script_family="devanagari",
        default_female_voice="Yashvi",
        default_male_voice="Zayan",
    ),
    "gu-IN": LanguageMeta(
        code="gu-IN",
        iso="gu",
        name="Gujarati",
        native_name="ગુજરાતી",
        script_family="gujarati",
        default_female_voice="Yashvi",
        default_male_voice="Zayan",
    ),
    "pa-IN": LanguageMeta(
        code="pa-IN",
        iso="pa",
        name="Punjabi",
        native_name="ਪੰਜਾਬੀ",
        script_family="gurmukhi",
        default_female_voice="Yashvi",
        default_male_voice="Zayan",
    ),
}

# Normalization mapping for common aliases and ISO-639 codes
LANGUAGE_ALIAS_MAP: Dict[str, str] = {
    # English
    "en": "en-IN",
    "eng": "en-IN",
    "en-in": "en-IN",
    "en_in": "en-IN",
    "en-us": "en-IN",
    "english": "en-IN",
    # Telugu
    "te": "te-IN",
    "tel": "te-IN",
    "te-in": "te-IN",
    "te_in": "te-IN",
    "telugu": "te-IN",
    "tenglish": "te-IN",
    # Hindi
    "hi": "hi-IN",
    "hin": "hi-IN",
    "hi-in": "hi-IN",
    "hi_in": "hi-IN",
    "hindi": "hi-IN",
    "hinglish": "hi-IN",
    # Tamil
    "ta": "ta-IN",
    "tam": "ta-IN",
    "ta-in": "ta-IN",
    "ta_in": "ta-IN",
    "tamil": "ta-IN",
    "tanglish": "ta-IN",
    # Kannada
    "kn": "kn-IN",
    "kan": "kn-IN",
    "kn-in": "kn-IN",
    "kn_in": "kn-IN",
    "kannada": "kn-IN",
    "kanglish": "kn-IN",
    # Malayalam
    "ml": "ml-IN",
    "mal": "ml-IN",
    "ml-in": "ml-IN",
    "ml_in": "ml-IN",
    "malayalam": "ml-IN",
    "manglish": "ml-IN",
    # Bengali
    "bn": "bn-IN",
    "ben": "bn-IN",
    "bn-in": "bn-IN",
    "bn_in": "bn-IN",
    "bengali": "bn-IN",
    "bangla": "bn-IN",
    # Marathi
    "mr": "mr-IN",
    "mar": "mr-IN",
    "mr-in": "mr-IN",
    "mr_in": "mr-IN",
    "marathi": "mr-IN",
    # Gujarati
    "gu": "gu-IN",
    "guj": "gu-IN",
    "gu-in": "gu-IN",
    "gu_in": "gu-IN",
    "gujarati": "gu-IN",
    # Punjabi
    "pa": "pa-IN",
    "pan": "pa-IN",
    "pa-in": "pa-IN",
    "pa_in": "pa-IN",
    "punjabi": "pa-IN",
    # Auto
    "auto": INDIC_AUTO_CODE,
    "indic-auto": INDIC_AUTO_CODE,
    "indic_auto": INDIC_AUTO_CODE,
}

# Distinctive romanized keywords for code-mixed / conversational Indian dialects
CODEMIX_MARKERS: Dict[str, Set[str]] = {
    "te-IN": {
        "enti", "cheppu", "cheppava", "chala", "ledu", "kadha", "kadu", "naku", "neeku",
        "chestunna", "ayipoyindi", "ra", "akkada", "ikkada", "emiti", "chudu", "unnava",
        "unnadu", "bagunnava", "lekapothe", "avuna", "nijama", "tinesi", "velli", "randi",
        "bore", "bavundi", "cheyyaku", "oddu", "pelli", "koddiga",
    },
    "hi-IN": {
        "kya", "nahi", "nahin", "hai", "hain", "bhai", "yaar", "tera", "mera", "kaise",
        "kaisa", "karo", "karna", "raha", "rahi", "rahe", "aaj", "kal", "bohot", "bahut",
        "dekh", "suno", "bol", "kuch", "apna", "mujhe", "tujhe", "hoga", "hogi", "shuru",
        "padhai", "mast", "theek", "batao", "chal", "ruk",
    },
    "ta-IN": {
        "enna", "solre", "sollu", "machan", "machi", "illa", "illai", "panra", "irukku",
        "irukka", "epdi", "romba", "thalaiva", "nalla", "poitu", "vaa", "poda", "podi",
        "theriyum", "theriyathu", "kudunga", "pa", "aama", "appadi",
    },
    "kn-IN": {
        "enu", "hege", "illa", "agide", "madu", "beku", "nange", "ninage", "gottha",
        "gothilla", "chennagidira", "hogu", "baa", "mado", "nodri", "dayavittu",
    },
    "ml-IN": {
        "entha", "engane", "illa", "undu", "cheyyanam", "cheythu", "evide", "njan", "ente",
        "ningal", "sukhamano", "poyi", "varaam", "sheriya", "nannayi",
    },
    "bn-IN": {
        "ki", "korcho", "bhalo", "shob", "koro", "amra", "tumi", "karon", "ekhon", "kothay",
        "khobor", "bolo", "hobe", "aami", "tomar",
    },
    "mr-IN": {
        "kay", "ahe", "nahi", "kasa", "kashi", "kase", "bhava", "jhala", "karto", "mala",
        "tula", "kuthe", "sang", "bara", "pan", "aapan",
    },
    "gu-IN": {
        "shu", "che", "kem", "nathi", "tame", "maro", "taro", "bhai", "karo", "su", "kya",
        "maja", "saru", "aavo", "jova",
    },
    "pa-IN": {
        "ki", "kivein", "honda", "changa", "bai", "veere", "painda", "karange", "kitta",
        "oye", "gal", "dass", "tussi", "assi", "hanji",
    },
}

# Specific Marathi words written in Devanagari to distinguish from Hindi (excluding words shared with Hindi)
MARATHI_DEVANAGARI_WORDS: Set[str] = {
    "आहे", "नाही", "काय", "कसा", "कशी", "कसे", "तुला", "मला", "भावा", "झाला", "झाली",
    "झाले", "केला", "केली", "केले", "करतो", "करते", "करायचा", "होतो", "होती", "आहोत",
    "कुठे", "सांग", "कळलं", "बरं", "नको", "आलास",
}


def normalize_language_code(code: Optional[str]) -> str:
    """
    Normalize any candidate language string or alias into canonical BCP-47 form
    (e.g., 'te' -> 'te-IN', 'hindi' -> 'hi-IN', 'auto' -> 'indic-auto').
    """
    if not code:
        return DEFAULT_LANGUAGE
    cleaned = code.strip().lower().replace("_", "-")
    if cleaned in LANGUAGE_ALIAS_MAP:
        return LANGUAGE_ALIAS_MAP[cleaned]
    # Check for direct match case-insensitively
    for canonical in SUPPORTED_LANGUAGES:
        if canonical.lower() == cleaned:
            return canonical
    return DEFAULT_LANGUAGE


def detect_script_language(text: str) -> Optional[str]:
    """
    Detect Indian language from native Unicode script blocks.
    Returns canonical code (e.g., 'te-IN') or None if mostly Latin / other.
    """
    if not text:
        return None

    counts: Dict[str, int] = {
        "te-IN": 0,
        "hi_or_mr": 0,
        "ta-IN": 0,
        "kn-IN": 0,
        "ml-IN": 0,
        "bn-IN": 0,
        "gu-IN": 0,
        "pa-IN": 0,
    }

    has_marathi_unique_char = False

    for ch in text:
        cp = ord(ch)
        if 0x0C00 <= cp <= 0x0C7F:
            counts["te-IN"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["hi_or_mr"] += 1
            if cp == 0x0933:  # Devanagari letter LLA (ळ), heavily used in Marathi
                has_marathi_unique_char = True
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["ta-IN"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            counts["kn-IN"] += 1
        elif 0x0D00 <= cp <= 0x0D7F:
            counts["ml-IN"] += 1
        elif 0x0980 <= cp <= 0x09FF:
            counts["bn-IN"] += 1
        elif 0x0A80 <= cp <= 0x0AFF:
            counts["gu-IN"] += 1
        elif 0x0A00 <= cp <= 0x0A7F:
            counts["pa-IN"] += 1

    max_lang, max_count = max(counts.items(), key=lambda x: x[1])
    if max_count == 0:
        return None

    if max_lang == "hi_or_mr":
        if has_marathi_unique_char:
            return "mr-IN"
        # Check Marathi vocabulary words
        words = set(re.findall(r"[\u0900-\u097F]+", text))
        if words.intersection(MARATHI_DEVANAGARI_WORDS):
            return "mr-IN"
        return "hi-IN"

    return max_lang


def detect_codemixed_language(text: str) -> Optional[str]:
    """
    Detect Indian language when written in Roman/Latin script (e.g. Hinglish, Tenglish).
    """
    if not text:
        return None

    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    if not words:
        return None

    word_set = set(words)
    best_lang = None
    best_matches = 0

    for lang, markers in CODEMIX_MARKERS.items():
        matches = len(word_set.intersection(markers))
        if matches > best_matches:
            best_matches = matches
            best_lang = lang

    return best_lang if best_matches >= 1 else None


def detect_language(text: str, fallback_language: str = DEFAULT_LANGUAGE) -> str:
    """
    Comprehensive automatic language detection for user speech/text.
    1. Checks native Indic scripts.
    2. Checks code-mixed romanized keywords.
    3. Falls back to fallback_language or 'en-IN'.
    """
    if not text or not text.strip():
        return normalize_language_code(fallback_language)

    # 1. Native script detection (high confidence)
    script_lang = detect_script_language(text)
    if script_lang:
        return script_lang

    # 2. Code-mixed / Romanized detection
    codemix_lang = detect_codemixed_language(text)
    if codemix_lang:
        return codemix_lang

    # 3. Fallback
    return normalize_language_code(fallback_language)


def get_language_prompt_instruction(language_code: str) -> str:
    """
    Generate language-specific system prompt instructions and few-shot examples
    so Gemma generates direct, natural, funny responses in the target language.
    """
    norm = normalize_language_code(language_code)

    instructions: Dict[str, str] = {
        "te-IN": (
            "CRITICAL LANGUAGE INSTRUCTION: The user is speaking/messaging in TELUGU (తెలుగు).\n"
            "- You MUST reply directly in NATURAL EVERYDAY SPOKEN TELUGU (తెలుగు) or conversational Tenglish if the user used English script.\n"
            "- DO NOT reply in English. DO NOT translate into English.\n"
            "- Keep TheONE AI's core identity: bold, sarcastic, playfully savage friend.\n"
            "- Strictly 1 single sentence. 3 to 10 words (maximum 15 words).\n"
            "Examples:\n"
            "User: నేను రేపటి నుంచి చదువుతాను.\n"
            "AI: రేపు ఆల్రెడీ నిన్ను బ్లాక్ చేసేసింది బ్రో.\n"
            "User: నాకు చాలా బోర్ కొడుతోంది.\n"
            "AI: బోర్ కూడా నీతో ఉండి బోర్ ఫీలవుతోంది.\n"
            "User: నేను ఎగ్జామ్ లో ఫెయిల్ అయ్యా.\n"
            "AI: క్వశ్చన్ పేపర్ గెలిచింది, నువ్వు విశ్రాంతి తీసుకో.\n"
            "User: Naku chala bore kodutondi bro.\n"
            "AI: Bro bore kuda nitho bore aypoyindi."
        ),
        "hi-IN": (
            "CRITICAL LANGUAGE INSTRUCTION: The user is speaking/messaging in HINDI (हिन्दी).\n"
            "- You MUST reply directly in NATURAL EVERYDAY SPOKEN HINDI (हिन्दी) or conversational Hinglish if the user used English script.\n"
            "- DO NOT reply in English. DO NOT translate into English.\n"
            "- Keep TheONE AI's core identity: bold, sarcastic, playfully savage friend.\n"
            "- Strictly 1 single sentence. 3 to 10 words (maximum 15 words).\n"
            "Examples:\n"
            "User: मैं कल से पढूंगा।\n"
            "AI: भाई, कल ने तुझे पहले ही ब्लॉक कर दिया।\n"
            "User: मुझे बहुत बोरियत हो रही है।\n"
            "AI: तुझसे तो बोरियत भी बोर हो गई भाई।\n"
            "User: मैं बहुत समझदार हूं।\n"
            "AI: हां भाई, तेरा आईना तो गर्व से फूल गया होगा।\n"
            "User: Aaj padhai karne ka mann nahi hai bro.\n"
            "AI: Tera mann to padhai ke liye bana hi nahi tha."
        ),
        "ta-IN": (
            "CRITICAL LANGUAGE INSTRUCTION: The user is speaking/messaging in TAMIL (தமிழ்).\n"
            "- You MUST reply directly in NATURAL EVERYDAY SPOKEN TAMIL (தமிழ்) or conversational Tanglish if the user used English script.\n"
            "- DO NOT reply in English. DO NOT translate into English.\n"
            "- Keep TheONE AI's core identity: bold, sarcastic, playfully savage friend.\n"
            "- Strictly 1 single sentence. 3 to 10 words (maximum 15 words).\n"
            "Examples:\n"
            "User: நான் நாளைக்கு படிக்க போறேன்.\n"
            "AI: தலைவா, நாளைக்கு உன்னை ஆல்ரெடி பிளாக் பண்ணிடுச்சு.\n"
            "User: எனக்கு ரொம்ப போர் அடிக்குது.\n"
            "AI: உன்ன பாத்து போரே போர் அடிச்சி போயிடுச்சு.\n"
            "User: நான் ரொம்ப அறிவாளி.\n"
            "AI: ஆமா, கண்ணாடி முன்னாடி போய் இத சொல்லாத."
        ),
        "kn-IN": (
            "CRITICAL LANGUAGE INSTRUCTION: The user is speaking/messaging in KANNADA (ಕನ್ನಡ).\n"
            "- You MUST reply directly in NATURAL EVERYDAY SPOKEN KANNADA (ಕನ್ನಡ) or conversational Kanglish.\n"
            "- DO NOT reply in English. DO NOT translate into English.\n"
            "- Keep TheONE AI's core identity: bold, sarcastic, playfully savage friend.\n"
            "- Strictly 1 single sentence. 3 to 10 words (maximum 15 words).\n"
            "Examples:\n"
            "User: ನಾನು ನಾಳೆಯಿಂದ ಓದುತ್ತೇನೆ.\n"
            "AI: ಬ್ರೋ, ನಾಳೆ ಆಲ್ರೆಡಿ ನಿನ್ನನ್ನು ಬ್ಲಾಕ್ ಮಾಡಿದೆ.\n"
            "User: ನನಗೆ ತುಂಬಾ ಬೋರ್ ಆಗ್ತಿದೆ.\n"
            "AI: ನಿನ್ನನ್ನು ನೋಡಿ ಬೋರ್ ಕೂಡ ಬೋರ್ ಆಗ್ತಿದೆ."
        ),
        "ml-IN": (
            "CRITICAL LANGUAGE INSTRUCTION: The user is speaking/messaging in MALAYALAM (മലയാളം).\n"
            "- You MUST reply directly in NATURAL EVERYDAY SPOKEN MALAYALAM (മലയാളം) or conversational Manglish.\n"
            "- DO NOT reply in English. DO NOT translate into English.\n"
            "- Keep TheONE AI's core identity: bold, sarcastic, playfully savage friend.\n"
            "- Strictly 1 single sentence. 3 to 10 words (maximum 15 words).\n"
            "Examples:\n"
            "User: ഞാൻ നാളെ മുതൽ പഠിക്കും.\n"
            "AI: ബ്രോ, നാളെ നിന്നെ ഓൾറെഡി ബ്ലോക്ക് ചെയ്തു.\n"
            "User: എനിക്ക് ഭയങ്കര ബോറടിക്കുന്നു.\n"
            "AI: നിന്നെ കണ്ടിട്ട് ബോറിനും ബോറടിച്ചു."
        ),
        "bn-IN": (
            "CRITICAL LANGUAGE INSTRUCTION: The user is speaking/messaging in BENGALI (বাংলা).\n"
            "- You MUST reply directly in NATURAL EVERYDAY SPOKEN BENGALI (বাংলা) or conversational Benglish.\n"
            "- DO NOT reply in English. DO NOT translate into English.\n"
            "- Keep TheONE AI's core identity: bold, sarcastic, playfully savage friend.\n"
            "- Strictly 1 single sentence. 3 to 10 words (maximum 15 words).\n"
            "Examples:\n"
            "User: আমি কাল থেকে পড়ব।\n"
            "AI: ভাই, কাল তোকে আগেই ব্লক করে দিয়েছে।\n"
            "User: আমার খুব বোরিং লাগছে।\n"
            "AI: তোর সাথে থেকে বোরডমও বোর হয়ে গেছে।"
        ),
        "mr-IN": (
            "CRITICAL LANGUAGE INSTRUCTION: The user is speaking/messaging in MARATHI (मराठी).\n"
            "- You MUST reply directly in NATURAL EVERYDAY SPOKEN MARATHI (मराठी) or conversational Minglish.\n"
            "- DO NOT reply in English. DO NOT translate into English.\n"
            "- Keep TheONE AI's core identity: bold, sarcastic, playfully savage friend.\n"
            "- Strictly 1 single sentence. 3 to 10 words (maximum 15 words).\n"
            "Examples:\n"
            "User: मी उद्यापासून अभ्यास करणार.\n"
            "AI: भावा, उद्याने तुला आधीच ब्लॉक केलंय.\n"
            "User: मला खूप कंटाळा आलाय.\n"
            "AI: तुझ्या कंटाळ्यालाही तुझा कंटाळा आला असेल."
        ),
        "gu-IN": (
            "CRITICAL LANGUAGE INSTRUCTION: The user is speaking/messaging in GUJARATI (ગુજરાતી).\n"
            "- You MUST reply directly in NATURAL EVERYDAY SPOKEN GUJARATI (ગુજરાતી) or conversational Gujlish.\n"
            "- DO NOT reply in English. DO NOT translate into English.\n"
            "- Keep TheONE AI's core identity: bold, sarcastic, playfully savage friend.\n"
            "- Strictly 1 single sentence. 3 to 10 words (maximum 15 words).\n"
            "Examples:\n"
            "User: હું કાલથી વાંચીશ.\n"
            "AI: ભાઈ, કાલે તને પહેલેથી બ્લોક કરી દીધો છે.\n"
            "User: મને બહુ કંટાળો આવે છે.\n"
            "AI: તારાથી તો કંટાળો પણ કંટાળી ગયો ભાઈ."
        ),
        "pa-IN": (
            "CRITICAL LANGUAGE INSTRUCTION: The user is speaking/messaging in PUNJABI (ਪੰਜਾਬੀ).\n"
            "- You MUST reply directly in NATURAL EVERYDAY SPOKEN PUNJABI (ਪੰਜਾਬੀ) or conversational Punglish.\n"
            "- DO NOT reply in English. DO NOT translate into English.\n"
            "- Keep TheONE AI's core identity: bold, sarcastic, playfully savage friend.\n"
            "- Strictly 1 single sentence. 3 to 10 words (maximum 15 words).\n"
            "Examples:\n"
            "User: ਮੈਂ ਕੱਲ੍ਹ ਤੋਂ ਪੜ੍ਹਾਂਗਾ।\n"
            "AI: ਬਾਈ, ਕੱਲ੍ਹ ਨੇ ਤੈਨੂੰ ਪਹਿਲਾਂ ਹੀ ਬਲਾਕ ਕਰਤਾ।\n"
            "User: ਮੈਨੂੰ ਬਹੁਤ ਬੋਰਿੰਗ ਲੱਗ ਰਿਹਾ।\n"
            "AI: ਤੇਰੇ ਕੋਲ ਰਹਿ ਕੇ ਤਾਂ ਬੋਰ ਵੀ ਬੋਰ ਹੋ ਗਿਆ।"
        ),
        "en-IN": (
            "CRITICAL LANGUAGE INSTRUCTION: The user is speaking/messaging in ENGLISH.\n"
            "- Reply in simple, everyday spoken conversational English like a naturally funny savage friend.\n"
            "- Strictly 1 single sentence. 3 to 10 words (maximum 15 words).\n"
            "Examples:\n"
            "User: I'm going to study tomorrow.\n"
            "AI: Bro, tomorrow has blocked you already.\n"
            "User: I'm feeling bored.\n"
            "AI: Even boredom got tired of hanging out with you."
        ),
    }

    return instructions.get(norm, instructions["en-IN"])


def resolve_tts_voice_and_model(
    requested_voice: Optional[str],
    language_code: str,
    configured_model: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve Gnani Timbre TTS voice and model compatibility.

    Rules:
    - In Gnani Timbre (timbre-v2.5), user voices 'Yashvi' (female) and 'Zayan' (male)
      natively support all 10 major Indian languages.
    - If user selected 'Yashvi' -> voice 'Yashvi', model 'timbre-v2.5'.
    - If user selected 'Zayan' -> voice 'Zayan', model 'timbre-v2.5'.
    - If a v3 voice is explicitly requested (e.g., 'Shivani', 'Avinash'),
      model is set to 'timbre-v3.0'.
    - If no voice requested, defaults to 'Yashvi' for unified comedy.
    """
    clean_voice = (requested_voice or "Yashvi").strip()
    norm_lang = normalize_language_code(language_code)

    # v3 voice set
    v3_voices = {
        "avinash", "shivani", "lakshmi", "kalpana", "shubham", "prakash",
        "devansh", "yashraj", "chetana", "vasudha", "harshad", "riyaa", "tarika", "kavitha",
    }

    if clean_voice.lower() in v3_voices:
        return clean_voice, "timbre-v3.0"

    # Default to user's selected v2.5 voice (Yashvi or Zayan)
    for canonical in ["Yashvi", "Zayan", "Nalini", "Kaveri", "Deepak"]:
        if canonical.lower() == clean_voice.lower():
            return canonical, "timbre-v2.5"

    # Fallback to Yashvi on timbre-v2.5
    return "Yashvi", "timbre-v2.5"
