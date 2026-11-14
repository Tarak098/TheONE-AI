"""
TheONE AI - Natural Spoken Voice-Switching Intent Service.

Recognizes natural voice-switching requests across English, 9 Indian languages,
and code-mixed commands. Distinguishes target voice requests (Yashvi vs Zayan),
handles relative switches ("previous voice", "other voice", "male voice", "female voice"),
and responds cleanly to ambiguous requests ("Change your voice" -> "Yashvi or Zayan?").
"""

from dataclasses import dataclass
import logging
import re
from typing import Optional

logger = logging.getLogger("theone.voice_intent")


@dataclass
class VoiceSwitchResult:
    """Outcome of voice-switch intent analysis."""
    is_voice_switch: bool
    target_voice: Optional[str]  # "Yashvi" | "Zayan" | None (if ambiguous)
    is_ambiguous: bool
    confirmation_text: str
    voice_to_speak: str  # Voice that will speak the confirmation aloud


# Confirmation phrases per language
CONFIRMATIONS_ZAYAN = {
    "en-IN": "New voice, same attitude.",
    "te-IN": "జాయన్ వాయిస్‌కి మారాను. చెప్పు బ్రో.",
    "hi-IN": "ज़यान की आवाज़ सेट कर दी। बोलो भाई।",
    "ta-IN": "சயான் குரலுக்கு மாறியாச்சு. சொல்லுங்க தலைவா.",
    "kn-IN": "ಜಯಾನ್ ಧ್ವನಿಗೆ ಬದಲಾಯಿಸಲಾಗಿದೆ. ಹೇಳಿ ಬ್ರೋ.",
    "ml-IN": "സയാന്റെ ശബ്ദത്തിലേക്ക് മാറ്റി. പറയൂ ബ്രോ.",
    "bn-IN": "জায়ান ভয়েসে সুইচ করেছি। বলো ভাই।",
    "mr-IN": "झयानच्या आवाजात आलोय. बोल भावा.",
    "gu-IN": "ઝયાનના અવાજમાં સ્વિચ કર્યું. બોલો ભાઈ.",
    "pa-IN": "ਜ਼ਯਾਨ ਦੀ ਆਵਾਜ਼ ਚਲਾ ਦਿੱਤੀ। ਦੱਸੋ ਬਾਈ।",
}

CONFIRMATIONS_YASHVI = {
    "en-IN": "Missed this voice? Back in Yashvi's tone.",
    "te-IN": "యశ్వి వాయిస్ మళ్లీ వచ్చేసింది. చెప్పు బ్రో.",
    "hi-IN": "यशवी की आवाज़ वापस आ गई। बोलो भाई।",
    "ta-IN": "யஷ்வி குரல் திரும்பியாச்சு. சொல்லுங்க.",
    "kn-IN": "ಯಶ್ವಿ ಧ್ವನಿ ವಾಪಸ್ ಬಂದಿದೆ. ಹೇಳಿ.",
    "ml-IN": "യശ്വിയുടെ ശബ്ദം തിരികെ എത്തി. പറയൂ.",
    "bn-IN": "যশবী ভয়েস ফিরে এসেছে। বলো।",
    "mr-IN": "यशवीचा आवाज परत आलाय. बोल.",
    "gu-IN": "યશવીનો અવાજ પાછો આવી ગયો. બોલો.",
    "pa-IN": "ਯਸ਼ਵੀ ਦੀ ਆਵਾਜ਼ ਵਾਪਸ ਆ ਗਈ। ਦੱਸੋ।",
}

CONFIRMATIONS_AMBIGUOUS = {
    "en-IN": "Yashvi or Zayan?",
    "te-IN": "యశ్వి లేదా జాయన్?",
    "hi-IN": "यशवी या ज़यान?",
    "ta-IN": "யஷ்வி அல்லது சயான்?",
    "kn-IN": "ಯಶ್ವಿ ಅಥವಾ ಜಯಾನ್?",
    "ml-IN": "യശ്വി അതോ സയാൻ?",
    "bn-IN": "যশবী নাকি জায়ান?",
    "mr-IN": "यशवी की झयान?",
    "gu-IN": "યશવી કે ઝયાન?",
    "pa-IN": "ਯਸ਼ਵੀ ਜਾਂ ਜ਼ਯਾਨ?",
}


# Intent detection patterns supporting English and Indic scripts
VOICE_TRIGGER_WORDS = re.compile(
    r"\b(?:voice|voices|sound|tone|accent|awaz|aawaz|awaaz|dhwani|shabdam|gola|kural)\b|"
    r"(?:వాయిస్|స్వర|ఆవాజ్|శబ్దం|ధ్వని|"
    r"आवाज़|आवाज|ध्वनि|गळा|"
    r"குரல்|குரலில்|குரலுக்கு|"
    r"ಧ್ವನಿ|ಧ್ವನಿಗೆ|ಶಬ್ದ|"
    r"ശബ്ദം|ശബ്ദത്തിലേക്ക്|"
    r"ভয়েস|ভয়েস|আওয়াজ|শব্দ|"
    r"अवाज़|अवाज|અવાજ|અવાજમાં|"
    r"ਆਵਾਜ਼|ਆਵਾਜ)",
    re.I,
)

SWITCH_TRIGGER_WORDS = re.compile(
    r"\b(?:switch|change|use|talk|speak|set|turn|give|want|make|shift|toggle|put|bolo|matladu|pesu|"
    r"marchu|maathu|badla|badal|maadi|podu|aakku|koro)\b|"
    r"(?:మార్చు|మారు|మారండి|మాట్లాడు|చెప్పు|పెట్టు|"
    r"बदलो|बदलिए|लगाओ|लगाना|चलाओ|बोलो|बात|करो|बोल|"
    r"மாற்று|மாறு|பேசு|சொல்லு|"
    r"ಬದಲಾಯಿಸಿ|ಬದಲು|ಮಾತನಾಡಿ|ಹೇಳಿ|"
    r"മാറ്റുക|മാറ്റ്|സംസാരിക്കുക|പറയൂ|"
    r"বদল|সুইচ|বলো|"
    r"बदल|करा|बोला|"
    r"બદલો|કરો|બોલો|"
    r"ਬਦਲੋ|ਚਲਾਓ|ਬੋਲੋ)",
    re.I,
)

ZAYAN_PATTERNS = re.compile(
    r"\b(?:zayan|zayaan|zaayan|jayan|zhayan|male|boy|guy|man|dude|brother|bro voice|"
    r"ladka|ladke|abbayi|purusha|aankural|purushan|chekkante)\b|"
    r"(?:జాయన్|జయాన్|అబ్బాయి|మగ|పురుష|"
    r"ज़यान|जयान|झयान|लड़का|लड़के|मर्दाना|पुरुष|"
    r"சயான்|ஆண்|"
    r"ಜಯಾನ್|ಜಾಯನ್|ಗಂಡು|ಹುಡುಗ|ಪುರುಷ|"
    r"സയാൻ|സയാന്റെ|ആൺ|പുരുഷ|"
    r"জায়ান|জায়ান|ছেলে|পুরুষ|"
    r"मुलगा|पुरुष|"
    r"ઝયાન|જયાન|છોકરો|પુરુષ|"
    r"ਜ਼ਯਾਨ|ਜਯਾਨ|ਮੁੰਡਾ|ਮਰਦ)",
    re.I,
)

YASHVI_PATTERNS = re.compile(
    r"\b(?:yashvi|yasvi|yashwi|yaswi|female|girl|woman|lady|sister|"
    r"ladki|ammayi|mahila|penkural|penninte|meyeder)\b|"
    r"(?:యశ్వి|యశస్వి|అమ్మాయి|ఆడ|మహిళ|"
    r"यशवी|यशस्वी|लड़की|महिला|स्त्री|"
    r"யஷ்வி|பெண்|"
    r"ಯಶ್ವಿ|ಹೆಣ್ಣು|ಹುಡುಗಿ|ಮಹಿಳೆ|"
    r"യശ്വി|യശ്വിയുടെ|പെൺ|സ്ത്രീ|"
    r"যশবী|যশস্বী|মেয়ে|মহিলা|স্ত্রী|"
    r"मुलगी|स्त्री|महिला|"
    r"યશવી|છોકરી|મહિલા|"
    r"ਯਸ਼ਵੀ|ਕੁੜੀ|ਔਰਤ)",
    re.I,
)

RELATIVE_OTHER_PATTERNS = re.compile(
    r"\b(?:other voice|different voice|previous voice|switch back|"
    r"dusri awaz|dusra voice|vere voice|maro awaz|innoru voice|bere voice|matthoru voice)\b",
    re.I,
)

AMBIGUOUS_PATTERNS = re.compile(
    r"\b(?:change|switch|modify|different)\s+(?:your\s+)?voice\b|"
    r"\bswitch\s+voice\b|"
    r"\bchange\s+voice\b|"
    r"\b(?:awaz|voice)\s+(?:badlo|change|marchu|maathu)\b|"
    r"(?:వాయిస్\s*మార్చు|ఆవాజ్\s*బదలో|"
    r"आवाज़\s*बदलो|आवाज\s*बदलो|ध्वनि\s*बदलो|"
    r"குரல்\s*மாற்று|"
    r"ಧ್ವನಿ\s*ಬದಲಾಯಿಸಿ|"
    r"ശബ്ദം\s*മാറ്റുക|"
    r"ভয়েস\s*বদল|আওয়াজ\s*বদল|"
    r"आवाज\s*बदल|"
    r"અવાજ\s*બદલો|"
    r"ਆਵਾਜ਼\s*ਬਦਲੋ)",
    re.I,
)


def detect_voice_switch_intent(
    text: str,
    current_voice: str = "Yashvi",
    language_code: str = "en-IN",
) -> VoiceSwitchResult:
    """
    Analyze user text to determine if they intend to switch the TTS voice.

    Args:
        text: User's recognized speech or typed message.
        current_voice: The currently active voice ('Yashvi' or 'Zayan').
        language_code: Target BCP-47 language tag (e.g. 'te-IN', 'hi-IN').

    Returns:
        VoiceSwitchResult with target voice and localized confirmation.
    """
    if not text or not text.strip():
        return VoiceSwitchResult(
            is_voice_switch=False,
            target_voice=None,
            is_ambiguous=False,
            confirmation_text="",
            voice_to_speak=current_voice,
        )

    clean_text = text.strip()
    norm_lang = language_code if language_code in CONFIRMATIONS_ZAYAN else "en-IN"

    has_voice_word = bool(VOICE_TRIGGER_WORDS.search(clean_text))
    has_switch_word = bool(SWITCH_TRIGGER_WORDS.search(clean_text))
    has_zayan = bool(ZAYAN_PATTERNS.search(clean_text))
    has_yashvi = bool(YASHVI_PATTERNS.search(clean_text))
    has_relative = bool(RELATIVE_OTHER_PATTERNS.search(clean_text))
    has_ambiguous = bool(AMBIGUOUS_PATTERNS.search(clean_text))

    # Fast-check: Explicit voice name or gender command
    # e.g., "Switch to Zayan", "Change to Yashvi", "Talk in Zayan", "Use female voice"
    is_direct_zayan = has_zayan and (has_voice_word or has_switch_word or re.search(r"\b(?:to|in|with)\s+zayan\b", clean_text, re.I))
    is_direct_yashvi = has_yashvi and (has_voice_word or has_switch_word or re.search(r"\b(?:to|in|with)\s+yashvi\b", clean_text, re.I))

    # Single-word or direct commands: "Zayan voice", "Yashvi voice"
    if not is_direct_zayan and re.search(r"^\s*(?:switch\s+to\s+)?zayan(?:\s+voice)?\s*$", clean_text, re.I):
        is_direct_zayan = True
    if not is_direct_yashvi and re.search(r"^\s*(?:switch\s+to\s+)?yashvi(?:\s+voice)?\s*$", clean_text, re.I):
        is_direct_yashvi = True

    # 1. Target: Zayan
    if is_direct_zayan and not is_direct_yashvi:
        conf = CONFIRMATIONS_ZAYAN.get(norm_lang, CONFIRMATIONS_ZAYAN["en-IN"])
        logger.info("Voice switch detected -> Zayan (Lang: %s)", norm_lang)
        return VoiceSwitchResult(
            is_voice_switch=True,
            target_voice="Zayan",
            is_ambiguous=False,
            confirmation_text=conf,
            voice_to_speak="Zayan",
        )

    # 2. Target: Yashvi
    if is_direct_yashvi and not is_direct_zayan:
        conf = CONFIRMATIONS_YASHVI.get(norm_lang, CONFIRMATIONS_YASHVI["en-IN"])
        logger.info("Voice switch detected -> Yashvi (Lang: %s)", norm_lang)
        return VoiceSwitchResult(
            is_voice_switch=True,
            target_voice="Yashvi",
            is_ambiguous=False,
            confirmation_text=conf,
            voice_to_speak="Yashvi",
        )

    # 3. Relative switch: "Switch to other voice", "Use previous voice", "dusri awaz"
    if has_relative:
        new_voice = "Zayan" if current_voice.lower() == "yashvi" else "Yashvi"
        conf_map = CONFIRMATIONS_ZAYAN if new_voice == "Zayan" else CONFIRMATIONS_YASHVI
        conf = conf_map.get(norm_lang, conf_map["en-IN"])
        logger.info("Relative voice switch detected -> %s (Lang: %s)", new_voice, norm_lang)
        return VoiceSwitchResult(
            is_voice_switch=True,
            target_voice=new_voice,
            is_ambiguous=False,
            confirmation_text=conf,
            voice_to_speak=new_voice,
        )

    # 4. Ambiguous command: "Change your voice", "Switch voice", "Voice marchu", "Awaz badlo"
    if has_ambiguous or (has_voice_word and has_switch_word and not has_zayan and not has_yashvi):
        conf = CONFIRMATIONS_AMBIGUOUS.get(norm_lang, CONFIRMATIONS_AMBIGUOUS["en-IN"])
        logger.info("Ambiguous voice switch detected -> Asking 'Yashvi or Zayan?'")
        return VoiceSwitchResult(
            is_voice_switch=True,
            target_voice=None,
            is_ambiguous=True,
            confirmation_text=conf,
            voice_to_speak=current_voice,
        )

    # Not a voice switch request -> normal conversational message
    return VoiceSwitchResult(
        is_voice_switch=False,
        target_voice=None,
        is_ambiguous=False,
        confirmation_text="",
        voice_to_speak=current_voice,
    )
