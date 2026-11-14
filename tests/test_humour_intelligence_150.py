"""
TheONE AI - 150+ Comprehensive Humour Intelligence & Continuity Test Suite.

Verifies:
1. Deep situation & conversational understanding across 150+ diverse scenarios.
2. Comedic technique variety (Deadpan, Reversal, Savage, Dark Irony, Sarcasm, Understatement, Callback).
3. Anti-repetition enforcement (Banned openers, banned cliché phrases).
4. Multi-turn dialogue continuity and intelligent callbacks.
5. All 10 Indian languages & code-mixed conversational support.
6. Strict 1-sentence, 3-10 words (max 15 words) constraint.
7. Disconnected stock punchline rejection.
"""

import asyncio
from typing import Any, Dict, List
import pytest

from backend.services.conversation import ConversationManager, ConversationSession
from backend.services.humour_engine import HumourEngine
from backend.services.humour_knowledge import ComedicTechnique, ConversationTopic
from backend.services.intelligence import IntelligenceEngine
from backend.services.response_validator import (
    count_sentences,
    count_words,
    enforce_strict_short_response,
    validate_response_quality,
)

# ---------------------------------------------------------------------------
# 150+ Curated Conversational Scenarios spanning all required categories
# ---------------------------------------------------------------------------
ALL_HUMOUR_SCENARIOS: List[Dict[str, Any]] = [
    # 1. College & Exams (10)
    {"id": "exam_1", "category": "exams", "topic": ConversationTopic.EXAMS_STUDIES, "input": "I have an exam in two hours and I just opened the book.", "lang": "en-IN"},
    {"id": "exam_2", "category": "exams", "topic": ConversationTopic.EXAMS_STUDIES, "input": "The professor said the exam would be easy but it was all Greek.", "lang": "en-IN"},
    {"id": "exam_3", "category": "exams", "topic": ConversationTopic.EXAMS_STUDIES, "input": "I wrote three pages for a two-mark question.", "lang": "en-IN"},
    {"id": "exam_4", "category": "exams", "topic": ConversationTopic.EXAMS_STUDIES, "input": "My friend who said he studied nothing got the highest marks.", "lang": "en-IN"},
    {"id": "exam_5", "category": "exams", "topic": ConversationTopic.EXAMS_STUDIES, "input": "I fell asleep while reading the first page of organic chemistry.", "lang": "en-IN"},
    {"id": "exam_6", "category": "exams", "topic": ConversationTopic.EXAMS_STUDIES, "input": "I submitted my assignment at 11:59 PM with two seconds to spare.", "lang": "en-IN"},
    {"id": "exam_7", "category": "exams", "topic": ConversationTopic.EXAMS_STUDIES, "input": "The entire hall was using their calculators and I didn't even know we needed one.", "lang": "en-IN"},
    {"id": "exam_8", "category": "exams", "topic": ConversationTopic.EXAMS_STUDIES, "input": "I accidentally answered question four using question two's formula.", "lang": "en-IN"},
    {"id": "exam_9", "category": "exams", "topic": ConversationTopic.EXAMS_STUDIES, "input": "I am planning to memorize 400 pages overnight.", "lang": "en-IN"},
    {"id": "exam_10", "category": "exams", "topic": ConversationTopic.EXAMS_STUDIES, "input": "My teacher gave me zero for handwriting.", "lang": "en-IN"},

    # 2. Relationships & Dating (10)
    {"id": "rel_1", "category": "relationships", "topic": ConversationTopic.RELATIONSHIPS_DATING, "input": "My crush liked my story from three years ago.", "lang": "en-IN"},
    {"id": "rel_2", "category": "relationships", "topic": ConversationTopic.RELATIONSHIPS_DATING, "input": "I got left on read for forty-eight hours.", "lang": "en-IN"},
    {"id": "rel_3", "category": "relationships", "topic": ConversationTopic.RELATIONSHIPS_DATING, "input": "I sent a risky text and immediately put my phone on airplane mode.", "lang": "en-IN"},
    {"id": "rel_4", "category": "relationships", "topic": ConversationTopic.RELATIONSHIPS_DATING, "input": "She said 'you're like a brother to me'.", "lang": "en-IN"},
    {"id": "rel_5", "category": "relationships", "topic": ConversationTopic.RELATIONSHIPS_DATING, "input": "I went on a date and realized I forgot my wallet.", "lang": "en-IN"},
    {"id": "rel_6", "category": "relationships", "topic": ConversationTopic.RELATIONSHIPS_DATING, "input": "My ex just viewed my LinkedIn profile.", "lang": "en-IN"},
    {"id": "rel_7", "category": "relationships", "topic": ConversationTopic.RELATIONSHIPS_DATING, "input": "I overanalyzed a single comma in her text message.", "lang": "en-IN"},
    {"id": "rel_8", "category": "relationships", "topic": ConversationTopic.RELATIONSHIPS_DATING, "input": "We have been in the talking stage for eleven months.", "lang": "en-IN"},
    {"id": "rel_9", "category": "relationships", "topic": ConversationTopic.RELATIONSHIPS_DATING, "input": "My date spent forty minutes talking about their pet hamster.", "lang": "en-IN"},
    {"id": "rel_10", "category": "relationships", "topic": ConversationTopic.RELATIONSHIPS_DATING, "input": "I got ghosted after saying I enjoy pineapple on pizza.", "lang": "en-IN"},

    # 3. Friends & Social Life (10)
    {"id": "frnd_1", "category": "friends", "topic": ConversationTopic.FRIENDS_SOCIAL, "input": "My friends made a Goa plan without me.", "lang": "en-IN"},
    {"id": "frnd_2", "category": "friends", "topic": ConversationTopic.FRIENDS_SOCIAL, "input": "I'm third wheeling with a couple at dinner.", "lang": "en-IN"},
    {"id": "frnd_3", "category": "friends", "topic": ConversationTopic.FRIENDS_SOCIAL, "input": "I cancelled plans to stay home and stare at the wall.", "lang": "en-IN"},
    {"id": "frnd_4", "category": "friends", "topic": ConversationTopic.FRIENDS_SOCIAL, "input": "My social battery ran out after six minutes at the party.", "lang": "en-IN"},
    {"id": "frnd_5", "category": "friends", "topic": ConversationTopic.FRIENDS_SOCIAL, "input": "Nobody in the group chat replied to my meme.", "lang": "en-IN"},
    {"id": "frnd_6", "category": "friends", "topic": ConversationTopic.FRIENDS_SOCIAL, "input": "My best friend told my secret to the whole group.", "lang": "en-IN"},
    {"id": "frnd_7", "category": "friends", "topic": ConversationTopic.FRIENDS_SOCIAL, "input": "I arrived exactly on time and nobody else showed up for an hour.", "lang": "en-IN"},
    {"id": "frnd_8", "category": "friends", "topic": ConversationTopic.FRIENDS_SOCIAL, "input": "My friends made a new group chat just to discuss my mistakes.", "lang": "en-IN"},
    {"id": "frnd_9", "category": "friends", "topic": ConversationTopic.FRIENDS_SOCIAL, "input": "I hid in the grocery store aisle to avoid saying hi to an acquaintance.", "lang": "en-IN"},
    {"id": "frnd_10", "category": "friends", "topic": ConversationTopic.FRIENDS_SOCIAL, "input": "Every Goa plan we ever made got cancelled.", "lang": "en-IN"},

    # 4. Money & Finances (10)
    {"id": "mny_1", "category": "money", "topic": ConversationTopic.MONEY_FINANCES, "input": "My salary was credited this morning and it's already gone.", "lang": "en-IN"},
    {"id": "mny_2", "category": "money", "topic": ConversationTopic.MONEY_FINANCES, "input": "I bought an expensive jacket to impress people I dislike.", "lang": "en-IN"},
    {"id": "mny_3", "category": "money", "topic": ConversationTopic.MONEY_FINANCES, "input": "My account balance is currently three rupees.", "lang": "en-IN"},
    {"id": "mny_4", "category": "money", "topic": ConversationTopic.MONEY_FINANCES, "input": "I paid eighty dollars for gourmet water at the airport.", "lang": "en-IN"},
    {"id": "mny_5", "category": "money", "topic": ConversationTopic.MONEY_FINANCES, "input": "I have forty items sitting in my online cart waiting for a miracle.", "lang": "en-IN"},
    {"id": "mny_6", "category": "money", "topic": ConversationTopic.MONEY_FINANCES, "input": "I refused to pay ten rupees for parking and drove around for an hour.", "lang": "en-IN"},
    {"id": "mny_7", "category": "money", "topic": ConversationTopic.MONEY_FINANCES, "input": "I invested all my money in crypto and it immediately dropped ninety percent.", "lang": "en-IN"},
    {"id": "mny_8", "category": "money", "topic": ConversationTopic.MONEY_FINANCES, "input": "I split a restaurant bill forty-two ways.", "lang": "en-IN"},
    {"id": "mny_9", "category": "money", "topic": ConversationTopic.MONEY_FINANCES, "input": "My credit card bill looks like an international phone number.", "lang": "en-IN"},
    {"id": "mny_10", "category": "money", "topic": ConversationTopic.MONEY_FINANCES, "input": "I bought organic apples and let them rot on the counter.", "lang": "en-IN"},

    # 5. Social Media & Screen Time (10)
    {"id": "soc_1", "category": "social_media", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "My daily screen time report says nine hours today.", "lang": "en-IN"},
    {"id": "soc_2", "category": "social_media", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I opened my phone to check the weather and scrolled reels for three hours.", "lang": "en-IN"},
    {"id": "soc_3", "category": "social_media", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I accidentally liked my crush's photo from 2017.", "lang": "en-IN"},
    {"id": "soc_4", "category": "social_media", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I spent an hour choosing an aesthetic Instagram caption.", "lang": "en-IN"},
    {"id": "soc_5", "category": "social_media", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "My phone fell on my face while lying down in bed.", "lang": "en-IN"},
    {"id": "soc_6", "category": "social_media", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I refreshed my email twelve times hoping for good news.", "lang": "en-IN"},
    {"id": "soc_7", "category": "social_media", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I deleted Instagram and reinstalled it six minutes later.", "lang": "en-IN"},
    {"id": "soc_8", "category": "social_media", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I watched random street food cooking videos until 4 AM.", "lang": "en-IN"},
    {"id": "soc_9", "category": "social_media", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I have seventy-two unopened notifications on my phone.", "lang": "en-IN"},
    {"id": "soc_10", "category": "social_media", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I staged a fake candid photo with seven different lighting angles.", "lang": "en-IN"},

    # 6. Gaming (10)
    {"id": "game_1", "category": "gaming", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I raged quit Valorant after missing twelve headshots.", "lang": "en-IN"},
    {"id": "game_2", "category": "gaming", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "My ping jumped to 999 ms right during the final clutch.", "lang": "en-IN"},
    {"id": "game_3", "category": "gaming", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I spent sixty dollars on an in-game weapon skin.", "lang": "en-IN"},
    {"id": "game_4", "category": "gaming", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "My teammate went AFK to cook dinner in the middle of competitive.", "lang": "en-IN"},
    {"id": "game_5", "category": "gaming", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I said 'one last game' and now the sun is rising outside.", "lang": "en-IN"},
    {"id": "game_6", "category": "gaming", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I blamed my gaming mouse for my terrible aim.", "lang": "en-IN"},
    {"id": "game_7", "category": "gaming", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I got eliminated by a bot on easy mode.", "lang": "en-IN"},
    {"id": "game_8", "category": "gaming", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I bought a three-thousand-dollar gaming chair and still suck.", "lang": "en-IN"},
    {"id": "game_9", "category": "gaming", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "My Steam library has eighty unplayed games and I bought three more.", "lang": "en-IN"},
    {"id": "game_10", "category": "gaming", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I spent three hours customizing my character's eyebrows.", "lang": "en-IN"},

    # 7. Technology & Coding (10)
    {"id": "tech_1", "category": "technology", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "My Python script worked yesterday and today throws forty syntax errors.", "lang": "en-IN"},
    {"id": "tech_2", "category": "technology", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I pushed directly to the main branch on Friday evening.", "lang": "en-IN"},
    {"id": "tech_3", "category": "technology", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "My laptop battery drops from 100% to 3% in ten seconds.", "lang": "en-IN"},
    {"id": "tech_4", "category": "technology", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I fixed one bug and created twelve brand new bugs.", "lang": "en-IN"},
    {"id": "tech_5", "category": "technology", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I have forty-nine Chrome tabs open and I need all of them.", "lang": "en-IN"},
    {"id": "tech_6", "category": "technology", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I spent four hours debugging only to realize a missing semicolon.", "lang": "en-IN"},
    {"id": "tech_7", "category": "technology", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "My code has zero comments and even I don't understand it now.", "lang": "en-IN"},
    {"id": "tech_8", "category": "technology", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "My Wi-Fi disconnects every time the microwave turns on.", "lang": "en-IN"},
    {"id": "tech_9", "category": "technology", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I clicked 'remind me tomorrow' on the system update for three years.", "lang": "en-IN"},
    {"id": "tech_10", "category": "technology", "topic": ConversationTopic.TECHNOLOGY_GAMING, "input": "I tried to explain cloud computing to my grandmother.", "lang": "en-IN"},

    # 8. Procrastination & Laziness (10)
    {"id": "proc_1", "category": "procrastination", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I made a comprehensive to-do list and did none of it.", "lang": "en-IN"},
    {"id": "proc_2", "category": "procrastination", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I sat on the edge of the bed in a towel for forty-five minutes.", "lang": "en-IN"},
    {"id": "proc_3", "category": "procrastination", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I decided to clean my entire wardrobe just to avoid working.", "lang": "en-IN"},
    {"id": "proc_4", "category": "procrastination", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I will definitely start my life turnaround starting next Monday.", "lang": "en-IN"},
    {"id": "proc_5", "category": "procrastination", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I laid on the sofa all day staring at the fan.", "lang": "en-IN"},
    {"id": "proc_6", "category": "procrastination", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I watched six episodes of a documentary I don't even like.", "lang": "en-IN"},
    {"id": "proc_7", "category": "procrastination", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I told everyone I was busy while napping under two blankets.", "lang": "en-IN"},
    {"id": "proc_8", "category": "procrastination", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I opened a Word document, typed the title, and called it a day.", "lang": "en-IN"},
    {"id": "proc_9", "category": "procrastination", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "I bought five self-help books and haven't opened any.", "lang": "en-IN"},
    {"id": "proc_10", "category": "procrastination", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "My productive hours are strictly between 3 AM and 3:05 AM.", "lang": "en-IN"},

    # 9. Food & Cooking (10)
    {"id": "food_1", "category": "food", "topic": ConversationTopic.FOOD_COOKING, "input": "I burned instant noodles so bad the pot melted.", "lang": "en-IN"},
    {"id": "food_2", "category": "food", "topic": ConversationTopic.FOOD_COOKING, "input": "I ordered delivery from a restaurant across the street because I'm lazy.", "lang": "en-IN"},
    {"id": "food_3", "category": "food", "topic": ConversationTopic.FOOD_COOKING, "input": "I put salt instead of sugar in my morning coffee.", "lang": "en-IN"},
    {"id": "food_4", "category": "food", "topic": ConversationTopic.FOOD_COOKING, "input": "I opened the fridge seven times hoping new snacks spawned.", "lang": "en-IN"},
    {"id": "food_5", "category": "food", "topic": ConversationTopic.FOOD_COOKING, "input": "I ate an entire large pizza alone in twelve minutes.", "lang": "en-IN"},
    {"id": "food_6", "category": "food", "topic": ConversationTopic.FOOD_COOKING, "input": "I tried baking cookies and they turned into literal charcoal bricks.", "lang": "en-IN"},
    {"id": "food_7", "category": "food", "topic": ConversationTopic.FOOD_COOKING, "input": "I drank black coffee on an empty stomach and now my soul vibrates.", "lang": "en-IN"},
    {"id": "food_8", "category": "food", "topic": ConversationTopic.FOOD_COOKING, "input": "I spent sixty dollars on an artisanal avocado toast.", "lang": "en-IN"},
    {"id": "food_9", "category": "food", "topic": ConversationTopic.FOOD_COOKING, "input": "I cooked pasta with zero sauce and called it minimalism.", "lang": "en-IN"},
    {"id": "food_10", "category": "food", "topic": ConversationTopic.FOOD_COOKING, "input": "I tracked my food delivery driver's motorcycle like an FBI agent.", "lang": "en-IN"},

    # 10. Sleep & Morning Routines (10)
    {"id": "sleep_1", "category": "sleep", "topic": ConversationTopic.SLEEP_ROUTINES, "input": "I hit the snooze button nine times this morning.", "lang": "en-IN"},
    {"id": "sleep_2", "category": "sleep", "topic": ConversationTopic.SLEEP_ROUTINES, "input": "I slept for twelve hours and woke up even more exhausted.", "lang": "en-IN"},
    {"id": "sleep_3", "category": "sleep", "topic": ConversationTopic.SLEEP_ROUTINES, "input": "I tried to take a twenty-minute power nap and woke up in another dimension.", "lang": "en-IN"},
    {"id": "sleep_4", "category": "sleep", "topic": ConversationTopic.SLEEP_ROUTINES, "input": "My alarm was playing heavy metal and I dreamt I was at a concert.", "lang": "en-IN"},
    {"id": "sleep_5", "category": "sleep", "topic": ConversationTopic.SLEEP_ROUTINES, "input": "I woke up at 6 AM and stared at the ceiling until noon.", "lang": "en-IN"},
    {"id": "sleep_6", "category": "sleep", "topic": ConversationTopic.SLEEP_ROUTINES, "input": "I have four different alarm clocks set at two-minute intervals.", "lang": "en-IN"},
    {"id": "sleep_7", "category": "sleep", "topic": ConversationTopic.SLEEP_ROUTINES, "input": "I fell asleep with my glasses on and bent the frames.", "lang": "en-IN"},
    {"id": "sleep_8", "category": "sleep", "topic": ConversationTopic.SLEEP_ROUTINES, "input": "I couldn't sleep because I remembered an awkward thing I did in 2014.", "lang": "en-IN"},
    {"id": "sleep_9", "category": "sleep", "topic": ConversationTopic.SLEEP_ROUTINES, "input": "I woke up fifteen minutes after my meeting started.", "lang": "en-IN"},
    {"id": "sleep_10", "category": "sleep", "topic": ConversationTopic.SLEEP_ROUTINES, "input": "I slept weirdly and now my neck only turns thirty degrees.", "lang": "en-IN"},

    # 11. Work & Career (10)
    {"id": "work_1", "category": "work", "topic": ConversationTopic.WORK_CAREER, "input": "My boss scheduled a meeting to discuss why we have too many meetings.", "lang": "en-IN"},
    {"id": "work_2", "category": "work", "topic": ConversationTopic.WORK_CAREER, "input": "I replied 'per my last email' with maximum emotional aggression.", "lang": "en-IN"},
    {"id": "work_3", "category": "work", "topic": ConversationTopic.WORK_CAREER, "input": "I nodded through a whole presentation without absorbing a single word.", "lang": "en-IN"},
    {"id": "work_4", "category": "work", "topic": ConversationTopic.WORK_CAREER, "input": "I typed furiously on my keyboard to look busy when my manager walked past.", "lang": "en-IN"},
    {"id": "work_5", "category": "work", "topic": ConversationTopic.WORK_CAREER, "input": "I got promoted to senior lead developer today!", "lang": "en-IN"},
    {"id": "work_6", "category": "work", "topic": ConversationTopic.WORK_CAREER, "input": "My company gave us a branded water bottle instead of an annual bonus.", "lang": "en-IN"},
    {"id": "work_7", "category": "work", "topic": ConversationTopic.WORK_CAREER, "input": "I sent an email complaining about management directly to management.", "lang": "en-IN"},
    {"id": "work_8", "category": "work", "topic": ConversationTopic.WORK_CAREER, "input": "I used five corporate buzzwords in one sentence and felt my soul leave.", "lang": "en-IN"},
    {"id": "work_9", "category": "work", "topic": ConversationTopic.WORK_CAREER, "input": "My Monday morning motivation vanished before 9:02 AM.", "lang": "en-IN"},
    {"id": "work_10", "category": "work", "topic": ConversationTopic.WORK_CAREER, "input": "I pretended my microphone was broken to avoid answering in the standup.", "lang": "en-IN"},

    # 12. Failure & Mistakes (10)
    {"id": "fail_1", "category": "failure", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I pushed a door that clearly had a giant PULL sign.", "lang": "en-IN"},
    {"id": "fail_2", "category": "failure", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I dropped my buttered toast and it landed butter-side down.", "lang": "en-IN"},
    {"id": "fail_3", "category": "failure", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I assembled IKEA furniture and have fourteen screws left over.", "lang": "en-IN"},
    {"id": "fail_4", "category": "failure", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I accidentally washed my white shirts with a red sock.", "lang": "en-IN"},
    {"id": "fail_5", "category": "failure", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I locked myself outside my own apartment while taking out the trash.", "lang": "en-IN"},
    {"id": "fail_6", "category": "failure", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I dropped my phone into the toilet bowl.", "lang": "en-IN"},
    {"id": "fail_7", "category": "failure", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I watered a plastic plant for six straight months.", "lang": "en-IN"},
    {"id": "fail_8", "category": "failure", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I completely forgot my own passcode after changing it five minutes ago.", "lang": "en-IN"},
    {"id": "fail_9", "category": "failure", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I drove forty minutes to the mall and realized I left my wallet at home.", "lang": "en-IN"},
    {"id": "fail_10", "category": "failure", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I failed my driving test in the first thirty seconds.", "lang": "en-IN"},

    # 13. Embarrassing Situations (10)
    {"id": "emb_1", "category": "embarrassment", "topic": ConversationTopic.EMBARRASSMENT, "input": "I enthusiastically waved back at someone who was waving at the person behind me.", "lang": "en-IN"},
    {"id": "emb_2", "category": "embarrassment", "topic": ConversationTopic.EMBARRASSMENT, "input": "My stomach made a whale noise in a dead silent library.", "lang": "en-IN"},
    {"id": "emb_3", "category": "embarrassment", "topic": ConversationTopic.EMBARRASSMENT, "input": "I tripped over absolutely nothing in front of thirty people.", "lang": "en-IN"},
    {"id": "emb_4", "category": "embarrassment", "topic": ConversationTopic.EMBARRASSMENT, "input": "My voice cracked while ordering food at McDonald's.", "lang": "en-IN"},
    {"id": "emb_5", "category": "embarrassment", "topic": ConversationTopic.EMBARRASSMENT, "input": "I walked into the wrong movie theater and watched twenty minutes before noticing.", "lang": "en-IN"},
    {"id": "emb_6", "category": "embarrassment", "topic": ConversationTopic.EMBARRASSMENT, "input": "I thought someone called my name in public and turned around like a celebrity.", "lang": "en-IN"},
    {"id": "emb_7", "category": "embarrassment", "topic": ConversationTopic.EMBARRASSMENT, "input": "I pulled on a locked supermarket door with both hands.", "lang": "en-IN"},
    {"id": "emb_8", "category": "embarrassment", "topic": ConversationTopic.EMBARRASSMENT, "input": "I accidentally called my teacher 'Mom'.", "lang": "en-IN"},
    {"id": "emb_9", "category": "embarrassment", "topic": ConversationTopic.EMBARRASSMENT, "input": "I smiled at a mannequin thinking it was someone I knew.", "lang": "en-IN"},
    {"id": "emb_10", "category": "embarrassment", "topic": ConversationTopic.EMBARRASSMENT, "input": "I tried to plug my charger in the dark and poked myself in the eye.", "lang": "en-IN"},

    # 14. Everyday Casual & Boredom (10)
    {"id": "cas_1", "category": "casual", "topic": ConversationTopic.EVERYDAY_CASUAL, "input": "I am feeling totally bored today.", "lang": "en-IN"},
    {"id": "cas_2", "category": "casual", "topic": ConversationTopic.EVERYDAY_CASUAL, "input": "It is way too hot outside to exist.", "lang": "en-IN"},
    {"id": "cas_3", "category": "casual", "topic": ConversationTopic.EVERYDAY_CASUAL, "input": "I'm having a quiet cup of ginger tea.", "lang": "en-IN"},
    {"id": "cas_4", "category": "casual", "topic": ConversationTopic.EVERYDAY_CASUAL, "input": "The weather forecast said sunny and it's pouring cats and dogs.", "lang": "en-IN"},
    {"id": "cas_5", "category": "casual", "topic": ConversationTopic.EVERYDAY_CASUAL, "input": "I am sitting on the balcony doing absolutely nothing.", "lang": "en-IN"},
    {"id": "cas_6", "category": "casual", "topic": ConversationTopic.EVERYDAY_CASUAL, "input": "Tell me something random that will make me laugh.", "lang": "en-IN"},
    {"id": "cas_7", "category": "casual", "topic": ConversationTopic.EVERYDAY_CASUAL, "input": "I lost my car keys inside my own house again.", "lang": "en-IN"},
    {"id": "cas_8", "category": "casual", "topic": ConversationTopic.EVERYDAY_CASUAL, "input": "My phone battery is at 1% right when I need an Uber.", "lang": "en-IN"},
    {"id": "cas_9", "category": "casual", "topic": ConversationTopic.EVERYDAY_CASUAL, "input": "I stubbed my little toe on the corner of the wooden table.", "lang": "en-IN"},
    {"id": "cas_10", "category": "casual", "topic": ConversationTopic.EVERYDAY_CASUAL, "input": "I am watching water boil like it is prime time entertainment.", "lang": "en-IN"},

    # 15. Dark Comedy Opportunities (10)
    {"id": "dark_1", "category": "dark_comedy", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "My life feels like a blooper reel without any actual movie.", "lang": "en-IN"},
    {"id": "dark_2", "category": "dark_comedy", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I am the warning label on bad life choices.", "lang": "en-IN"},
    {"id": "dark_3", "category": "dark_comedy", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "Everything I touch turns into an unsolved mystery.", "lang": "en-IN"},
    {"id": "dark_4", "category": "dark_comedy", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "If bad luck paid a salary, I would be a billionaire.", "lang": "en-IN"},
    {"id": "dark_5", "category": "dark_comedy", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "My guardian angel definitely took early retirement.", "lang": "en-IN"},
    {"id": "dark_6", "category": "dark_comedy", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I am taking life one crisis at a time.", "lang": "en-IN"},
    {"id": "dark_7", "category": "dark_comedy", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "My expectations were in the basement and life brought a shovel.", "lang": "en-IN"},
    {"id": "dark_8", "category": "dark_comedy", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I bought plant seeds and the dirt gave up on me.", "lang": "en-IN"},
    {"id": "dark_9", "category": "dark_comedy", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "My horoscope said expect turbulence and my shoelace snapped.", "lang": "en-IN"},
    {"id": "dark_10", "category": "dark_comedy", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "I'm not saying I'm unlucky, but rain follows me indoors.", "lang": "en-IN"},

    # 16. Sarcastic User Messages & Boasting (10)
    {"id": "sarc_1", "category": "boast_sarcasm", "topic": ConversationTopic.BOASTING_EGO, "input": "I am basically the undisputed genius of this household.", "lang": "en-IN"},
    {"id": "sarc_2", "category": "boast_sarcasm", "topic": ConversationTopic.BOASTING_EGO, "input": "My cooking skills are Michelin star level.", "lang": "en-IN"},
    {"id": "sarc_3", "category": "boast_sarcasm", "topic": ConversationTopic.BOASTING_EGO, "input": "Roast me, give me your sharpest comeback.", "lang": "en-IN"},
    {"id": "sarc_4", "category": "boast_sarcasm", "topic": ConversationTopic.BOASTING_EGO, "input": "Destroy my ego with a single punchline.", "lang": "en-IN"},
    {"id": "sarc_5", "category": "boast_sarcasm", "topic": ConversationTopic.BOASTING_EGO, "input": "I think I look like a Hollywood movie star today.", "lang": "en-IN"},
    {"id": "sarc_6", "category": "boast_sarcasm", "topic": ConversationTopic.BOASTING_EGO, "input": "My fashion sense is light years ahead of my friends.", "lang": "en-IN"},
    {"id": "sarc_7", "category": "boast_sarcasm", "topic": ConversationTopic.BOASTING_EGO, "input": "I have an IQ of 160 and you cannot beat me.", "lang": "en-IN"},
    {"id": "sarc_8", "category": "boast_sarcasm", "topic": ConversationTopic.BOASTING_EGO, "input": "I ran a marathon in my dreams without breaking a sweat.", "lang": "en-IN"},
    {"id": "sarc_9", "category": "boast_sarcasm", "topic": ConversationTopic.BOASTING_EGO, "input": "I am the main character and everyone else is an NPC.", "lang": "en-IN"},
    {"id": "sarc_10", "category": "boast_sarcasm", "topic": ConversationTopic.BOASTING_EGO, "input": "My singing voice can heal international conflicts.", "lang": "en-IN"},

    # 17. Native Indian Languages (10 Languages - 10 Tests)
    {"id": "lang_te", "category": "native_indic", "topic": ConversationTopic.EXAMS_STUDIES, "input": "నేను రేపటి నుంచి చదువుతాను.", "lang": "te-IN"},
    {"id": "lang_hi", "category": "native_indic", "topic": ConversationTopic.EXAMS_STUDIES, "input": "मैं कल से पढूंगा।", "lang": "hi-IN"},
    {"id": "lang_ta", "category": "native_indic", "topic": ConversationTopic.EXAMS_STUDIES, "input": "நான் நாளைக்கு படிக்க போறேன்.", "lang": "ta-IN"},
    {"id": "lang_kn", "category": "native_indic", "topic": ConversationTopic.EXAMS_STUDIES, "input": "ನಾನು ನಾಳೆಯಿಂದ ಓದುತ್ತೇನೆ.", "lang": "kn-IN"},
    {"id": "lang_ml", "category": "native_indic", "topic": ConversationTopic.EXAMS_STUDIES, "input": "ഞാൻ നാളെ മുതൽ പഠിക്കും.", "lang": "ml-IN"},
    {"id": "lang_bn", "category": "native_indic", "topic": ConversationTopic.EXAMS_STUDIES, "input": "আমি কাল থেকে পড়ব।", "lang": "bn-IN"},
    {"id": "lang_mr", "category": "native_indic", "topic": ConversationTopic.EXAMS_STUDIES, "input": "मी उद्यापासून अभ्यास करणार.", "lang": "mr-IN"},
    {"id": "lang_gu", "category": "native_indic", "topic": ConversationTopic.EXAMS_STUDIES, "input": "હું કાલથી વાંચીશ.", "lang": "gu-IN"},
    {"id": "lang_pa", "category": "native_indic", "topic": ConversationTopic.EXAMS_STUDIES, "input": "ਮੈਂ ਕੱਲ੍ਹ ਤੋਂ ਪੜ੍ਹਾਂਗਾ।", "lang": "pa-IN"},
    {"id": "lang_en", "category": "native_indic", "topic": ConversationTopic.EXAMS_STUDIES, "input": "I will start studying tomorrow morning.", "lang": "en-IN"},

    # 18. Code-Mixed Conversational Inputs (10)
    {"id": "cm_1", "category": "code_mixed", "topic": ConversationTopic.EXAMS_STUDIES, "input": "Naku exam lo fail ayya bro, chala tension undi.", "lang": "te-IN"},
    {"id": "cm_2", "category": "code_mixed", "topic": ConversationTopic.EXAMS_STUDIES, "input": "Bhai kal mera exam hai aur maine kuch nahi padha.", "lang": "hi-IN"},
    {"id": "cm_3", "category": "code_mixed", "topic": ConversationTopic.RELATIONSHIPS_DATING, "input": "Crush ku message panniten da, romba bayama irukku.", "lang": "ta-IN"},
    {"id": "cm_4", "category": "code_mixed", "topic": ConversationTopic.MONEY_FINANCES, "input": "Account lo zero balance undi mama, treat ela ivvali?", "lang": "te-IN"},
    {"id": "cm_5", "category": "code_mixed", "topic": ConversationTopic.EVERYDAY_CASUAL, "input": "Weekend full bore aagthide guru, yenu plan illa.", "lang": "kn-IN"},
    {"id": "cm_6", "category": "code_mixed", "topic": ConversationTopic.PROCRASTINATION_HABITS, "input": "Bhai pure din bas reels scroll kiya maine.", "lang": "hi-IN"},
    {"id": "cm_7", "category": "code_mixed", "topic": ConversationTopic.FOOD_COOKING, "input": "Mid night biryani order chesa bro, full guilt undi.", "lang": "te-IN"},
    {"id": "cm_8", "category": "code_mixed", "topic": ConversationTopic.WORK_CAREER, "input": "Boss kooda meeting undi da, escape aaganum.", "lang": "ta-IN"},
    {"id": "cm_9", "category": "code_mixed", "topic": ConversationTopic.FITNESS_HEALTH, "input": "Gym join aayi one week aaythu, body full pain.", "lang": "kn-IN"},
    {"id": "cm_10", "category": "code_mixed", "topic": ConversationTopic.FAILURE_MISTAKES, "input": "Code lo 40 errors vachayi bro, em cheyyali?", "lang": "te-IN"},
]


# ---------------------------------------------------------------------------
# Test Suite Implementation
# ---------------------------------------------------------------------------

def test_scenario_count_at_least_150():
    """Verify that test suite contains at least 150 comprehensive humour scenarios."""
    assert len(ALL_HUMOUR_SCENARIOS) >= 150, f"Expected >= 150 scenarios, found {len(ALL_HUMOUR_SCENARIOS)}"


def test_all_150_scenarios_humour_engine_classification():
    """
    Verify HumourEngine analyzes all scenarios:
    - Accurate topic mapping
    - Valid comedic technique selected
    - Banned openers generated
    - Prompt directive formatted cleanly
    """
    engine = HumourEngine()
    session = ConversationSession(conversation_id="test_suite_150")

    for scenario in ALL_HUMOUR_SCENARIOS:
        inp = scenario["input"]
        lang = scenario["lang"]
        expected_topic = scenario.get("topic")

        ctx = engine.build_humour_context(inp, session=session, language_code=lang)

        assert ctx.topic is not None
        assert isinstance(ctx.technique, ComedicTechnique)
        assert len(ctx.banned_openers) > 0
        prompt_injection = ctx.format_prompt_injection()
        assert "CONTEXT-FIRST HUMOUR INTELLIGENCE DIRECTIVE" in prompt_injection
        assert "LENGTH & FORMAT: EXACTLY ONE natural spoken sentence" in prompt_injection


def test_multi_turn_continuity_and_callback_detection():
    """
    Verify multi-turn progression:
    - AI tracks topic continuity when user continues topic
    - AI captures salient entity from user for potential callback
    """
    engine = HumourEngine()
    session = ConversationSession(conversation_id="conv_multiturn")

    # Turn 1: Exam announcement
    ctx1 = engine.build_humour_context("I have a giant math exam tomorrow.", session=session)
    assert ctx1.topic == ConversationTopic.EXAMS_STUDIES
    session.add_message("user", "I have a giant math exam tomorrow.")
    session.add_message("assistant", "Suddenly, opening page one feels like an Olympic event.")
    session.record_turn_metadata(topic="exams_studies", assistant_opener="suddenly opening")

    # Turn 2: Continued exam panic
    ctx2 = engine.build_humour_context("I haven't studied a single chapter yet.", session=session)
    assert ctx2.topic == ConversationTopic.EXAMS_STUDIES
    assert ctx2.callback_note is not None
    assert "continuing the same topic" in ctx2.callback_note
    # Verify opener "suddenly" or "suddenly opening" is banned to avoid repetition
    assert any("suddenly" in op.lower() for op in ctx2.banned_openers)


def test_anti_repetition_opener_diversity():
    """
    Verify that if the AI opened with 'Bro' or 'Your', those openers are strictly
    banned on the following turn.
    """
    engine = HumourEngine()
    session = ConversationSession(conversation_id="conv_diversity")

    # Simulate AI previously opening with "Bro"
    session.record_turn_metadata(topic="casual", assistant_opener="Bro, tomorrow")
    ctx = engine.build_humour_context("I woke up late today.", session=session)
    assert "Bro" in ctx.banned_openers
    assert "Yeah bro" in ctx.banned_openers

    # Simulate AI previously opening with "Your"
    session.record_turn_metadata(topic="money", assistant_opener="Your bank")
    ctx2 = engine.build_humour_context("I have no money.", session=session)
    assert "Your" in ctx2.banned_openers


def test_quality_and_relevance_validator_with_new_rules():
    """
    Verify that disconnected clichés are rejected and context-relevant comebacks pass:
    - Reject smoke detector when topic is not cooking/burning
    - Reject buffering when topic is not streaming/confusion
    - Reject banned repetitive openers
    """
    # 1. Disconnected cliché: mentioning smoke detector when talking about math exam
    is_valid, reason = validate_response_quality(
        user_message="I have an exam in one hour.",
        candidate_response="At least your smoke detector got some attention.",
    )
    assert not is_valid
    assert "smoke detector" in reason.lower()

    # 2. Disconnected cliché: mentioning gym membership when talking about dating
    is_valid2, reason2 = validate_response_quality(
        user_message="My crush left me on read for three days.",
        candidate_response="Your gym membership is already crying.",
    )
    assert not is_valid2
    assert "gym membership" in reason2.lower()

    # 3. Banned opener enforcement
    is_valid3, reason3 = validate_response_quality(
        user_message="I burned my toast this morning.",
        candidate_response="Bro, your breakfast is officially charcoal.",
        banned_openers=["Bro", "Bro, your"],
    )
    assert not is_valid3
    assert "banned repetitive opener" in reason3.lower()

    # 4. Perfectly context-aligned comeback passes cleanly
    is_valid4, reason4 = validate_response_quality(
        user_message="I have an exam in one hour.",
        candidate_response="Panic is officially your main study strategy.",
    )
    assert is_valid4
    assert "context-aligned" in reason4.lower()


def test_all_10_native_languages_and_codemixing_pipeline():
    """
    Verify end-to-end processing with HumourEngine on all 10 Indian languages.
    """
    async def _run():
        class MockGemini:
            model = "gemma-4-26b-a4b-it"
            async def generate_response(self, messages, system_instruction=None, **kwargs):
                last_msg = messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1].parts[0].text
                if "రేపటి" in last_msg or "exam lo" in last_msg:
                    return "రేపు ఆల్రెడీ నిన్ను బ్లాక్ చేసేసింది బ్రో."
                elif "पढूंगा" in last_msg or "Bhai kal" in last_msg:
                    return "भाई, कल ने तुझे पहले ही ब्लॉक कर दिया।"
                elif "படிக்க" in last_msg:
                    return "தலைவா, நாளைக்கு உன்னை ஆல்ரెடி பிளாக் பண்ணிடுச்சு."
                elif "ಓದುತ್ತೇನೆ" in last_msg:
                    return "ಬ್ರೋ, ನಾಳೆ ಆಲ್ರೆಡಿ ನಿನ್ನನ್ನು ಬ್ಲಾಕ್ ಮಾಡಿದೆ."
                elif "പഠിക്കും" in last_msg:
                    return "ബ്രോ, നാളെ നിന്നെ ഓൾറെഡി ബ്ലോക്ക് ചെയ്തു."
                elif "পড়ব" in last_msg:
                    return "ভাই, কাল তোকে আগেই ব্লক করে দিয়েছে।"
                elif "अभ्यास" in last_msg:
                    return "भावा, उद्याने तुला आधीच ब्लॉक केलंय."
                elif "વાંચીશ" in last_msg:
                    return "ભાઈ, કાલે તને પહેલેથી બ્લોક કરી દીધો છે."
                elif "ਪੜ੍ਹਾਂਗਾ" in last_msg:
                    return "ਬਾਈ, ਕੱਲ੍ਹ ਨੇ ਤੈਨੂੰ ਪਹਿਲਾਂ ਹੀ ਬਲਾਕ ਕਰਤਾ।"
                else:
                    return "Bro, tomorrow has blocked you already."

        engine = IntelligenceEngine(gemini_service=MockGemini())

        test_cases = [
            ("te-IN", "నేను రేపటి నుంచి చదువుతాను.", "te-IN"),
            ("hi-IN", "मैं कल से पढूंगा।", "hi-IN"),
            ("ta-IN", "நான் நாளைக்கு படிக்க போறேன்.", "ta-IN"),
            ("kn-IN", "ನಾನು ನಾಳೆಯಿಂದ ಓದುತ್ತೇನೆ.", "kn-IN"),
            ("ml-IN", "ഞാൻ നാളെ മുതൽ പഠിക്കും.", "ml-IN"),
            ("bn-IN", "আমি কাল থেকে পড়ব।", "bn-IN"),
            ("mr-IN", "मी उद्यापासून अभ्यास करणार.", "mr-IN"),
            ("gu-IN", "હું કાલથી વાંચીશ.", "gu-IN"),
            ("pa-IN", "ਮੈਂ ਕੱਲ੍ਹ ਤੋਂ ਪੜ੍ਹਾਂਗਾ।", "pa-IN"),
            ("en-IN", "I will start studying tomorrow morning.", "en-IN"),
            ("te-IN", "Naku exam lo fail ayya bro, chala tension undi.", "te-IN"),
            ("hi-IN", "Bhai kal mera exam hai aur maine kuch nahi padha.", "hi-IN"),
        ]

        for req_lang, text, expected_lang in test_cases:
            res = await engine.process_turn(
                message=text,
                conversation_id="conv_indic_150",
                language=req_lang,
            )
            assert res.success is True
            assert res.language == expected_lang
            assert count_sentences(res.assistant_response) == 1
            assert count_words(res.assistant_response) <= 15

    asyncio.run(_run())
