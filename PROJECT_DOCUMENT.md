# TheONE AI — Project Overview & Architecture Document

**Application Name:** TheONE AI  
**Live Production Website:** [https://theone-ai.onrender.com](https://theone-ai.onrender.com)  
**Repository:** [https://github.com/Tarak098/TheONE-AI.git](https://github.com/Tarak098/TheONE-AI.git)  
**Deployment Platform:** Render (Web Service)  
**Primary Interface:** Minimal, voice-first AI companion inspired by ChatGPT Advanced Voice Mode  

---

## 1. Executive Summary
**TheONE AI** is a real-time, voice-first artificial intelligence companion designed for ultra-low latency, witty, and contextually grounded verbal interaction. Unlike traditional chatbot interfaces with dense text bubbles, TheONE AI centers interaction around an audio-reactive **Voice Orb**, which visualizes conversational states dynamically. The AI speaks with user-selected neural Indian voices and enforces a strict **one-sentence response limit** ($\le 15$ words) to deliver maximum comedic timing and sharp sarcasm without paragraphs or filler.

---

## 2. System Architecture & Voice Pipeline

```
[ User Microphone ]
        │  (Linear PCM 16-bit, 16 kHz Mono WAV)
        ▼
[ Gnani Prisma STT v3 ] ──► Recognized Text Transcript
        │
        ▼
[ Intelligence Orchestrator ]
  ├── Unified Comedy Personality (Always funny, sarcastic & witty across all moods)
  ├── Adaptive Comedy Engine (Witty, sarcastic, roast, situational, gentle teasing)
  ├── Google Gemma 4 26B (with Gemini 2.5 Flash High-Availability Failover)
  └── Response Length Validator (Strictly ≤ 1 Sentence, 3–10 Words, Max 15 Words)
        │
        ▼
[ Gnani Timbre TTS v2.5 ]
  ├── User Voice Selection: "Yashvi" (Female, Cheerful 1.04x speed)
  └── User Voice Selection: "Zayan" (Male, Deadpan 0.98x speed)
        │  (48 kHz WAV Audio)
        ▼
[ Web Audio API Player ] ──► Real-Time AnalyserNode ──► Reactive Voice Orb Animation
```

---

## 3. Core Capabilities & Feature Set

1. **Central Voice Orb (5 State Machine)**:
   - `IDLE`: Calming ambient breathing glow.
   - `LISTENING`: Real-time microphone audio volume reactivity via Web Audio API.
   - `THINKING`: Dynamic orbital swirling glow during LLM generation.
   - `SPEAKING`: Acoustic ripple waves pulsating to synthesized speech amplitude.
   - `ERROR`: Subtle warm amber feedback.
2. **Natural, Savage & Human-Like Comedy**:
   - TheONE AI operates like a naturally funny, savage friend who always has an instant comeback.
   - **Simple Everyday Spoken English**: Strictly uses common, daily words that real people use in casual conversation. Zero complicated vocabulary, zero corporate phrases, and zero textbook jokes.
   - **Playfully Savage**: Bold, cheeky, relatable comebacks, funny observations, and spontaneous roasting across all topics and moods.
   - **Zero Assistant Mode**: Never turns into a dry, boring, or motivational assistant. Sensitive topics handled with gentle lightheartedness without breaking character.
3. **Strict Response Length**:
   - Maximum 1 sentence (3–10 words preferred, maximum 15 words).
   - Zero paragraphs, zero lectures, zero stock filler.
4. **Independent Voice Selection**:
   - User toggles between `Yashvi` (Female) and `Zayan` (Male) directly via the top navigation bar.
   - Voice selection is completely independent of conversational humor style and persisted in `localStorage`.
5. **Secondary Controls & Visual Feedback**:
   - **Real-Time Speech Transcription**: Real-time chunk-based STT transcription with Gnani Prisma. Words update dynamically in a temporary live transcription area while the user is speaking.
   - **Voice/Audio-Only AI Responses**: The AI responds purely via voice audio through Gnani Timbre TTS without displaying AI response textboxes, message cards, or chat bubbles on the active screen.
   - **Playback-Linked Transcription Lifecycle**: The live transcription remains visible during AI thinking and speech playback, and smoothly fades out after playback completes.
   - **Full Conversation History Drawer**: Both user inputs and AI replies are preserved in the slide-over transcript drawer with voice replay buttons and timestamps.
   - Light and dark theme toggle.

---

## 4. Technical Stack
- **Backend**: FastAPI (Python 3.11), Uvicorn ASGI, Pydantic Settings.
- **Frontend**: Vanilla JavaScript (ES6+), HTML5 Canvas, Web Audio API, Modern CSS3.
- **Speech Recognition (STT)**: Gnani Prisma v3 REST API.
- **LLM Reasoning**: Google Gemma 4 26B (`gemma-4-26b-a4b-it`) via Google GenAI SDK.
- **Speech Synthesis (TTS)**: Gnani Timbre v2.5 REST API.

---

## 5. Production & Deployment Specification
- **Single Full-Stack ASGI Service**: FastAPI serves both `/api/v1` routes and static assets (`/`, `/style.css`, `/script.js`) on the same origin, eliminating CORS issues and supporting browser microphone access over HTTPS.
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Build Command**: `pip install -r requirements.txt`
- **Health Check**: `GET /health` returns `{"status": "healthy"}`.

