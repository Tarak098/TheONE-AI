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
  ├── Mode Prompt Resolution (Always Funny / Roast Companion)
  ├── Context-Aware Subject Alignment & Anti-Cliché Validation
  ├── Google Gemma 4 26B (with Gemini 2.5 Flash High-Availability Failover)
  └── Response Length Validator (Strictly ≤ 1 Sentence, 3–10 Words)
        │
        ▼
[ Gnani Timbre TTS v2.5 ]
  ├── Always Funny: "Yashvi" (Female, Cheerful 1.04x speed)
  └── Roast Companion: "Zayan" (Male, Deadpan 0.98x speed with dramatic pauses "...")
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
2. **Dual Personalities**:
   - **Always Funny**: Cheerful, sarcastic best friend turning everyday moments into instant comedy.
   - **Roast Companion**: Witty, confident roast master delivering razor-sharp, unexpected comebacks.
3. **Context-Aware Humour & Safety**:
   - Responds strictly to the subject mentioned (no disconnected generic jokes).
   - Sensitive message detection ensures genuine grief or emergencies receive warm, supportive care.
4. **Independent Voice Persistence**:
   - User choices between `Yashvi` (Female) and `Zayan` (Male) are maintained independently per personality mode and persisted across sessions.
5. **Secondary Controls**:
   - Floating live transcript card with replay button.
   - Slide-over Conversation History Drawer and Settings Drawer.
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

