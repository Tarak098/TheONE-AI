# TheONE AI ✨

> **A minimal, voice-first AI companion inspired by ChatGPT's Advanced Voice Mode.**  
> Powered by **FastAPI**, **Gnani Prisma STT**, **Google Gemma 4 26B**, and **Gnani Timbre TTS**.

[![Live Production](https://img.shields.io/badge/Live_App-theone--ai.onrender.com-brightgreen?style=for-the-badge&logo=render)](https://theone-ai.onrender.com)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Tarak098%2FTheONE--AI-blue?style=for-the-badge&logo=github)](https://github.com/Tarak098/TheONE-AI)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

---

### 🌐 Live Production Application
🔗 **[https://theone-ai.onrender.com](https://theone-ai.onrender.com)**

Experience real-time expressive voice interaction with dual personality modes (`Always Funny` & `Roast Companion`), real-time voice orb visualizer, and selectable Indian neural voices (Yashvi & Zayan).

---

## 🌟 Overview

**TheONE AI** is designed around a central, animated **Voice Orb** that visualizes voice interactions in real time. It delivers razor-sharp, hilarious, or sarcastic comebacks while adhering to a strict **one-sentence response limit** ($\le 15$ words).

### ✨ Key Features
- **Central Voice Orb**: 5-state animated audio visualizer (`IDLE`, `LISTENING`, `THINKING`, `SPEAKING`, `ERROR`) reacting to real-time audio volume via Web Audio API.
- **Dual Personality Modes**:
  - 😂 **Always Funny**: Cheerful, sarcastic best friend.
  - 🔥 **Roast Companion**: Witty, savage roast master.
- **Strict Response Constraint**: Maximum 1 sentence (3–10 words preferred, max 15 words). No paragraphs, no explanations.
- **User-Selectable Voices**: Choose between **Yashvi** (Female) and **Zayan** (Male) independently per mode, persisted in local storage.
- **Context-Aware Humour**: Responses are strictly anchored to what you said (no disconnected stock jokes). Sensitive topics are handled with empathy.
- **Unified Full-Stack Deployment**: FastAPI serves the REST API and the frontend UI on a single origin.

---

## 📁 Repository Structure

```
TheONE-AI/
├── backend/
│   ├── main.py              # FastAPI application, static file mounts & API routes
│   ├── config.py            # Centralized Pydantic settings & environment resolution
│   ├── services/
│   │   ├── prisma.py        # Gnani Prisma (STT v3) service client
│   │   ├── gemini.py        # Google Gemma 4 26B / Gemini Flash client with auto-failover
│   │   ├── timbre.py        # Gnani Timbre TTS client with speed prosody & tag safety
│   │   ├── intelligence.py  # Conversational orchestrator & quality validator
│   │   ├── personality.py   # Mode prompt manager & cache
│   │   └── conversation.py  # In-memory session manager
│   ├── prompts/
│   │   ├── funny_mode.txt   # Always Funny persona prompt
│   │   └── roast_mode.txt   # Roast Companion persona prompt
│   └── utils/
│       └── audio.py         # 16 kHz Mono WAV audio validation
├── frontend/
│   ├── index.html           # Minimal voice-first UI stage & slide-over drawers
│   ├── style.css            # Dark/light theme & Voice Orb fluid animations
│   └── script.js            # 5-state visualizer engine, Web Audio API & pipeline dispatch
├── tests/                   # Comprehensive test suites (30 scenarios, Timbre, STT, validator)
├── main.py                  # Production ASGI entry point for Render / local deployment
├── render.yaml              # Render Blueprint deployment specification
├── PROJECT_DOCUMENT.md      # 1-page architecture & design document
├── requirements.txt         # Pinned production Python dependencies
├── .env.example             # Clean environment template (no credentials)
└── .gitignore               # Strict exclude list preventing secret leakage
```

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- Python 3.11+
- Virtual environment

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Tarak098/TheONE-AI.git
cd TheONE-AI

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup
Copy `.env.example` to `.env` and fill in your API credentials:
```bash
cp .env.example .env
```
Key required credentials:
- `GEMINI_API_KEY`: Google AI Studio API key.
- `GNANI_API_KEY`: Gnani.ai API key (powers both Prisma STT and Timbre TTS).

### 4. Run the Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000/`** in your browser.

---

## ☁️ Deployment on Render

This project is 100% pre-configured for **Render** via `render.yaml`.

1. Go to the [Render Dashboard](https://dashboard.render.com/).
2. Click **New** → **Blueprint**.
3. Connect your GitHub repository `Tarak098/TheONE-AI`.
4. Render will automatically detect `render.yaml` and configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
5. Enter your secret environment variables in the Render Dashboard:
   - `GEMINI_API_KEY`
   - `GNANI_API_KEY`
6. Click **Apply**. Your app will be live with HTTPS (enabling microphone access in all mobile and desktop browsers)!

---

## 🧪 Testing

Run the automated test suite:
```bash
# Run unit tests and context validation
pytest tests/test_response_validator.py tests/test_intelligence.py

# Run Timbre TTS tests
pytest tests/test_timbre.py

# Run 30-message live conversational scenario test suite
pytest tests/test_context_and_expressive_voice.py -v
```

---

## 📄 License
MIT License. Created by Tarak098.
