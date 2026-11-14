"""
LaughBot AI - FastAPI Backend Application.

Main entry point providing core routing, CORS configuration,
health checks, Gnani Prisma STT, and the AI Intelligence Engine (Evon v3.3).
"""

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.services.conversation import ConversationManager
from backend.services.evon import (
    EvonAPIError,
    EvonConnectionError,
    EvonService,
    EvonServiceUnavailableError,
    EvonTimeoutError,
)
from backend.services.gemini import (
    GeminiAPIError,
    GeminiInvalidResponseError,
    GeminiRateLimitError,
    GeminiService,
    GeminiServiceUnavailableError,
    GeminiTimeoutError,
)
from backend.services.intelligence import (
    IntelligenceEngine,
    IntelligenceValidationError,
)
from backend.services.personality import PersonalityManager
from backend.services.prisma import (
    PrismaAPIError,
    PrismaConfigError,
    PrismaConnectionError,
    PrismaService,
    PrismaTimeoutError,
)
from backend.services.timbre import (
    TimbreAPIError,
    TimbreConfigError,
    TimbreInvalidAudioError,
    TimbreRateLimitError,
    TimbreService,
    TimbreServiceUnavailableError,
    TimbreTimeoutError,
    TimbreValidationError,
)
from backend.utils.audio import AudioValidationError, validate_wav_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("laughbot.main")

settings = get_settings()
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Supported Indian languages for Gnani Prisma STT
SUPPORTED_LANGUAGES: Dict[str, str] = {
    "en-IN": "English (India)",
    "te-IN": "Telugu",
    "hi-IN": "Hindi",
    "ta-IN": "Tamil",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown logging."""
    logger.info("Initializing %s in %s mode...", settings.APP_NAME, settings.APP_ENV)
    yield
    logger.info("Shutting down %s...", settings.APP_NAME)


# Initialize FastAPI application with title from configuration
app = FastAPI(
    title=settings.APP_NAME,
    description="Backend API for TheONE AI — Voice-First AI Companion with Gnani Prisma STT, Google Gemma, and Gnani Timbre TTS",
    version="0.3.0",
    lifespan=lifespan,
)

# Configure CORS using configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
prisma_service = PrismaService()
gemini_service = GeminiService()
evon_service = EvonService()
timbre_service = TimbreService()
personality_manager = PersonalityManager()
conversation_manager = ConversationManager()
intelligence_engine = IntelligenceEngine(
    gemini_service=gemini_service,
    evon_service=evon_service,
    personality_manager=personality_manager,
    conversation_manager=conversation_manager,
)


# ------------------------------------------------------------------------------
# Pydantic Request Models
# ------------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Request payload for intelligence conversation turns."""
    conversation_id: Optional[str] = Field(default=None, description="Existing conversation ID or null for new")
    message: str = Field(..., min_length=1, max_length=2000, description="User message text")
    mode: str = Field(default="unified_comedy", description="Personality mode: 'unified_comedy'")
    language: str = Field(default="en", description="Language code")


class ClearChatRequest(BaseModel):
    """Request payload for clearing conversation history."""
    conversation_id: str = Field(..., min_length=1, description="ID of conversation to clear")


class SwitchModeRequest(BaseModel):
    """Request payload for switching personality mode."""
    conversation_id: str = Field(..., min_length=1, description="Target conversation ID")
    mode: str = Field(default="unified_comedy", description="Personality mode")


class TTSRequest(BaseModel):
    """Request payload for Text-to-Speech synthesis via Gnani Timbre."""
    text: str = Field(..., min_length=1, max_length=4000, description="Text to synthesize into speech")
    mode: Optional[str] = Field(default="unified_comedy", description="Personality mode ('unified_comedy')")
    voice: Optional[str] = Field(default=None, description="Explicit voice override (e.g., Yashvi, Zayan)")
    language: Optional[str] = Field(default=None, description="Language code (defaults to en-IN)")
    speed: Optional[float] = Field(default=None, ge=0.5, le=2.0, description="Speech speed multiplier")


# ------------------------------------------------------------------------------
# Error Handlers
# ------------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle standard HTTP exceptions with structured response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Request validation failed", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all error handler for unexpected server exceptions."""
    logger.exception("Unhandled server exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An internal server error occurred", "status_code": 500},
    )


# ------------------------------------------------------------------------------
# Core Endpoints
# ------------------------------------------------------------------------------
@app.get("/", summary="Root UI Endpoint")
async def root(request: Request):
    """Serve TheONE AI Frontend UI or JSON status based on request headers."""
    index_file = FRONTEND_DIR / "index.html"
    accept = request.headers.get("accept", "")
    if ("text/html" in accept or "*/*" in accept) and index_file.exists():
        return FileResponse(index_file)
    if "application/json" in accept and "text/html" not in accept:
        return {
            "project": settings.APP_NAME,
            "status": "running",
        }
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "project": settings.APP_NAME,
        "status": "running",
    }


@app.get("/health", summary="Health Check")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
    }


@app.get("/api/v1/status", summary="Service Credentials & Readiness Status")
async def api_status() -> Dict[str, Any]:
    """
    Return overall application status and whether each service
    has credentials configured. Returns boolean flags only, never secrets.
    """
    return {
        "project": settings.APP_NAME,
        "status": "running",
        "environment": settings.APP_ENV,
        "active_provider": settings.active_llm_provider,
        "services": {
            "prisma": prisma_service.is_configured,
            "gemini": gemini_service.is_configured,
            "evon": evon_service.is_configured,
            "timbre": timbre_service.is_configured,
        },
    }


# ------------------------------------------------------------------------------
# Speech-to-Text Endpoint (Gnani Prisma v3)
# ------------------------------------------------------------------------------
@app.post(
    "/api/v1/stt/transcribe",
    summary="Transcribe Audio via Gnani Prisma v3",
    tags=["Speech-to-Text"],
)
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="WAV format audio recording"),
    language_code: str = Form("en-IN", description="Selected BCP-47 language tag"),
) -> Dict[str, Any]:
    """
    Upload a WAV audio recording and return the speech-to-text transcript
    generated by Gnani Prisma v3.
    """
    normalized_lang = language_code.strip()
    if normalized_lang not in SUPPORTED_LANGUAGES:
        valid_options = ", ".join(SUPPORTED_LANGUAGES.keys())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language_code '{language_code}'. Supported languages: {valid_options}",
        )

    if not prisma_service.is_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gnani Prisma STT credentials are not configured. Please supply API credentials in .env.",
        )

    try:
        audio_bytes = await audio_file.read()
    except Exception as err:
        logger.error("Failed to read uploaded audio stream: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded audio file stream.",
        ) from err

    try:
        validate_wav_audio(audio_bytes)
    except AudioValidationError as err:
        logger.warning("Audio validation failed: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        ) from err

    filename = audio_file.filename or "recording.wav"
    try:
        result = await asyncio.to_thread(
            prisma_service.transcribe_audio,
            audio_bytes=audio_bytes,
            language_code=normalized_lang,
            filename=filename,
        )
    except PrismaConfigError as err:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(err)) from err
    except PrismaTimeoutError as err:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(err)) from err
    except PrismaConnectionError as err:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(err)) from err
    except PrismaAPIError as err:
        http_code = err.status_code if err.status_code in (400, 401, 403, 422) else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=http_code, detail=str(err)) from err
    except Exception as err:
        logger.exception("Unexpected error during speech transcription: %s", err)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error during speech transcription.") from err

    return {
        "success": True,
        "transcript": result.transcript,
        "language": result.language_code,
        "request_id": result.request_id,
    }


# ------------------------------------------------------------------------------
# Text-to-Speech Endpoints (Gnani Timbre v2.5)
# ------------------------------------------------------------------------------
@app.post(
    "/api/v1/tts",
    summary="Synthesize Speech via Gnani Timbre TTS",
    tags=["Text-to-Speech"],
)
async def synthesize_speech(request: TTSRequest) -> Dict[str, Any]:
    """
    Synthesize text into speech audio using Gnani Timbre (timbre-v2.5).
    Adapts voice to the chosen personality (Always Funny -> Yashvi, Roast Companion -> Zayan).
    Returns playable audio base64, data URL, voice, and audio format metadata.
    """
    if not timbre_service.is_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gnani Timbre TTS credentials are not configured. Please supply API credentials in .env.",
        )

    try:
        result = await timbre_service.async_synthesize_speech(
            text=request.text,
            mode=request.mode,
            voice=request.voice,
            language=request.language,
            speed=request.speed,
        )
        return {
            "success": True,
            "status": "completed",
            "voice": result.voice,
            "model": result.model,
            "language": result.language,
            "sample_rate": result.sample_rate,
            "format": result.format,
            "audio_base64": result.audio_base64,
            "audio_url": result.data_url,
        }
    except TimbreValidationError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except TimbreRateLimitError as err:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(err)) from err
    except TimbreTimeoutError as err:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(err)) from err
    except TimbreInvalidAudioError as err:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(err)) from err
    except TimbreAPIError as err:
        http_code = err.status_code if err.status_code in (400, 401, 403, 404, 422, 429) else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=http_code, detail=str(err)) from err
    except Exception as err:
        logger.exception("Unexpected error during speech synthesis: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during speech synthesis.",
        ) from err


@app.get(
    "/api/v1/tts/health",
    summary="Gnani Timbre TTS Service Health & Status",
    tags=["Text-to-Speech"],
)
async def tts_health() -> Dict[str, Any]:
    """Return health and configuration status of Gnani Timbre TTS."""
    return timbre_service.health_check()


@app.get(
    "/api/v1/tts/voices",
    summary="List Supported Gnani Timbre Voices",
    tags=["Text-to-Speech"],
)
async def list_voices() -> Dict[str, Any]:
    """Return catalog of supported voices with gender and delivery metadata."""
    return {
        "success": True,
        "voices": TimbreService.get_supported_voices(),
        "default_voice": timbre_service.default_voice,
        "voice_funny": timbre_service.voice_funny,
        "voice_roast": timbre_service.voice_roast,
    }


# ------------------------------------------------------------------------------
# Intelligence & Personality Engine Endpoints (Step 3: Google Gemini & Gnani Evon)
# ------------------------------------------------------------------------------
@app.post(
    "/api/v1/chat",
    summary="Process Conversational Chat Turn via Intelligence Engine",
    tags=["Intelligence Engine"],
)
async def chat(request: ChatRequest) -> Dict[str, Any]:
    """
    Process a chat message using the LaughBot Intelligence Engine.
    Coordinates personality system prompts, context memory, and the active LLM provider.
    """
    try:
        result = await intelligence_engine.process_turn(
            message=request.message,
            conversation_id=request.conversation_id,
            mode=request.mode,
            language=request.language,
        )
        return result.to_dict()
    except IntelligenceValidationError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except ValueError as err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    except GeminiServiceUnavailableError as err:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(err)) from err
    except GeminiRateLimitError as err:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(err)) from err
    except GeminiTimeoutError as err:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(err)) from err
    except GeminiInvalidResponseError as err:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(err)) from err
    except GeminiAPIError as err:
        http_code = err.status_code if err.status_code in (400, 401, 403, 404, 422, 429) else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=http_code, detail=str(err)) from err
    except EvonServiceUnavailableError as err:
        # Explicitly return 503 rather than fake content when Evon is not configured
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(err)) from err
    except EvonTimeoutError as err:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(err)) from err
    except EvonConnectionError as err:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(err)) from err
    except EvonAPIError as err:
        http_code = err.status_code if err.status_code in (400, 401, 403, 422) else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=http_code, detail=str(err)) from err
    except Exception as err:
        logger.exception("Unexpected error in chat endpoint: %s", err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during chat processing.",
        ) from err


@app.get(
    "/api/v1/chat/health",
    summary="Intelligence Engine Health & Model Readiness",
    tags=["Intelligence Engine"],
)
async def chat_health() -> Dict[str, Any]:
    """Return health and configuration status of the Intelligence Engine."""
    active_provider = intelligence_engine.active_provider
    active_configured = (
        gemini_service.is_configured if active_provider == "gemini" else evon_service.is_configured
    )
    return {
        "status": "healthy" if active_configured else "ready_for_configuration",
        "intelligence_engine": "active",
        "active_provider": active_provider,
        "model": intelligence_engine.active_model,
        "provider_configured": active_configured,
        "supported_modes": PersonalityManager.get_supported_modes(),
        "active_conversations": conversation_manager.get_session_count(),
        "providers": {
            "gemini": gemini_service.health_check(),
            "evon": evon_service.health_check(),
        },
    }


@app.post(
    "/api/v1/chat/clear",
    summary="Clear Conversation History",
    tags=["Intelligence Engine"],
)
async def clear_chat(request: ClearChatRequest) -> Dict[str, Any]:
    """Clear dialogue history for a specific conversation session."""
    conversation_manager.clear_conversation(request.conversation_id)
    return {
        "success": True,
        "conversation_id": request.conversation_id,
        "message": "Conversation history cleared.",
    }


@app.post(
    "/api/v1/chat/switch-mode",
    summary="Switch Personality Mode (Unified Comedy)",
    tags=["Intelligence Engine"],
)
async def switch_mode(request: SwitchModeRequest) -> Dict[str, Any]:
    """
    Maintains compatibility for switch-mode calls, mapping all requests
    to TheONE AI's unified comedy personality.
    """
    valid_mode = personality_manager.validate_mode(request.mode)
    conversation_manager.switch_mode(request.conversation_id, valid_mode)
    return {
        "success": True,
        "conversation_id": request.conversation_id,
        "mode": valid_mode,
        "message": f"TheONE AI unified comedy personality active.",
    }


# Mount static assets for frontend (CSS, JS, media)
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
