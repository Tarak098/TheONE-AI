"""
LaughBot AI - Gnani Prisma Real-Time STT WebSocket Streaming Service.

Implements official Gnani Prisma v3 WebSocket streaming via
wss://api.vachana.ai/stt/v3/stream with x-partials: true, delivering
progressive, low-latency live speech transcription events.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from backend.config import Settings, get_settings

logger = logging.getLogger("laughbot.prisma_stream")

GNANI_STREAM_ENDPOINT = "wss://api.vachana.ai/stt/v3/stream"
PCM_FRAME_SIZE = 1024  # 512 samples @ 16-bit 16kHz (~32ms)
SILENCE_FRAME = b"\x00" * PCM_FRAME_SIZE


def is_ws_open(ws: Any) -> bool:
    """Check if a WebSocket connection is currently in OPEN state across library versions."""
    if ws is None:
        return False
    state = getattr(ws, "state", None)
    if state is not None:
        name = getattr(state, "name", "")
        return name.upper() == "OPEN" or str(state).lower() == "open"
    return bool(getattr(ws, "open", True) and not getattr(ws, "closed", False))


class PrismaStreamError(Exception):
    """Base exception for Prisma WebSocket streaming errors."""
    pass


class PrismaStreamConfigError(PrismaStreamError):
    """Raised when Prisma credentials are missing or invalid."""
    pass


class PrismaStreamConnectionError(PrismaStreamError):
    """Raised when connecting or communicating with Gnani WebSocket fails."""
    pass


class PrismaStreamService:
    """Service for managing Gnani Prisma real-time STT WebSocket sessions."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

    @property
    def api_key(self) -> Optional[str]:
        """Retrieve sanitized Prisma API key."""
        return self._settings.prisma_api_key

    @property
    def is_configured(self) -> bool:
        """Check whether Prisma has valid API credentials configured."""
        return bool(self.api_key)

    def build_headers(self, language_code: str = "en-IN") -> Dict[str, str]:
        """
        Build required WebSocket upgrade headers for Gnani Prisma STT v3.
        Includes 'x-partials': 'true' to enable progressive interim transcription.
        """
        raw_key = (self.api_key or "").strip().strip("\r\n\t\"'“”‘’ ")
        if "=" in raw_key:
            prefix, _, after = raw_key.partition("=")
            if prefix.strip().replace("_", "").isalnum():
                raw_key = after.strip().strip("\r\n\t\"'“”‘’ ")
        clean_key = "".join(c for c in raw_key if 33 <= ord(c) <= 126)

        if not clean_key:
            raise PrismaStreamConfigError("Gnani Prisma STT API credentials are not configured.")

        return {
            "x-api-key-id": clean_key,
            "language_code": language_code or "en-IN",
            "model": "prisma-v3.0",
            "format": "transcribe",
            "x-sample-rate": "16000",
            "x-partials": "true",
        }

    async def bridge_stream(
        self,
        client_ws: WebSocket,
        language_code: str = "en-IN",
    ) -> None:
        """
        Bridge browser WebSocket audio frames with Gnani Prisma WebSocket STT.

        Flow:
        1. Connects to wss://api.vachana.ai/stt/v3/stream with authentication headers.
        2. Concurrently forwards raw 16kHz PCM binary frames from client to Gnani.
        3. Concurrently forwards partial and final transcription events from Gnani to client.
        4. Handles stop signals, silence padding for VAD finalization, and clean disconnection.
        """
        if not self.is_configured:
            logger.error("Attempted streaming STT without configured Prisma credentials.")
            await client_ws.send_json({
                "type": "error",
                "message": "Gnani Prisma STT credentials are not configured.",
            })
            await client_ws.close(code=1008)
            return

        headers = self.build_headers(language_code)
        logger.info(
            "Connecting to Gnani STT stream at %s (lang=%s, model=%s)",
            GNANI_STREAM_ENDPOINT,
            headers["language_code"],
            headers["model"],
        )

        try:
            gnani_ws = await websockets.connect(
                GNANI_STREAM_ENDPOINT,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=15,
                close_timeout=5,
            )
        except Exception as err:
            logger.error("Failed to connect to Gnani Prisma WebSocket: %s", err)
            err_msg = str(err)
            if clean_key and clean_key in err_msg:
                err_msg = err_msg.replace(clean_key, "[REDACTED]")
            if "header value" in err_msg.lower():
                err_msg = "Invalid API key format or header characters. Check GNANI_API_KEY environment variable."
            await client_ws.send_json({
                "type": "error",
                "message": f"Could not connect to Gnani speech service: {err_msg}",
            })
            await client_ws.close(code=1011)
            return

        stop_requested = asyncio.Event()

        async def forward_gnani_to_client():
            """Relay transcription events from Gnani Prisma to the browser client."""
            try:
                async for raw_msg in gnani_ws:
                    try:
                        data = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        logger.warning("Gnani sent non-JSON message: %s", raw_msg[:100])
                        continue

                    event_type = data.get("type")
                    if event_type == "connected":
                        logger.info("Gnani STT stream session established: %s", data.get("request_id"))
                        await client_ws.send_json({
                            "type": "connected",
                            "message": "Speech recognition connected.",
                            "request_id": data.get("request_id"),
                            "config": data.get("config"),
                        })
                    elif event_type == "speech_start":
                        logger.debug("Speech start detected: utterance=%s", data.get("utterance_id"))
                        await client_ws.send_json({
                            "type": "speech_start",
                            "utterance_id": data.get("utterance_id"),
                        })
                    elif event_type == "partial_transcript":
                        # Progressive interim update while user is speaking
                        text = (data.get("text") or "").strip()
                        seq = data.get("seq", 0)
                        stable = data.get("stable_text", "")
                        volatile = data.get("volatile_text", "")
                        logger.debug("Partial STT [seq=%d]: %s", seq, text)
                        await client_ws.send_json({
                            "type": "partial",
                            "text": text,
                            "stable_text": stable,
                            "volatile_text": volatile,
                            "seq": seq,
                            "is_final": False,
                        })
                    elif event_type == "speech_end":
                        logger.debug("Speech end detected: utterance=%s", data.get("utterance_id"))
                        await client_ws.send_json({
                            "type": "speech_end",
                            "utterance_id": data.get("utterance_id"),
                        })
                    elif event_type == "final_transcript":
                        text = (data.get("text") or "").strip()
                        seg_idx = data.get("segment_index", 1)
                        detected_lang = data.get("detected_language_code", language_code)
                        logger.info("Final STT [seg=%d]: %s (lang=%s)", seg_idx, text, detected_lang)
                        await client_ws.send_json({
                            "type": "final",
                            "text": text,
                            "segment_index": seg_idx,
                            "language": detected_lang,
                            "is_final": True,
                        })
                    elif event_type == "error":
                        err_msg = data.get("message") or data.get("error") or "Unknown Prisma error"
                        logger.error("Gnani STT error event: %s", err_msg)
                        await client_ws.send_json({
                            "type": "error",
                            "message": f"Gnani STT error: {err_msg}",
                        })
            except (ConnectionClosed, asyncio.CancelledError):
                pass
            except Exception as err:
                logger.exception("Error in forward_gnani_to_client: %s", err)

        async def forward_client_to_gnani():
            """Relay 16kHz PCM binary frames from browser client to Gnani Prisma."""
            try:
                while True:
                    msg = await client_ws.receive()
                    msg_type = msg.get("type")

                    if msg_type == "websocket.disconnect":
                        logger.info("Client WebSocket disconnected.")
                        break

                    # Binary PCM frames
                    if "bytes" in msg and msg["bytes"]:
                        pcm_chunk = msg["bytes"]
                        if is_ws_open(gnani_ws):
                            await gnani_ws.send(pcm_chunk)
                        continue

                    # Text control commands
                    if "text" in msg and msg["text"]:
                        try:
                            ctrl = json.loads(msg["text"])
                        except json.JSONDecodeError:
                            continue

                        cmd = ctrl.get("type")
                        if cmd == "stop":
                            logger.info("Client requested stop. Sending silence padding for VAD finalization...")
                            stop_requested.set()
                            # Send 45 frames of silence (~1.4s) to ensure Gnani's server VAD closes the utterance
                            if is_ws_open(gnani_ws):
                                for _ in range(45):
                                    await gnani_ws.send(SILENCE_FRAME)
                                    await asyncio.sleep(0.032)
                        elif cmd == "close":
                            logger.info("Client requested stream close.")
                            break
            except (WebSocketDisconnect, asyncio.CancelledError):
                pass
            except Exception as err:
                logger.exception("Error in forward_client_to_gnani: %s", err)

        # Run client -> gnani and gnani -> client concurrently
        client_task = asyncio.create_task(forward_client_to_gnani())
        gnani_task = asyncio.create_task(forward_gnani_to_client())

        done, pending = await asyncio.wait(
            [client_task, gnani_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        # Cleanly close Gnani WebSocket
        if is_ws_open(gnani_ws):
            try:
                await gnani_ws.close()
            except Exception:
                pass

        logger.info("Prisma STT streaming session ended.")
