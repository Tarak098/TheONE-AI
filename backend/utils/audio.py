"""
LaughBot AI - Audio Validation and Utilities.

Validates uploaded WAV audio files, ensuring structural integrity,
size compliance, duration limits, and audio format requirements for
the Gnani Prisma Speech-to-Text service.
"""

from dataclasses import dataclass
import io
import logging
from pathlib import Path
from typing import List, Optional
import uuid
import wave

from backend.config import get_settings

logger = logging.getLogger("laughbot.audio")

# Maximum upload size: 10 MB
MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024

# Minimum and maximum duration in seconds
MIN_AUDIO_DURATION_SECONDS = 0.1
MAX_AUDIO_DURATION_SECONDS = 60.0

SUPPORTED_AUDIO_EXTENSIONS: List[str] = [
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".m4a",
    ".webm",
]


class AudioValidationError(ValueError):
    """Raised when uploaded audio fails format, size, or integrity checks."""
    pass


@dataclass
class AudioMetadata:
    """Metadata extracted from a validated WAV audio payload."""
    channels: int
    sample_rate: int
    sample_width: int
    frames: int
    duration_seconds: float
    size_bytes: int


def get_supported_audio_formats() -> List[str]:
    """Return the list of supported audio file extensions."""
    return list(SUPPORTED_AUDIO_EXTENSIONS)


def is_supported_audio_format(filename_or_extension: str) -> bool:
    """Check whether a given filename or extension is in the supported audio formats."""
    ext = filename_or_extension.lower().strip()
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext in SUPPORTED_AUDIO_EXTENSIONS


def save_temp_audio_file(
    audio_bytes: bytes,
    extension: str = ".wav",
    temp_dir: Optional[Path] = None,
) -> Path:
    """Save raw audio bytes to a temporary file in the configured audio directory."""
    settings = get_settings()
    target_dir = temp_dir or (settings.PROJECT_ROOT / "temp_audio" if hasattr(settings, "PROJECT_ROOT") else Path("temp_audio"))
    target_dir.mkdir(parents=True, exist_ok=True)

    ext = extension if extension.startswith(".") else f".{extension}"
    unique_filename = f"audio_{uuid.uuid4().hex[:12]}{ext}"
    output_path = target_dir / unique_filename

    output_path.write_bytes(audio_bytes)
    return output_path


def cleanup_temp_file(file_path: Path) -> bool:
    """Safely delete a temporary audio file if it exists."""
    try:
        if file_path.exists():
            file_path.unlink()
            return True
    except OSError as err:
        logger.warning("Failed to remove temporary audio file %s: %s", file_path, err)
    return False


def validate_wav_audio(
    audio_bytes: bytes,
    max_size: int = MAX_AUDIO_SIZE_BYTES,
    min_duration: float = MIN_AUDIO_DURATION_SECONDS,
    max_duration: float = MAX_AUDIO_DURATION_SECONDS,
) -> AudioMetadata:
    """
    Validate that the provided bytes represent a structurally valid WAV file
    meeting size, duration, and encoding requirements for Gnani Prisma.

    Args:
        audio_bytes: Raw binary bytes of the WAV file.
        max_size: Maximum allowed file size in bytes (default 10 MB).
        min_duration: Minimum required audio duration in seconds.
        max_duration: Maximum allowed audio duration in seconds.

    Returns:
        AudioMetadata: Information about the validated audio.

    Raises:
        AudioValidationError: If the file is missing, empty, too large, corrupt,
                              or has an invalid duration/format.
    """
    if not audio_bytes:
        raise AudioValidationError("Audio file is empty (0 bytes received).")

    size_bytes = len(audio_bytes)
    if size_bytes > max_size:
        max_mb = max_size / (1024 * 1024)
        actual_mb = size_bytes / (1024 * 1024)
        raise AudioValidationError(
            f"Audio file size ({actual_mb:.2f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB."
        )

    # Basic RIFF/WAVE header sanity check
    if size_bytes < 44 or not (audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:16]):
        raise AudioValidationError("Invalid audio format: file does not contain a valid WAV/RIFF header.")

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.getnframes()
    except wave.Error as err:
        logger.warning("Corrupted WAV payload: %s", err)
        raise AudioValidationError(f"Corrupted or invalid WAV file: {err}") from err
    except Exception as err:
        logger.warning("Unexpected error inspecting WAV header: %s", err)
        raise AudioValidationError("Unable to parse audio: corrupted WAV file.") from err

    if sample_rate <= 0:
        raise AudioValidationError(f"Invalid WAV sample rate: {sample_rate} Hz.")

    duration_seconds = frames / float(sample_rate)

    if frames == 0 or duration_seconds < min_duration:
        raise AudioValidationError(
            f"Audio recording is too short or silent ({duration_seconds:.2f}s). "
            f"Minimum required is {min_duration}s."
        )

    if duration_seconds > max_duration:
        raise AudioValidationError(
            f"Audio duration ({duration_seconds:.1f}s) exceeds the maximum allowed limit of {max_duration:.0f}s."
        )

    # Supported sample widths (16-bit PCM = 2 bytes is standard for Prisma)
    if sample_width not in (1, 2, 3, 4):
        raise AudioValidationError(f"Unsupported sample width: {sample_width * 8}-bit.")

    logger.debug(
        "WAV audio validated: channels=%d, sample_rate=%d Hz, width=%d bytes, duration=%.2fs, size=%d bytes",
        channels,
        sample_rate,
        sample_width,
        duration_seconds,
        size_bytes,
    )

    return AudioMetadata(
        channels=channels,
        sample_rate=sample_rate,
        sample_width=sample_width,
        frames=frames,
        duration_seconds=duration_seconds,
        size_bytes=size_bytes,
    )
