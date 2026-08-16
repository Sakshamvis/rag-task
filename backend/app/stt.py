from __future__ import annotations

import time
from typing import Optional, Tuple

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings


class STTError(RuntimeError):
    pass


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.3, min=0.3, max=2))
def _sarvam_transcribe(audio_bytes: bytes, filename: str, mime: str) -> str:
    settings = get_settings()
    if not settings.sarvam_api_key:
        raise STTError(
            "SARVAM_API_KEY missing. Add it to .env to enable voice transcription."
        )

    # Sarvam Speech-to-Text API
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": settings.sarvam_api_key}
    files = {"file": (filename or "audio.wav", audio_bytes, mime or "audio/wav")}
    data = {
        "model": "saarika:v2.5",
        "language_code": "unknown",  # auto-detect Indic + English
    }
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, headers=headers, files=files, data=data)
        if r.status_code >= 400:
            raise STTError(f"Sarvam STT failed ({r.status_code}): {r.text[:400]}")
        payload = r.json()
        text = (
            payload.get("transcript")
            or payload.get("text")
            or payload.get("transcription")
            or ""
        )
        if not str(text).strip():
            raise STTError("Sarvam returned empty transcript.")
        return str(text).strip()


def transcribe_audio(
    audio_bytes: bytes,
    filename: str = "audio.webm",
    mime: str = "audio/webm",
) -> Tuple[str, float]:
    """Returns (transcript, stt_ms)."""
    t0 = time.perf_counter()
    text = _sarvam_transcribe(audio_bytes, filename, mime)
    stt_ms = (time.perf_counter() - t0) * 1000
    return text, stt_ms


def maybe_mock_transcript(fallback: Optional[str] = None) -> Optional[str]:
    return fallback
