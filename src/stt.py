"""
stt.py  —  Speech-to-text (the "ear")
=====================================
Turns the candidate's recorded audio into text.

Provider: Groq's hosted `whisper-large-v3-turbo`.
Why this model:
  * Multilingual out of the box (covers en/hi/de — the hard requirement) with
    a single model, so we don't juggle one ASR per language.
  * On Groq it runs at ~216x real-time, which keeps the first leg of the
    pipeline well under the latency budget for a turn-based voice agent.
  * It is a managed API — no GPU, no model hosting — which is exactly what the
    brief asks for ("use managed APIs freely").

Whisper auto-detects language, but we pass an explicit hint for hi/de because a
short, accented utterance is occasionally mis-detected; the hint removes that
failure mode at zero cost.

The provider sits behind a tiny `SpeechToText` interface so swapping to
Deepgram/Azure later is a one-class change, not a refactor.
"""
from __future__ import annotations

import io
from typing import Protocol

from groq import Groq

from .config import AppConfig


class SpeechToText(Protocol):
    def transcribe(self, audio_bytes: bytes, *, filename: str, language: str) -> str:
        ...


class GroqSTT:
    def __init__(self, config: AppConfig):
        self._cfg = config
        self._model = config.stt.model
        self._pass_hint = config.stt.pass_language_hint
        self._client = Groq(api_key=config.groq_api_key)

    def transcribe(self, audio_bytes: bytes, *, filename: str = "audio.wav",
                   language: str = "en") -> str:
        # Groq accepts a (filename, bytes) tuple; the extension tells it the
        # container format (wav/webm/mp3/m4a/...).
        file_tuple = (filename, io.BytesIO(audio_bytes).read())

        kwargs = {
            "file": file_tuple,
            "model": self._model,
            "response_format": "text",
            "temperature": 0.0,   # deterministic transcription
        }
        if self._pass_hint and language in {"en", "hi", "de"}:
            kwargs["language"] = language

        result = self._client.audio.transcriptions.create(**kwargs)
        # With response_format="text" the SDK returns a plain string;
        # with json it returns an object exposing `.text`. Handle both.
        text = result if isinstance(result, str) else getattr(result, "text", str(result))
        return text.strip()


def build_stt(config: AppConfig) -> SpeechToText:
    if config.stt.provider == "groq":
        return GroqSTT(config)
    raise ValueError(f"Unknown STT provider: {config.stt.provider}")
