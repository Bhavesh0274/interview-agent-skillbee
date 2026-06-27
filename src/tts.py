"""
tts.py  —  Text-to-speech (the "mouth")
=======================================
Turns the interviewer's reply text into audio.

Provider: ElevenLabs, model `eleven_flash_v2_5` by default.
Why this model:
  * ~75ms model latency — the lowest-latency option that still covers all 32
    languages we care about (en/hi/de). For a voice agent the brief explicitly
    flags latency, so the *responsive* model is the right default.
  * One model + one multilingual voice handles all three languages; we pass an
    ISO `language_code` to pin pronunciation rather than relying on auto-detect.
  * `eleven_multilingual_v2` is available as a higher-quality (but slower)
    swap — change one line in config.yaml.

Like STT, this sits behind a `TextToSpeech` interface so the synth provider is
a drop-in swap.
"""
from __future__ import annotations

from typing import Protocol

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

from .config import AppConfig


class TextToSpeech(Protocol):
    def synthesize(self, text: str, *, language: str) -> bytes:
        ...


class ElevenLabsTTS:
    def __init__(self, config: AppConfig):
        self._cfg = config
        self._model = config.tts.model
        self._output_format = config.tts.output_format
        vs = config.tts.voice_settings
        self._voice_settings = VoiceSettings(
            stability=vs.get("stability", 0.5),
            similarity_boost=vs.get("similarity_boost", 0.75),
            style=vs.get("style", 0.0),
            use_speaker_boost=vs.get("use_speaker_boost", True),
        )
        self._client = ElevenLabs(api_key=config.elevenlabs_api_key)

    # Only the low-latency models accept an explicit language_code; the
    # higher-quality `eleven_multilingual_v2` rejects it (it auto-detects).
    # We guard for it so swapping the model in config never breaks the call.
    _LANG_CODE_MODELS = {"eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_v3"}

    def synthesize(self, text: str, *, language: str = "en") -> bytes:
        voice_id = self._cfg.tts.voice_for(language)
        kwargs = dict(
            text=text,
            voice_id=voice_id,
            model_id=self._model,
            output_format=self._output_format,
            voice_settings=self._voice_settings,
        )
        if self._model in self._LANG_CODE_MODELS:
            kwargs["language_code"] = language   # pin pronunciation to en/hi/de
        # `convert` returns an iterator of byte chunks; join into one blob that
        # Streamlit can play in a single widget. (For a production streaming UI
        # you would forward the chunks instead of buffering — see ARCHITECTURE.)
        audio_iter = self._client.text_to_speech.convert(**kwargs)
        return b"".join(chunk for chunk in audio_iter if chunk)


def build_tts(config: AppConfig) -> TextToSpeech:
    if config.tts.provider == "elevenlabs":
        return ElevenLabsTTS(config)
    raise ValueError(f"Unknown TTS provider: {config.tts.provider}")
