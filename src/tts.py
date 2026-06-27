"""
tts.py  —  Text-to-speech (the "mouth")
=======================================
Turns the interviewer's reply text into audio. Two providers, behind one
`TextToSpeech` interface so the synth is a drop-in swap:

* `edge` (DEFAULT) — Microsoft Edge's online neural voices via `edge-tts`.
  Free, no API key, and native voices for en/hi/de. This is the default so the
  whole project runs on just a free Groq key with nothing paid.

* `elevenlabs` — model `eleven_flash_v2_5` (~75 ms latency, 32 languages).
  The quality/latency upgrade when you have a key. `eleven_multilingual_v2` is
  a one-line higher-quality (slower) swap from there.

Set the provider in config.yaml. Both expose `synthesize(text, *, language)`.
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


class EdgeTTS:
    """Free, no-API-key TTS via Microsoft Edge's online neural voices.

    `edge-tts` talks to the same voice catalogue Azure Speech exposes, so it has
    native voices for en/hi/de at zero cost and no key — which is why it is the
    default here. Quality is very good; ElevenLabs is the swap when you want the
    extra polish. The library is async, so we wrap it to present the same sync
    `synthesize(text, *, language) -> bytes` interface as the other providers.

    `voice_by_language` in config carries Edge voice *names* for this provider
    (e.g. "hi-IN-SwaraNeural"), so switching voices is still a config edit.
    """

    # Sensible native defaults if config doesn't override per language.
    _DEFAULT_VOICES = {
        "en": "en-US-AriaNeural",
        "hi": "hi-IN-SwaraNeural",
        "de": "de-DE-KatjaNeural",
    }

    def __init__(self, config: AppConfig):
        self._cfg = config

    def _voice(self, language: str) -> str:
        # Prefer the configured voice; fall back to a native default; finally en.
        configured = self._cfg.tts.voice_by_language.get(language)
        if configured:
            return configured
        return self._DEFAULT_VOICES.get(language, self._DEFAULT_VOICES["en"])

    def synthesize(self, text: str, *, language: str = "en") -> bytes:
        import asyncio
        import edge_tts

        voice = self._voice(language)

        async def _run() -> bytes:
            communicate = edge_tts.Communicate(text, voice)
            chunks = bytearray()
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio" and chunk.get("data"):
                    chunks.extend(chunk["data"])
            return bytes(chunks)

        return _run_async(_run())


def _run_async(coro) -> bytes:
    """Run an async coroutine from sync code, even if a loop is already running.

    Streamlit's script thread normally has no running loop, so asyncio.run works
    directly; but if one is running we offload to a worker thread with its own
    loop so we never hit 'event loop is already running'.
    """
    import asyncio

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is not None:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def build_tts(config: AppConfig) -> TextToSpeech:
    if config.tts.provider == "edge":
        return EdgeTTS(config)
    if config.tts.provider == "elevenlabs":
        return ElevenLabsTTS(config)
    raise ValueError(f"Unknown TTS provider: {config.tts.provider}")
