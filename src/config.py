"""
config.py
=========
Loads `config.yaml` and the `.env` secrets into a single typed `AppConfig`
object so the rest of the code never has to touch raw dicts or os.environ.

Design choice: config (behaviour) is separated from secrets (API keys).
Behaviour is versioned in git via config.yaml; secrets live in .env which is
git-ignored. This is standard 12-factor hygiene and means a reviewer can read
exactly how the system is wired without ever seeing a key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

# Load .env once, as early as possible.
load_dotenv()

SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "de": "German"}


# --------------------------------------------------------------------------- #
# Typed sub-configs                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class STTConfig:
    provider: str
    model: str
    pass_language_hint: bool


@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    reasoning_effort: str


@dataclass
class TTSConfig:
    provider: str
    model: str
    output_format: str
    default_voice_id: str
    voice_by_language: dict[str, str]
    voice_settings: dict[str, Any]

    def voice_for(self, language: str) -> str:
        return self.voice_by_language.get(language, self.default_voice_id)


@dataclass
class RetrievalConfig:
    mode: str                 # "id" | "semantic"
    embedding_model: str
    top_k: int


@dataclass
class InterviewConfig:
    dataset_path: str
    max_follow_ups_per_question: int
    max_questions: Optional[int]


@dataclass
class AppConfig:
    language: str
    interview: InterviewConfig
    stt: STTConfig
    llm: LLMConfig
    tts: TTSConfig
    retrieval: RetrievalConfig
    # secrets (from .env)
    groq_api_key: Optional[str] = field(default=None, repr=False)
    elevenlabs_api_key: Optional[str] = field(default=None, repr=False)

    @property
    def language_name(self) -> str:
        return SUPPORTED_LANGUAGES.get(self.language, "English")

    def require_keys(self) -> None:
        """Fail fast with a clear message if a needed key is missing."""
        missing = []
        if self.stt.provider == "groq" or self.llm.provider == "groq":
            if not self.groq_api_key:
                missing.append("GROQ_API_KEY")
        if self.tts.provider == "elevenlabs" and not self.elevenlabs_api_key:
            missing.append("ELEVENLABS_API_KEY")
        if missing:
            raise RuntimeError(
                "Missing API key(s): "
                + ", ".join(missing)
                + ". Copy .env.example to .env and fill them in."
            )


# --------------------------------------------------------------------------- #
# Loader                                                                       #
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = "config.yaml") -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    language = raw.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Choose one of: {', '.join(SUPPORTED_LANGUAGES)}"
        )

    cfg = AppConfig(
        language=language,
        interview=InterviewConfig(**raw["interview"]),
        stt=STTConfig(**raw["stt"]),
        llm=LLMConfig(**raw["llm"]),
        tts=TTSConfig(**raw["tts"]),
        retrieval=RetrievalConfig(**raw["retrieval"]),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
    )
    return cfg
