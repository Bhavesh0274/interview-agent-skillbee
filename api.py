"""
api.py  —  FastAPI backend for the voice interview agent
========================================================
This exposes the SAME interview core (interview.py / llm.py / feedback.py) as an
HTTP API instead of a Streamlit app. Nothing in src/ changes — that core was
written to be pure text-in/text-out, so the only things this layer has to add
are the two jobs Streamlit was quietly doing for us:

  1. STATE  — Streamlit kept the InterviewSession in st.session_state (one per
     browser tab). HTTP is stateless, so we keep sessions in a server-side
     store keyed by a session_id that the client passes back on each call.
     (In-memory dict here for a prototype; swap for Redis in production so
     sessions survive restarts and are shared across worker processes.)

  2. AUDIO  — Streamlit had st.audio_input (mic) and st.audio (playback) built
     in. An API has no UI, so the *client* records the audio and POSTs it; we
     run STT -> session -> TTS and return the interviewer's reply audio
     (base64) + transcript in the JSON response, and the client plays it.

Concurrency note (worth understanding for the interview):
  The STT/LLM/TTS calls are blocking, synchronous SDK calls. We therefore define
  the path operations as plain `def` (not `async def`). FastAPI runs sync
  endpoints in a threadpool, so a slow model call blocks only its own worker
  thread, never the event loop — which keeps the server responsive to other
  requests. (The alternative is `async def` + run_in_threadpool / async SDK
  clients; `def` is the simplest correct choice here.)

Run locally:
    uvicorn api:app --reload --port 8000
Then open the interactive docs at  http://localhost:8000/docs
"""
from __future__ import annotations

import base64
import uuid
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import load_config, SUPPORTED_LANGUAGES
from src.reference_store import ReferenceStore
from src.stt import build_stt
from src.tts import build_tts
from src.llm import InterviewerLLM
from src.interview import InterviewSession
from src.feedback import FeedbackGenerator

CONFIG_PATH = "config.yaml"

# --------------------------------------------------------------------------- #
# Shared resources (built once at startup; all are language-agnostic because   #
# language is passed per-call, so one instance serves every session/language). #
# --------------------------------------------------------------------------- #
_BASE_CFG = load_config(CONFIG_PATH)
_BASE_CFG.require_keys()  # fail fast if GROQ_API_KEY is missing

STORE = ReferenceStore(
    _BASE_CFG.interview.dataset_path,
    retrieval_mode=_BASE_CFG.retrieval.mode,
    embedding_model=_BASE_CFG.retrieval.embedding_model,
)
STT = build_stt(_BASE_CFG)
TTS = build_tts(_BASE_CFG)
LLM = InterviewerLLM(_BASE_CFG)
FEEDBACK = FeedbackGenerator(_BASE_CFG)

# Server-side session store. NOTE: in-memory + single-process only.
# Production: replace with Redis (and add a TTL so abandoned sessions expire).
SESSIONS: dict[str, InterviewSession] = {}

app = FastAPI(title="Voice Interview Agent API", version="1.0")

# Allow a browser front-end on another origin to call this API.
# Tighten allow_origins to your real front-end domain in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Response models (FastAPI serialises these to JSON + documents them)          #
# --------------------------------------------------------------------------- #
class TurnResponse(BaseModel):
    session_id: str
    interviewer_text: str
    audio_base64: Optional[str]          # mp3 bytes, base64; None if TTS failed
    question_number: int
    total_questions: int
    finished: bool
    your_transcript: Optional[str] = None  # what STT heard (answer turns only)


class SessionState(BaseModel):
    session_id: str
    domain: str
    language: str
    question_number: int
    total_questions: int
    finished: bool
    transcript: list[dict]


class PerQuestionFB(BaseModel):
    question_id: str
    topic: str
    score: Optional[int]
    strengths: str
    improvements: str


class FeedbackResponse(BaseModel):
    overall_summary: str
    overall_score: float
    strengths: list[str]
    improvement_areas: list[str]
    per_question: list[PerQuestionFB]
    spoken_summary: str
    audio_base64: Optional[str] = None   # optional spoken summary audio


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _safe_tts(text: str, language: str) -> Optional[str]:
    """Synthesize speech, base64-encoded. Returns None on failure so the API
    degrades to text-only instead of 500-ing (graceful degradation)."""
    try:
        audio = TTS.synthesize(text, language=language)
        return base64.b64encode(audio).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def _get_session(session_id: str) -> InterviewSession:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id.")
    return session


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "domain": STORE.domain,
            "languages": list(SUPPORTED_LANGUAGES.keys())}


@app.post("/sessions", response_model=TurnResponse)
def create_session(language: str = "en", max_questions: Optional[int] = None):
    """Start a new interview in the chosen language. Returns the opening line
    (text + audio). The returned session_id must be sent on every later call."""
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language '{language}'. "
                   f"Choose from {list(SUPPORTED_LANGUAGES.keys())}.",
        )

    # Per-session config so each session can run in its own language
    # (InterviewSession reads config.language). Cheap: just a YAML parse.
    cfg = load_config(CONFIG_PATH)
    cfg.language = language
    if max_questions:
        cfg.interview.max_questions = max_questions

    session = InterviewSession(cfg, STORE, LLM)
    opening = session.start()

    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = session

    return TurnResponse(
        session_id=session_id,
        interviewer_text=opening,
        audio_base64=_safe_tts(opening, language),
        question_number=session.idx + 1,
        total_questions=session.total_questions,
        finished=session.is_finished,
    )


@app.post("/sessions/{session_id}/answer", response_model=TurnResponse)
def submit_answer(session_id: str, audio: UploadFile = File(...)):
    """Upload the candidate's recorded answer (any common audio format).
    Pipeline: STT -> interviewer LLM -> TTS. Returns the interviewer's reply."""
    session = _get_session(session_id)
    if session.is_finished:
        raise HTTPException(status_code=409, detail="Interview already finished.")

    # Sync read of the upload (we're in a sync endpoint running in a threadpool).
    audio_bytes = audio.file.read()
    filename = audio.filename or "answer.wav"

    # --- speech -> text ---
    try:
        text = STT.transcribe(audio_bytes, filename=filename,
                              language=session.language)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Transcription failed: {exc}")
    if not (text or "").strip():
        raise HTTPException(status_code=422,
                            detail="Could not hear any speech — please re-record.")

    # --- interviewer brain ---
    result = session.submit_answer(text)

    # --- text -> speech ---
    audio_b64 = _safe_tts(result.spoken_response, session.language)

    return TurnResponse(
        session_id=session_id,
        interviewer_text=result.spoken_response,
        audio_base64=audio_b64,
        question_number=result.question_number,
        total_questions=result.total_questions,
        finished=result.finished,
        your_transcript=text,
    )


@app.post("/sessions/{session_id}/feedback", response_model=FeedbackResponse)
def get_feedback(session_id: str, speak: bool = False):
    """End the interview (if not already) and return the structured feedback
    report. Pass ?speak=true to also get a short spoken summary as audio."""
    session = _get_session(session_id)
    # Allow ending early: the orchestrator just marks the phase finished.
    session.phase = "finished"

    fb = FEEDBACK.generate(session)
    audio_b64 = _safe_tts(fb.spoken_summary, session.language) if (speak and fb.spoken_summary) else None

    return FeedbackResponse(
        overall_summary=fb.overall_summary,
        overall_score=fb.overall_score,
        strengths=fb.strengths,
        improvement_areas=fb.improvement_areas,
        per_question=[
            PerQuestionFB(
                question_id=p.question_id, topic=p.topic, score=p.score,
                strengths=p.strengths, improvements=p.improvements,
            )
            for p in fb.per_question
        ],
        spoken_summary=fb.spoken_summary,
        audio_base64=audio_b64,
    )


@app.get("/sessions/{session_id}", response_model=SessionState)
def session_state(session_id: str):
    """Inspect the current state / full transcript of a session."""
    session = _get_session(session_id)
    return SessionState(
        session_id=session_id,
        domain=session.domain,
        language=session.language,
        question_number=min(session.idx + 1, session.total_questions),
        total_questions=session.total_questions,
        finished=session.is_finished,
        transcript=session.transcript,
    )


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Free a finished/abandoned session."""
    SESSIONS.pop(session_id, None)
    return {"deleted": session_id}


if __name__ == "__main__":
    import os
    import uvicorn
    # Hosts (Render/Railway/Cloud Run/…) inject the port to bind via $PORT.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
