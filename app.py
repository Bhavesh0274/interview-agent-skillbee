"""
app.py  —  Streamlit voice interface
====================================
The thin "voice shell" around the core session. It is deliberately the ONLY
place that touches microphones, audio playback, and Streamlit state — the
interview logic itself (interview.py / llm.py / feedback.py) is pure
text-in/text-out and unaware that voice exists. That separation is what let us
unit-test the control flow offline.

Per turn the shell does four things:
    record  ->  STT (Groq Whisper)  ->  session.submit_answer (LLM)  ->  TTS
and renders the running transcript.

State & the audio-rerun gotcha
------------------------------
Streamlit reruns the whole script on every interaction. `st.audio_input`
returns the SAME recording object on each rerun until a new one is recorded, so
naively transcribing whatever it returns would reprocess the same audio over
and over. We guard against that by remembering the processed recording's
`file_id` in `st.session_state` and only acting when a genuinely new id appears.
Everything that must survive a rerun (the session object, the transcript, the
last audio blob, the feedback) lives in `st.session_state`.

Graceful degradation
--------------------
Voice is best-effort around a text core: if TTS fails we still show the text; if
STT fails we keep the turn and ask the candidate to repeat. The interview never
hard-crashes on a transient API error.
"""
from __future__ import annotations

import streamlit as st

from src.config import load_config, SUPPORTED_LANGUAGES
from src.reference_store import ReferenceStore
from src.stt import build_stt
from src.tts import build_tts
from src.llm import InterviewerLLM
from src.interview import InterviewSession
from src.feedback import FeedbackGenerator, render_feedback_markdown

st.set_page_config(page_title="Voice Interview Agent", page_icon="🎙️")

LANG_LABELS = {"en": "English", "hi": "हिन्दी (Hindi)", "de": "Deutsch (German)"}


# --------------------------------------------------------------------------- #
# Resource construction (cached so we don't rebuild clients every rerun)       #
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_base_config():
    return load_config("config.yaml")


@st.cache_resource(show_spinner=False)
def get_store(dataset_path: str, mode: str, embedding_model: str):
    return ReferenceStore(dataset_path, retrieval_mode=mode,
                          embedding_model=embedding_model)


@st.cache_resource(show_spinner=False)
def get_clients(language: str):
    """Build STT/LLM/TTS for a given language. Cached per language."""
    cfg = load_config("config.yaml")
    cfg.language = language
    cfg.require_keys()
    return cfg, build_stt(cfg), build_tts(cfg), InterviewerLLM(cfg)


# --------------------------------------------------------------------------- #
# Session bootstrap                                                            #
# --------------------------------------------------------------------------- #
def init_state():
    ss = st.session_state
    ss.setdefault("session", None)
    ss.setdefault("last_audio_id", None)
    ss.setdefault("last_audio_bytes", None)
    ss.setdefault("feedback_md", None)
    ss.setdefault("status", "")
    ss.setdefault("started", False)


def start_interview(language: str):
    cfg, stt, tts, llm = get_clients(language)
    store = get_store(cfg.interview.dataset_path, cfg.retrieval.mode,
                      cfg.retrieval.embedding_model)
    session = InterviewSession(cfg, store, llm)
    opening = session.start()

    ss = st.session_state
    ss.session = session
    ss.stt, ss.tts = stt, tts
    ss.cfg = cfg
    ss.feedback_md = None
    ss.last_audio_id = None
    ss.started = True
    ss.status = ""
    # Speak the opening line.
    ss.last_audio_bytes = _safe_tts(tts, opening, language)


def reset_interview():
    for k in ("session", "last_audio_id", "last_audio_bytes",
              "feedback_md", "status", "started", "stt", "tts", "cfg"):
        st.session_state.pop(k, None)
    init_state()


# --------------------------------------------------------------------------- #
# Best-effort speech helpers                                                   #
# --------------------------------------------------------------------------- #
def _safe_tts(tts, text: str, language: str):
    try:
        return tts.synthesize(text, language=language)
    except Exception as exc:  # noqa: BLE001
        st.session_state.status = f"(voice output unavailable: {exc})"
        return None


def _safe_stt(stt, audio_bytes: bytes, filename: str, language: str):
    try:
        return stt.transcribe(audio_bytes, filename=filename, language=language)
    except Exception as exc:  # noqa: BLE001
        st.session_state.status = f"(could not transcribe — please try again: {exc})"
        return None


# --------------------------------------------------------------------------- #
# Turn handling                                                                #
# --------------------------------------------------------------------------- #
def handle_answer(audio_file):
    """Process one freshly recorded answer (only when it's new)."""
    ss = st.session_state
    # Guard against Streamlit's rerun replaying the same recording.
    if audio_file is None or audio_file.file_id == ss.last_audio_id:
        return
    ss.last_audio_id = audio_file.file_id

    session: InterviewSession = ss.session
    language = session.language
    audio_bytes = audio_file.getvalue()
    filename = getattr(audio_file, "name", "answer.wav") or "answer.wav"

    with st.spinner("Transcribing…"):
        text = _safe_stt(ss.stt, audio_bytes, filename, language)
    if not text:
        # Transcription failed or was empty; let the user re-record.
        return

    with st.spinner("Interviewer is thinking…"):
        result = session.submit_answer(text)

    with st.spinner("Speaking…"):
        ss.last_audio_bytes = _safe_tts(ss.tts, result.spoken_response, language)

    if result.finished:
        _generate_feedback()


def _generate_feedback():
    ss = st.session_state
    with st.spinner("Preparing your feedback…"):
        try:
            fb = FeedbackGenerator(ss.cfg).generate(ss.session)
            ss.feedback_md = render_feedback_markdown(
                fb, domain=ss.session.domain,
                language_name=ss.cfg.language_name,
            )
            # Optional short spoken wrap-up.
            if fb.spoken_summary:
                ss.last_audio_bytes = _safe_tts(
                    ss.tts, fb.spoken_summary, ss.session.language
                )
        except Exception as exc:  # noqa: BLE001
            ss.feedback_md = f"_Could not generate feedback: {exc}_"


# --------------------------------------------------------------------------- #
# UI                                                                           #
# --------------------------------------------------------------------------- #
def render_transcript(session: InterviewSession):
    for turn in session.transcript:
        if turn["speaker"] == "interviewer":
            with st.chat_message("assistant", avatar="🎙️"):
                st.write(turn["text"])
        else:
            with st.chat_message("user", avatar="🧑"):
                st.write(turn["text"] or "_(no answer)_")


def main():
    init_state()
    ss = st.session_state

    # Domain label (read from the dataset, no key needed).
    base_cfg = get_base_config()
    store_preview = get_store(base_cfg.interview.dataset_path,
                              base_cfg.retrieval.mode,
                              base_cfg.retrieval.embedding_model)

    st.title("🎙️ Voice Interview Agent")
    st.caption(f"Domain: **{store_preview.domain}** · "
               "Speak your answers; the agent listens, follows up, and scores you.")

    # ---- Sidebar: configuration ----
    with st.sidebar:
        st.header("Setup")
        lang_code = st.selectbox(
            "Interview language",
            options=list(SUPPORTED_LANGUAGES.keys()),
            format_func=lambda c: LANG_LABELS.get(c, c),
            index=list(SUPPORTED_LANGUAGES.keys()).index(base_cfg.language)
            if base_cfg.language in SUPPORTED_LANGUAGES else 0,
            disabled=ss.started,
            help="Language is locked once the interview starts. Reset to change it.",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Start", type="primary", disabled=ss.started,
                         use_container_width=True):
                try:
                    start_interview(lang_code)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not start: {exc}")
                st.rerun()
        with col_b:
            if st.button("Reset", disabled=not ss.started,
                         use_container_width=True):
                reset_interview()
                st.rerun()

        st.divider()
        st.markdown(
            "**Tips**\n\n"
            "- Answer out loud, then stop the recording.\n"
            "- The agent may probe deeper before moving on.\n"
            "- Click **End interview** any time for feedback."
        )

    # ---- Not started yet ----
    if not ss.started or ss.session is None:
        st.info("Pick a language in the sidebar and press **Start** to begin.")
        return

    session: InterviewSession = ss.session

    # ---- Progress ----
    q_num = min(session.idx + 1, session.total_questions)
    st.progress(
        (session.idx) / session.total_questions if session.total_questions else 0.0,
        text=f"Question {q_num} of {session.total_questions}",
    )

    # ---- Transcript ----
    render_transcript(session)

    # ---- Latest interviewer audio ----
    if ss.last_audio_bytes:
        st.audio(ss.last_audio_bytes, format="audio/mp3", autoplay=True)
    if ss.status:
        st.caption(ss.status)

    # ---- Feedback (if finished) ----
    if session.is_finished:
        st.success("Interview complete.")
        if ss.feedback_md is None:
            _generate_feedback()
        if ss.feedback_md:
            st.markdown(ss.feedback_md)
            st.download_button("⬇️ Download feedback (Markdown)",
                               ss.feedback_md, file_name="interview_feedback.md")
        return

    # ---- Answer recorder (active turn) ----
    st.subheader("Your answer")
    audio_file = st.audio_input("Record, then stop to submit",
                                key=f"rec_{session.idx}_{session.follow_ups_used}")
    if audio_file is not None:
        handle_answer(audio_file)
        st.rerun()

    if st.button("End interview & get feedback"):
        session.phase = "finished"
        _generate_feedback()
        st.rerun()


if __name__ == "__main__":
    main()
