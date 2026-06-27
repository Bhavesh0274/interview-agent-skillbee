"""
cli.py  —  Text-mode interview runner
=====================================
Runs the FULL interview loop in the terminal with typed answers instead of
voice. Same orchestrator, same LLM, same feedback — only the speech I/O is
skipped. This exists so the core logic can be exercised and debugged without a
microphone, audio drivers, or the TTS bill, and so a reviewer can try the agent
in five seconds.

Usage:
    python cli.py                 # uses language from config.yaml
    python cli.py --language hi   # override language for this run
    python cli.py --max-questions 3   # short run

Type your answer and press Enter. Type ':skip' to give an empty answer,
or ':quit' to abort.
"""
from __future__ import annotations

import argparse

from src.config import load_config, SUPPORTED_LANGUAGES
from src.reference_store import ReferenceStore
from src.llm import InterviewerLLM
from src.interview import InterviewSession
from src.feedback import FeedbackGenerator, render_feedback_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Text-mode mock interview.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--language", choices=sorted(SUPPORTED_LANGUAGES),
                        help="Override interview language (en/hi/de).")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions (quick demo).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.language:
        cfg.language = args.language
    if args.max_questions:
        cfg.interview.max_questions = args.max_questions
    cfg.require_keys()  # fail fast if GROQ_API_KEY is missing

    store = ReferenceStore(
        cfg.interview.dataset_path,
        retrieval_mode=cfg.retrieval.mode,
        embedding_model=cfg.retrieval.embedding_model,
    )
    llm = InterviewerLLM(cfg)
    session = InterviewSession(cfg, store, llm)

    print("=" * 70)
    print(f"  Mock interview — {store.domain}  ({cfg.language_name})")
    print(f"  {session.total_questions} questions, "
          f"up to {cfg.interview.max_follow_ups_per_question} follow-ups each.")
    print("  Type ':quit' to abort.")
    print("=" * 70, "\n")

    opening = session.start()
    print(f"INTERVIEWER › {opening}\n")

    while not session.is_finished:
        try:
            answer = input("YOU        › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[aborted]")
            return
        if answer == ":quit":
            print("[aborted]")
            return
        if answer == ":skip":
            answer = ""

        result = session.submit_answer(answer)
        tag = f"(Q{result.question_number}/{result.total_questions})"
        print(f"\nINTERVIEWER › {result.spoken_response}  {tag}\n")

    print("=" * 70)
    print("  Interview complete — generating feedback...")
    print("=" * 70, "\n")

    fb = FeedbackGenerator(cfg).generate(session)
    print(render_feedback_markdown(
        fb, domain=store.domain, language_name=cfg.language_name
    ))


if __name__ == "__main__":
    main()
