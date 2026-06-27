"""
interview.py  —  Orchestration / state machine
==============================================
Owns the *control* of the interview so the LLM doesn't have to. This is the
piece that keeps the agent on-script:

  * The SESSION owns the question pointer (`idx`) and the per-question
    follow-up budget. The model only *requests* to advance or follow up; the
    session decides and enforces the cap. This is why the model can never skip
    questions, invent topics, or loop forever on one question.
  * The session keeps two transcripts: `history` (clean spoken turns fed back
    to the LLM for context) and `transcript` (for display). Grounding/rubric
    text is never stored in either — it is injected fresh each turn and thrown
    away, which keeps the context small (cheaper + faster) and avoids the
    reference leaking into later turns.
  * Per-question assessments are accumulated for the final feedback report.

Speech I/O lives at the app layer; this class is pure text-in/text-out so it
can be unit-tested with a mock LLM (see the __main__ smoke test).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import AppConfig
from .llm import InterviewerLLM, InterviewerTurn
from .reference_store import Question, ReferenceStore


@dataclass
class QuestionRecord:
    question_id: str
    topic: str
    question_text: str
    candidate_answers: list[str] = field(default_factory=list)
    latest_turn: Optional[InterviewerTurn] = None  # carries score/assessment


@dataclass
class TurnResult:
    spoken_response: str
    action: str
    advanced: bool
    finished: bool
    question_number: int          # 1-based index of the question now active
    total_questions: int


class InterviewSession:
    def __init__(self, config: AppConfig, store: ReferenceStore,
                 llm: InterviewerLLM):
        self._cfg = config
        self._store = store
        self._llm = llm
        self.language = config.language
        self.domain = store.domain

        # Build the question order (optionally truncated for a quick demo).
        questions = store.questions
        cap = config.interview.max_questions
        self._order: list[Question] = questions[:cap] if cap else questions

        self._max_follow_ups = config.interview.max_follow_ups_per_question
        self.idx = 0
        self.follow_ups_used = 0
        self.phase = "not_started"  # not_started | in_progress | finished

        self.history: list[dict] = []          # for the LLM
        self.transcript: list[dict] = []        # for display
        self.records: dict[str, QuestionRecord] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    @property
    def total_questions(self) -> int:
        return len(self._order)

    @property
    def is_finished(self) -> bool:
        return self.phase == "finished"

    @property
    def current_question(self) -> Question:
        return self._order[self.idx]

    def start(self) -> str:
        """Begin the interview: returns the interviewer's opening line."""
        if self.phase != "not_started":
            raise RuntimeError("Interview already started.")
        first = self._order[0]
        opening = self._llm.opening(
            first, language=self.language, domain=self.domain
        )
        self.phase = "in_progress"
        self.history.append({"role": "assistant", "content": opening})
        self.transcript.append({"speaker": "interviewer", "text": opening})
        self._ensure_record(first)
        return opening

    def submit_answer(self, answer_text: str) -> TurnResult:
        """Feed one transcribed candidate answer through the interviewer."""
        if self.phase != "in_progress":
            raise RuntimeError("Interview is not in progress.")

        answer_text = (answer_text or "").strip()
        self.transcript.append({"speaker": "candidate", "text": answer_text})

        current = self.current_question
        record = self._ensure_record(current)
        record.candidate_answers.append(answer_text)

        follow_ups_remaining = self._max_follow_ups - self.follow_ups_used
        next_q = self._order[self.idx + 1] if self.idx + 1 < self.total_questions else None
        is_last = self.idx == self.total_questions - 1

        turn = self._llm.respond(
            history=self.history,
            current_question=current,
            candidate_answer=answer_text,
            follow_ups_remaining=follow_ups_remaining,
            next_question=next_q,
            is_last_question=is_last,
            language=self.language,
            domain=self.domain,
        )
        record.latest_turn = turn

        # Update LLM-facing history with the clean spoken exchange.
        self.history.append({"role": "user", "content": answer_text})
        self.history.append({"role": "assistant", "content": turn.spoken_response})
        self.transcript.append(
            {"speaker": "interviewer", "text": turn.spoken_response}
        )

        advanced = turn.action == "advance"
        finished = False
        if advanced:
            if is_last:
                self.phase = "finished"
                finished = True
            else:
                self.idx += 1
                self.follow_ups_used = 0
                self._ensure_record(self.current_question)
        else:
            self.follow_ups_used += 1

        return TurnResult(
            spoken_response=turn.spoken_response,
            action=turn.action,
            advanced=advanced,
            finished=finished,
            question_number=self.idx + 1,
            total_questions=self.total_questions,
        )

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _ensure_record(self, q: Question) -> QuestionRecord:
        if q.id not in self.records:
            self.records[q.id] = QuestionRecord(
                question_id=q.id, topic=q.topic, question_text=q.question
            )
        return self.records[q.id]


# --------------------------------------------------------------------------- #
# Smoke test with a mock LLM (run: python -m src.interview)                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from .config import load_config

    class _MockLLM:
        """Pretends to be the interviewer so we can test control flow offline."""
        def __init__(self):
            self.calls = 0

        def opening(self, first_question, *, language, domain):
            return f"[opening in {language}] Welcome! First: {first_question.question[:40]}..."

        def respond(self, *, history, current_question, candidate_answer,
                    follow_ups_remaining, next_question, is_last_question,
                    language, domain):
            self.calls += 1
            # Advance only when budget is gone OR it's the very first answer;
            # otherwise follow up once — exercises both branches + the cap.
            if follow_ups_remaining <= 0:
                action = "advance"
                spoken = ("[advance] thanks. closing." if is_last_question
                          else f"[advance] thanks. Next: {next_question.question[:40]}...")
            else:
                action = "follow_up"
                spoken = "[follow_up] can you go a bit deeper?"
            return InterviewerTurn(
                spoken_response=spoken, action=action, score=3,
                assessment="mock", covered_key_points=["a"],
                missing_key_points=["b"], raw={},
            )

    cfg = load_config("config.yaml")
    cfg.interview.max_questions = 3   # keep the smoke test short
    store = ReferenceStore(cfg.interview.dataset_path, retrieval_mode="id")
    session = InterviewSession(cfg, store, _MockLLM())

    print("START:", session.start())
    print(f"(question {session.idx + 1}/{session.total_questions}, "
          f"follow_ups_used={session.follow_ups_used})\n")

    step = 0
    while not session.is_finished and step < 30:
        step += 1
        res = session.submit_answer(f"mock answer {step}")
        print(f"answer {step:>2} -> action={res.action:<9} advanced={res.advanced} "
              f"finished={res.finished} now Q{res.question_number}/{res.total_questions}")

    print("\nFINISHED:", session.is_finished)
    print("Records captured:", list(session.records.keys()))
    print("Transcript turns:", len(session.transcript))
