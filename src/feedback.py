"""
feedback.py  —  End-of-interview structured feedback
====================================================
After the last question, we summarise how the candidate did. The orchestrator
(interview.py) has already accumulated a per-question record containing, for
every question:
  * the question + topic,
  * every answer the candidate gave (including follow-up attempts),
  * the interviewer's LAST private judgement for that question
    (score 0-5, covered_key_points, missing_key_points, assessment).

Design choices
--------------
* We make ONE LLM call for the whole report rather than one per question. The
  raw material (the scores and the covered/missing points) was already produced
  turn-by-turn during the interview, so the final call is pure *synthesis*:
  cheaper, faster, and it lets the model write a coherent narrative that
  references the interview as a whole.
* We feed the model the private per-question judgements (this is the one place
  they are allowed to surface, because the interview is over and this IS the
  feedback). The numeric `overall_score` is computed in CODE from the per-turn
  scores, not by the model — deterministic and not subject to the model being
  generous. The model writes the prose; the code owns the number.
* Output is structured JSON so the UI can render sections cleanly, and the
  prose fields are written in the interview language.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from .config import AppConfig
from .interview import InterviewSession, QuestionRecord
from .llm import LANG_NAME, _extract_json   # reuse the tolerant JSON parser

from groq import Groq


@dataclass
class PerQuestionFeedback:
    question_id: str
    topic: str
    score: Optional[int]
    strengths: str
    improvements: str


@dataclass
class InterviewFeedback:
    overall_summary: str
    overall_score: float                 # 0-5, computed in code
    strengths: list[str] = field(default_factory=list)
    improvement_areas: list[str] = field(default_factory=list)
    per_question: list[PerQuestionFeedback] = field(default_factory=list)
    spoken_summary: str = ""             # short version, for optional TTS
    raw: dict = field(default_factory=dict)


class FeedbackGenerator:
    def __init__(self, config: AppConfig):
        self._cfg = config
        self._model = config.llm.model
        self._client = Groq(api_key=config.groq_api_key)

    # ------------------------------------------------------------------ #
    def generate(self, session: InterviewSession) -> InterviewFeedback:
        records = [session.records[q.id] for q in session._order
                   if q.id in session.records]

        # ----- deterministic numeric score (code owns the number) ----- #
        scored = [r.latest_turn.score for r in records
                  if r.latest_turn and r.latest_turn.score is not None]
        overall_score = round(sum(scored) / len(scored), 1) if scored else 0.0

        # ----- build the synthesis prompt ----- #
        lang = LANG_NAME.get(session.language, "English")
        system = self._system_prompt(lang, session.domain)
        user = self._summary_payload(records, overall_score, lang)

        raw_text = self._chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}]
        )
        data = _extract_json(raw_text) or {}

        per_q = self._build_per_question(records, data)

        return InterviewFeedback(
            overall_summary=str(data.get("overall_summary", "")).strip(),
            overall_score=overall_score,
            strengths=list(data.get("strengths", []) or []),
            improvement_areas=list(data.get("improvement_areas", []) or []),
            per_question=per_q,
            spoken_summary=str(data.get("spoken_summary", "")).strip(),
            raw=data,
        )

    # ------------------------------------------------------------------ #
    def _system_prompt(self, lang: str, domain: str) -> str:
        return f"""You are a senior interviewer writing the end-of-interview \
feedback for a candidate who just finished a "{domain}" screening.

You are given, for each question: the question, the candidate's answer(s), and \
the interviewer's private notes (a 0-5 score and which key points were covered \
or missed). The interview is OVER, so you may now be direct and specific about \
gaps — this document IS the feedback.

Write constructive, honest, and encouraging feedback. Be specific: refer to \
what they actually said. Do not invent facts that aren't supported by the notes.

LANGUAGE: write every prose field (overall_summary, strengths, \
improvement_areas, the per-question strengths/improvements, and spoken_summary) \
in {lang}. Keep the JSON keys exactly as specified, in English.

Return ONLY a JSON object (no markdown, no backticks) with this shape:
{{
  "overall_summary": "<2-4 sentence overall impression, {lang}>",
  "strengths": ["<top strengths across the interview, {lang}>"],
  "improvement_areas": ["<top 3 concrete things to work on, {lang}>"],
  "per_question": [
    {{
      "question_id": "<id exactly as given>",
      "strengths": "<1-2 sentences on what was good, {lang}>",
      "improvements": "<1-2 sentences on what to improve, {lang}>"
    }}
  ],
  "spoken_summary": "<a short, warm 2-3 sentence summary to read aloud, {lang}>"
}}"""

    def _summary_payload(self, records: list[QuestionRecord],
                         overall_score: float, lang: str) -> str:
        blocks = []
        for i, r in enumerate(records, 1):
            t = r.latest_turn
            score = t.score if t and t.score is not None else "n/a"
            covered = ", ".join(t.covered_key_points) if t and t.covered_key_points else "(none noted)"
            missing = ", ".join(t.missing_key_points) if t and t.missing_key_points else "(none noted)"
            assessment = t.assessment if t and t.assessment else "(no note)"
            answers = " || ".join(a for a in r.candidate_answers if a) or "(no answer given)"
            blocks.append(
                f"""Q{i} [{r.topic}] (id: {r.question_id})
  question: {r.question_text}
  candidate said: {answers}
  interviewer private score: {score}/5
  key points covered: {covered}
  key points missed: {missing}
  interviewer note: {assessment}"""
            )
        joined = "\n\n".join(blocks)
        return (
            f"Computed overall score (already averaged, use as-is): "
            f"{overall_score}/5\n\n"
            f"Per-question record:\n\n{joined}\n\n"
            f"Now write the feedback JSON in {lang} as specified."
        )

    def _build_per_question(self, records: list[QuestionRecord],
                            data: dict) -> list[PerQuestionFeedback]:
        # Index the model's per-question prose by id, then stitch it back to the
        # authoritative per-question scores we already hold.
        by_id = {
            str(item.get("question_id", "")): item
            for item in (data.get("per_question") or [])
        }
        out: list[PerQuestionFeedback] = []
        for r in records:
            item = by_id.get(r.question_id, {})
            score = r.latest_turn.score if r.latest_turn else None
            out.append(PerQuestionFeedback(
                question_id=r.question_id,
                topic=r.topic,
                score=score,
                strengths=str(item.get("strengths", "")).strip(),
                improvements=str(item.get("improvements", "")).strip(),
            ))
        return out

    # ------------------------------------------------------------------ #
    def _chat(self, messages: list[dict]) -> str:
        base = dict(
            model=self._model,
            messages=messages,
            temperature=self._cfg.llm.temperature,
            max_tokens=max(self._cfg.llm.max_tokens, 900),  # room for the report
        )
        attempts = []
        first = dict(base, response_format={"type": "json_object"})
        if self._cfg.llm.reasoning_effort:
            first["reasoning_effort"] = self._cfg.llm.reasoning_effort
        attempts.append(first)
        attempts.append(base)  # fallback if the model rejects the extras

        last_err = None
        for kwargs in attempts:
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        raise RuntimeError(f"Feedback LLM call failed: {last_err}")


def render_feedback_markdown(fb: InterviewFeedback, *, domain: str,
                             language_name: str) -> str:
    """Render the feedback object as Markdown (used by the CLI and the app)."""
    lines = [
        f"# Interview Feedback — {domain}",
        f"_Language: {language_name}_",
        "",
        f"**Overall readiness score: {fb.overall_score} / 5**",
        "",
        "## Overall",
        fb.overall_summary or "_(none)_",
        "",
    ]
    if fb.strengths:
        lines.append("## What you did well")
        lines += [f"- {s}" for s in fb.strengths]
        lines.append("")
    if fb.improvement_areas:
        lines.append("## Top things to improve")
        lines += [f"- {s}" for s in fb.improvement_areas]
        lines.append("")
    if fb.per_question:
        lines.append("## Question by question")
        for i, pq in enumerate(fb.per_question, 1):
            score = f"{pq.score}/5" if pq.score is not None else "n/a"
            lines.append(f"### Q{i} — {pq.topic}  ({score})")
            if pq.strengths:
                lines.append(f"- **Strength:** {pq.strengths}")
            if pq.improvements:
                lines.append(f"- **Improve:** {pq.improvements}")
            lines.append("")
    return "\n".join(lines)
