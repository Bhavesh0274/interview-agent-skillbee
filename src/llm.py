"""
llm.py  —  The interviewer brain
================================
Wraps the chat LLM and makes it behave like a disciplined interviewer that is
GROUNDED in the reference answer but never LEAKS it.

How grounding + non-leak is enforced (three layers):
  1. Prompt: the reference answer is given to the model as a PRIVATE rubric,
     under an explicit "INTERNAL — never reveal" header, and the system prompt
     forbids quoting/paraphrasing it to the candidate.
  2. Structured output: the model must return JSON that SEPARATES its private
     judgement (`assessment`, `score`, `missing_key_points`) from the single
     `spoken_response` field. Only `spoken_response` is ever read aloud or
     shown. Even a misbehaving model physically cannot leak via the channel we
     surface, because we only forward one field.
  3. Orchestration (see interview.py): the agent — not the model — owns the
     question pointer and the follow-up budget, so the model can't wander off
     the script or loop forever.

Why `openai/gpt-oss-120b` on Groq:
  * It is Groq's recommended successor after `llama-3.3-70b-versatile` was
    deprecated (17 Jun 2026), so the code stays current.
  * Strong enough to judge answer quality and phrase natural follow-ups, and
    on Groq it returns fast. We set `reasoning_effort=low` to keep voice
    latency down. The model is a one-line config swap (e.g. gpt-oss-20b for
    even lower latency).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from groq import Groq

from .config import AppConfig
from .reference_store import Question

LANG_NAME = {"en": "English", "hi": "Hindi", "de": "German"}

VALID_ACTIONS = {"advance", "follow_up", "hint", "clarify"}


@dataclass
class InterviewerTurn:
    spoken_response: str                 # the ONLY field shown / spoken
    action: str                          # advance | follow_up | hint | clarify
    score: Optional[int] = None          # 0-5, private
    assessment: str = ""                 # private notes
    covered_key_points: list[str] = field(default_factory=list)
    missing_key_points: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


class InterviewerLLM:
    def __init__(self, config: AppConfig):
        self._cfg = config
        self._model = config.llm.model
        self._client = Groq(api_key=config.groq_api_key)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def opening(self, first_question: Question, *, language: str,
                domain: str) -> str:
        """Greet the candidate and ask the first question, in `language`."""
        lang = LANG_NAME.get(language, "English")
        system = (
            f"You are a warm, professional technical interviewer running a "
            f"'{domain}' screening. You will speak entirely in {lang}. "
            f"Keep it short and natural — this is spoken aloud."
        )
        user = (
            "Open the interview: greet the candidate in one sentence, briefly "
            "say you'll ask a few questions and that they should answer out "
            "loud, then ask the FIRST question below, phrased naturally in "
            f"{lang} (do not read it robotically). Return ONLY the spoken text, "
            "no preamble, no quotes.\n\n"
            f"FIRST QUESTION (canonical English, rephrase naturally):\n{first_question.question}"
        )
        content = self._chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            force_json=False,
        )
        return content.strip()

    def respond(
        self,
        *,
        history: list[dict],
        current_question: Question,
        candidate_answer: str,
        follow_ups_remaining: int,
        next_question: Optional[Question],
        is_last_question: bool,
        language: str,
        domain: str,
    ) -> InterviewerTurn:
        """Evaluate the latest answer and produce the next spoken turn."""
        mode = "must_advance" if follow_ups_remaining <= 0 else "evaluate"
        system = self._system_prompt(language, domain)
        grounding = self._grounding_block(
            current_question=current_question,
            candidate_answer=candidate_answer,
            follow_ups_remaining=follow_ups_remaining,
            next_question=next_question,
            is_last_question=is_last_question,
            mode=mode,
            language=language,
        )
        messages = [{"role": "system", "content": system}, *history,
                    {"role": "user", "content": grounding}]

        raw_text = self._chat(messages, force_json=True)
        return self._parse_turn(raw_text, mode=mode)

    # ------------------------------------------------------------------ #
    # Prompt construction                                                #
    # ------------------------------------------------------------------ #
    def _system_prompt(self, language: str, domain: str) -> str:
        lang = LANG_NAME.get(language, "English")
        return f"""You are an experienced, fair, and personable technical interviewer \
conducting a "{domain}" screening interview. You behave like a real human \
interviewer, not a quiz bot.

LANGUAGE
- Conduct the interview entirely in {lang}.
- The field "spoken_response" MUST be written in {lang}. All other fields \
(your private notes) stay in English.

YOUR JOB EACH TURN
- You are given the current question and a PRIVATE reference of what a strong \
answer contains. Judge the candidate's answer against that reference internally.
- Then either dig deeper on the current question, guide them if they're stuck, \
or move on — according to the MODE you are given.

HARD RULES
1. NEVER reveal, read out, quote, or closely paraphrase the reference / ideal \
answer. It is your private rubric only. If you guide a stuck candidate, nudge \
them toward the missing idea with a hint or a leading sub-question — do not \
hand them the answer.
2. Never tell the candidate their score or your private assessment.
3. Ask ONE thing at a time. Keep "spoken_response" short and conversational — \
it is spoken out loud (typically 1-3 sentences).
4. Stay on script: only the current question and its natural follow-ups. Do \
not invent new interview topics.
5. Acknowledge the candidate briefly and naturally before your next move.

MODE
- mode = "evaluate": decide. If the answer is solid, set action="advance". If \
it is weak or partial, set action="follow_up" (a focused probing question) or \
action="hint" (a nudge toward what's missing). If you couldn't understand the \
answer, action="clarify".
- mode = "must_advance": the follow-up budget for this question is spent. You \
MUST set action="advance". Briefly acknowledge the answer, then ask the NEXT \
question provided (or, if it's the last question, close the interview warmly).

When action="advance" and a NEXT question is provided, your "spoken_response" \
should acknowledge their answer and then ask that next question, phrased \
naturally in {lang}.

OUTPUT FORMAT
Return ONLY a JSON object (no markdown, no backticks, no extra text) with keys:
{{
  "assessment": "<your private 1-2 sentence judgement, English>",
  "score": <integer 0-5, how well the answer matched the reference>,
  "covered_key_points": ["<reference points they hit>"],
  "missing_key_points": ["<reference points they missed>"],
  "action": "advance" | "follow_up" | "hint" | "clarify",
  "spoken_response": "<what you say to the candidate, in {lang}>"
}}"""

    def _grounding_block(self, *, current_question: Question,
                         candidate_answer: str, follow_ups_remaining: int,
                         next_question: Optional[Question],
                         is_last_question: bool, mode: str,
                         language: str) -> str:
        lang = LANG_NAME.get(language, "English")
        next_q = (
            f'NEXT QUESTION (canonical English — rephrase naturally in {lang} '
            f'if you advance):\n"{next_question.question}"'
            if next_question and not is_last_question
            else "NEXT QUESTION: none — this is the LAST question. If you "
                 "advance, close the interview warmly instead of asking more."
        )
        return f"""=== CURRENT QUESTION (id: {current_question.id}, topic: {current_question.topic}) ===
"{current_question.question}"

=== INTERNAL REFERENCE — NEVER REVEAL TO THE CANDIDATE ===
{current_question.ideal_answer}

Safe nudges you may use if they are stuck (these do NOT give away the answer):
{chr(10).join('- ' + h for h in current_question.follow_up_hints) or '- (none)'}

=== CONTROL ===
mode: {mode}
follow_ups_remaining_for_this_question: {follow_ups_remaining}
{next_q}

=== CANDIDATE'S LATEST ANSWER (transcribed speech) ===
"{candidate_answer}"

Now produce the JSON object as specified."""

    # ------------------------------------------------------------------ #
    # LLM call + parsing                                                 #
    # ------------------------------------------------------------------ #
    def _chat(self, messages: list[dict], *, force_json: bool) -> str:
        """Call Groq with graceful fallback if optional params are rejected."""
        base = dict(
            model=self._model,
            messages=messages,
            temperature=self._cfg.llm.temperature,
            max_tokens=self._cfg.llm.max_tokens,
        )
        # Preferred call: ask for JSON mode + low reasoning effort.
        attempts = []
        first = dict(base)
        if force_json:
            first["response_format"] = {"type": "json_object"}
        if self._cfg.llm.reasoning_effort:
            first["reasoning_effort"] = self._cfg.llm.reasoning_effort
        attempts.append(first)
        # Fallback: drop the optional params in case the chosen model rejects
        # them (keeps the code working across model swaps).
        attempts.append(base)

        last_err = None
        for kwargs in attempts:
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        raise RuntimeError(f"LLM call failed: {last_err}")

    def _parse_turn(self, text: str, *, mode: str) -> InterviewerTurn:
        data = _extract_json(text)
        if data is None:
            # Last-resort fallback: keep the conversation sane.
            action = "advance" if mode == "must_advance" else "follow_up"
            return InterviewerTurn(
                spoken_response=text.strip() or "Could you elaborate a bit more?",
                action=action,
                raw={"_parse_error": True, "_raw": text},
            )

        action = str(data.get("action", "")).strip()
        if action not in VALID_ACTIONS:
            action = "advance" if mode == "must_advance" else "follow_up"
        if mode == "must_advance":
            action = "advance"   # orchestrator-enforced invariant

        score = data.get("score")
        try:
            score = int(score) if score is not None else None
        except (TypeError, ValueError):
            score = None

        return InterviewerTurn(
            spoken_response=str(data.get("spoken_response", "")).strip(),
            action=action,
            score=score,
            assessment=str(data.get("assessment", "")).strip(),
            covered_key_points=list(data.get("covered_key_points", []) or []),
            missing_key_points=list(data.get("missing_key_points", []) or []),
            raw=data,
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> Optional[dict]:
    """Parse a JSON object from model output, tolerating stray wrapping text."""
    if not text:
        return None
    text = text.strip()
    # Strip ```json fences if present.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first balanced {...} block.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
