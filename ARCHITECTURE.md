# Architecture Note — Voice Interview Agent

This note explains the three decisions the brief asks about: **retrieval
design**, **keeping the LLM an interviewer (grounded but non-leaking)**, and
**latency**. The guiding principle throughout is *put determinism in code and
judgement in the model* — the orchestrator owns control flow, the model owns
natural language.



## 1. Retrieval design

**Storage & "chunking."** The reference set is a flat YAML list
(`data/questions.yaml`). Each interview question is exactly one record — `id`,
`topic`, `difficulty`, the canonical `question`, a **private** `ideal_answer`
(bullet rubric), and a few safe `follow_up_hints`. The natural chunk *is the
question*: each one is a self-contained unit of grounding, so there is no
arbitrary token-window splitting to do. Adding or editing a question is a pure
data edit — no code references ids directly; the orchestrator just walks the
list — which satisfies the "update without code changes" requirement.

**Matching the right reference.** The interview is **agent-driven**: the agent
decides what to ask next, so the dominant retrieval operation each turn is *"give
me the reference for the question I am currently on"*. That is an **exact lookup
by id** — O(1), deterministic, and immune to the false-match and embedding-drift
failure modes that semantic search can introduce. With ~10 questions, this exact
path is not a shortcut; it is the *correct* primary design, and it is what runs
every turn.

On top of that I build a **semantic index** for flexibility and scale-readiness:
multilingual sentence embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) over
`"topic. question"`, with cosine similarity computed as a **brute-force numpy
dot-product** over the normalized matrix. This lets free-form text map to the
nearest canonical question (e.g. resuming "can we go back to the caching one?",
or routing a candidate-initiated question), and it makes growing the bank to
hundreds of questions a config change rather than a rewrite. If the embedding
model can't load, the store transparently falls back to **BM25** lexical
matching, so grounding never breaks.

**Why no vector DB (FAISS/Pinecone).** For *N* in the tens, a numpy matrix
multiply beats any index — there is no build/query overhead and nothing to
operate. A vector database earns its complexity at ~10⁴+ vectors, not 10.
Reaching for Pinecone here would be over-engineering, and choosing *not* to is a
deliberate judgement, not an omission.

## 2. Keeping the LLM an interviewer (grounded, not leaking)

Three independent layers, so no single failure leaks the answer or loses the
thread:

**Layer 1 — Prompt.** The `ideal_answer` is handed to the model as a *private
rubric* under an explicit `INTERNAL — NEVER REVEAL` header. The system prompt
forbids quoting or closely paraphrasing it, and tells the model that if a
candidate is stuck it must *nudge* toward the missing idea (a hint or a leading
sub-question) rather than hand over the answer. The provided `follow_up_hints`
exist precisely so guidance can be helpful without being a giveaway.

**Layer 2 — Structured output (the real guarantee).** Every turn the model
returns JSON that **separates private judgement from speech**: `assessment`,
`score` (0–5), `covered_key_points`, and `missing_key_points` are private; only
the single `spoken_response` field is ever shown or sent to TTS. Because the app
*physically forwards only that one field*, a model that "thinks out loud" about
the rubric in its private fields still cannot leak through the channel the
candidate hears. Prompts are soft constraints; this output contract is the hard
one. (A tolerant parser strips stray fences/prose and falls back gracefully so a
malformed response never crashes the turn.)

**Layer 3 — Orchestration owns control.** The model *requests* an action
(`advance` / `follow_up` / `hint` / `clarify`); the **orchestrator decides and
enforces**. It owns the question pointer and a per-question follow-up budget
(default 2). When the budget is spent, the turn is sent in `must_advance` mode
and the code forces `action = advance` regardless of what the model returns. So
the model can never skip a question, invent new topics, or loop forever — it
stays on script because the script isn't its to change.

**Grounding is ephemeral.** The reference for the current question is injected
fresh each turn and **never stored** in the conversation history fed back to the
model. This keeps the context small (cheaper, faster) and removes any chance of
a past turn's rubric bleeding into a later answer.

**Feedback.** At the end, one synthesis call turns the per-question records into
a structured report (overall summary, strengths, top-3 improvements,
per-question notes) in the interview language. The **numeric overall score is
averaged in code** from the per-turn scores — the model writes the prose, the
code owns the number, so the score can't drift with the model's mood.

## 3. Latency

Per turn, time is spent in four places: **STT → retrieval → LLM → TTS**.
Retrieval is effectively free (an id lookup, or one small matrix multiply).
Speech and the LLM dominate. Concrete choices already made to keep a turn
responsive:

- **STT:** `whisper-large-v3-turbo` on Groq runs at ~216× real-time, so a normal
  spoken answer transcribes in a fraction of a second.
- **TTS:** `eleven_flash_v2_5` (~75 ms model latency) is chosen specifically
  because the brief flags responsiveness; the higher-quality `multilingual_v2`
  is a one-line swap when quality matters more than speed.
- **LLM:** Groq inference is fast; `reasoning_effort=low`, a bounded
  `max_tokens`, and short spoken replies (1–3 sentences) keep generation brief.
  The **grounding** and small history keep the prompt short, which
  directly reduces time-to-first-token.

Currently, my app follows a record → process → respond approach. To make it feel more like a real conversation, I'd stream each stage instead of waiting for the previous one to finish. While the user is speaking, speech-to-text would run continuously. As soon as the AI generates the first sentence, I'd start converting it to speech and play it immediately. This reduces perceived latency because speech recognition, AI response generation, and text-to-speech all happen simultaneously. I'd also use a faster model, keep API connections warm, and support interruptions so users can speak naturally without waiting for the AI to finish


