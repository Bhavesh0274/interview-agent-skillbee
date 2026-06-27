# Engineering Notes — How This Was Built, Stage by Stage

This is the companion to `ARCHITECTURE.md`. Where the architecture note is the
*what and why* at a system level, this document walks through **every stage of
the code in the order it was built**, the **approach taken**, the **tradeoffs**,
and specifically **why each model/library was chosen over the alternatives**. The
last section is a set of **anticipated interviewer questions with sharp answers**
— the things a senior engineer would actually probe.

> One sentence to hang everything on: **determinism lives in code, judgement
> lives in the model.** Almost every design choice below is an application of
> that line. The orchestrator owns control flow and the numeric score; the LLM
> owns natural language and qualitative judgement. Whenever those two could
> conflict, code wins.

---

## Stage 0 — Framing the problem before writing code

The brief is a 5-stage pipeline: **voice in → grounding → LLM interviewer →
voice out → feedback.** Two requirements shaped the whole structure:

1. *"The reference Q&A set must be easy to update without code changes."* → the
   Q&A lives in **data** (`questions.yaml`), and **nothing in the code
   references a question by id**. The orchestrator just walks the list. Adding a
   question is a data edit.
2. *"Configurable in English, Hindi, German."* → language is a single config
   value threaded through STT (hint), the LLM (the language it writes/speaks),
   and TTS (voice language). I chose **one multilingual model per stage** rather
   than three per-language pipelines, so switching language is one switch, not a
   re-wire.

I also made an early structural decision that paid off repeatedly: **isolate
voice at the edge.** The interview core (`llm.py`, `interview.py`,
`feedback.py`) is pure text-in/text-out and has no idea audio exists. Speech I/O
lives only in `app.py`. That is what let me unit-test the entire control flow
offline with a mock LLM — no mic, no API bill, no flakiness.

---

## Stage 1 — Config & secrets (`config.py`, `config.yaml`, `.env`)

**Problem.** Keep "things you'd tune" separate from "things you must not commit,"
and give the rest of the code typed objects instead of raw dicts.

**Approach.** `config.yaml` holds *behaviour* (language, models, follow-up
budget, voices, retrieval mode) and is versioned in git. `.env` holds *secrets*
(API keys) and is git-ignored. `config.py` loads both into a typed `AppConfig`
(nested dataclasses: `STTConfig`, `LLMConfig`, `TTSConfig`, `RetrievalConfig`,
`InterviewConfig`). It also exposes `require_keys()` for **fail-fast** behaviour —
if a needed key is missing you get one clear error at startup, not a confusing
500 mid-interview.

**Tradeoffs.**
- *Dataclasses vs. passing dicts around.* Dataclasses cost a few lines up front
  but give autocomplete, typo-safety, and a single source of truth for defaults.
  Worth it.
- *YAML vs. env-only config.* I could have stuffed everything in env vars
  (12-factor purist style). YAML is friendlier for the *behavioural* knobs a
  reviewer wants to read and edit (especially nested things like per-language
  voices), while still keeping secrets in env. So: behaviour in YAML, secrets in
  env — the split that gives the best of both.

---

## Stage 2 — Grounding / the reference store (`reference_store.py`, `questions.yaml`)

**Problem.** Store the reference Q&A and, each turn, hand the agent the right
reference — "what is being asked and what a strong answer contains."

**Approach.** `questions.yaml` is a flat list; each question is one record with a
**private** `ideal_answer` (bullet rubric) and safe `follow_up_hints`. The store
exposes two retrieval paths:

- **`get(id)` — exact lookup (PRIMARY).** The interview is agent-driven, so the
  real question each turn is "give me the reference for the question I'm on."
  That's an O(1) dict lookup — deterministic, no false matches.
- **`match(query, top_k)` — semantic search (SECONDARY).** Multilingual
  embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) over `"topic. question"`,
  cosine via a brute-force numpy dot-product on the normalized matrix. Used for
  free-form navigation and to be ready to scale the bank. If the embedding model
  can't load, it **falls back to BM25** (lexical), so the store always works.

**Why this and not "just do RAG with a vector DB."** This is the most important
judgement call in the project. The instinct on hearing "retrieve reference Q&A"
is FAISS/Pinecone + embeddings. But:
- The dominant operation is **exact** (the agent already knows which question it
  asked), so embeddings aren't even on the critical path — id lookup is.
- With **N ≈ 10**, a numpy matmul is *faster* than any vector index (no
  build/query overhead) and far simpler to operate. A vector DB earns its keep
  around **10⁴+ vectors**, not ten.
- So I built the semantic index for *flexibility and scale-readiness*, but kept
  it secondary and deliberately avoided a vector DB. **Knowing when *not* to use
  the heavy tool is the signal here.**

**Why `paraphrase-multilingual-MiniLM-L12-v2`.** It's tiny (fast, low memory),
runs locally (no per-call cost or network hop), and — critically — it's
**multilingual**, so the same index works for en/hi/de. I don't need a
state-of-the-art retriever to disambiguate among ten questions; I need a small,
fast, multilingual one.

**Tradeoffs.**
- *Embeddings pull in torch (heavy).* Mitigated: the interview runs on the id
  path, so `retrieval.mode: id` lets you skip sentence-transformers/torch
  entirely. The BM25 fallback covers free-form matching without torch too.
- *"topic. question" as the embedded text.* Including the topic gives the
  embedding a little extra signal and disambiguates terse questions. The ideal
  answer is deliberately **not** embedded — it's private and embedding it would
  risk surfacing it via a match.

---

## Stage 3 — Speech-to-text (`stt.py`)

**Problem.** Turn recorded audio into text, in any of the three languages.

**Approach.** A tiny `SpeechToText` Protocol with a `GroqSTT` implementation
calling Groq's hosted `whisper-large-v3-turbo`. We pass `response_format="text"`,
`temperature=0.0` (deterministic transcription), and a **language hint** for
hi/de.

**Why `whisper-large-v3-turbo` on Groq.**
- **Multilingual in one model** — covers en/hi/de, so I don't juggle one ASR per
  language (directly serves the multilingual requirement).
- **~216× real-time on Groq** — the first leg of the pipeline is well under
  budget for a responsive turn.
- **Managed API** — no GPU, no model hosting; exactly what the brief invites
  ("use managed APIs freely").

**Why pass a language hint at all if Whisper auto-detects?** Auto-detect is
excellent on a clean sentence but can occasionally mis-detect a *short, accented*
utterance (common in a mock interview). The hint removes that failure mode at
zero cost, and we already know the language from config — so why gamble.

**Tradeoff.** Hard-coding `language` per call vs. always auto-detecting: I gate
the hint behind `pass_language_hint` so it's configurable, and only pass it for
the three supported languages.

---

## Stage 4 — Text-to-speech (`tts.py`)

**Problem.** Speak the interviewer's reply, in the interview language.

**Approach.** A `TextToSpeech` Protocol with an `ElevenLabsTTS` implementation.
`convert()` returns an iterator of byte chunks which we join into one blob for
Streamlit to play. We pin pronunciation with an ISO `language_code`.

**Why ElevenLabs `eleven_flash_v2_5`.**
- **~75 ms model latency** — the brief explicitly flags responsiveness, so the
  *fast* model is the right default for a voice agent.
- **32 languages incl. en/hi/de**, and one multilingual voice
  (`JBFqnCBsd6RMkjVDRZzb`, "George") works across all three, so we don't manage a
  voice per language unless we want native-sounding ones (configurable).
- **`eleven_multilingual_v2`** is a one-line swap when quality matters more than
  latency.

**A correctness detail I handled.** `language_code` is only accepted by the
low-latency models (flash/turbo v2.5, v3); `multilingual_v2` rejects it. So
`synthesize()` **only passes `language_code` for models that support it** —
otherwise the documented quality-swap would crash. Small thing, but it's the
difference between "swap one config line" working and not.

**Tradeoff (buffer vs. stream).** I buffer the chunks into one blob because
Streamlit plays a single audio widget per turn and turn-based UX is the right
prototype scope. For a production real-time feel you'd forward the chunks to the
player as they arrive (noted in code and in ARCHITECTURE's latency section).

---

## Stage 5 — The interviewer brain (`llm.py`)

**Problem.** Make an LLM behave like a real interviewer — grounded in the
reference, asking natural follow-ups, guiding without giving the answer away,
staying on track.

**Approach.** `InterviewerLLM` has two entry points:
- `opening(...)` — greet + ask Q1, phrased naturally in the target language.
- `respond(...)` — evaluate the latest answer and return a structured
  `InterviewerTurn` (the spoken line + private judgement + a requested `action`).

The **non-leak design is three layers** (full detail in ARCHITECTURE §2):
1. **Prompt:** the ideal answer is a PRIVATE rubric under an `INTERNAL — NEVER
   REVEAL` header; guidance must nudge, not hand over.
2. **Structured output (the real guarantee):** JSON separates private fields
   (`assessment`, `score`, `covered/missing_key_points`) from the single
   `spoken_response`. **Only `spoken_response` is ever surfaced or spoken**, so
   even a misbehaving model can't leak through the channel the candidate hears.
3. **Modes:** the orchestrator passes `mode = evaluate` or `must_advance`; in
   `must_advance` the code forces `action = advance` after parsing.

**Robustness details that matter.**
- `_chat()` requests **JSON mode + `reasoning_effort=low`**, and **falls back to
  dropping those params** if a swapped-in model rejects them. So changing
  `llm.model` never breaks the call.
- `_extract_json()` is a **tolerant parser**: strips ```` ```json ```` fences,
  tries `json.loads`, then does balanced-brace extraction, then a safe fallback
  turn. A malformed model response degrades to a sensible follow-up instead of a
  crash.

**Why `openai/gpt-oss-120b` on Groq — and why this is a talking point.** Groq
**deprecated `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` on 17 Jun
2026**. I checked current provider docs rather than trusting training-data
muscle memory, and `gpt-oss-120b` is Groq's recommended successor for the 70B
tier. It's reasoning-capable (good for judging answer quality and phrasing
follow-ups), supports **JSON mode** (which my anti-leak contract relies on), and
is fast on Groq. `gpt-oss-20b` is the lower-latency swap. **The fact that the
model is a single config line — and that I verified currency instead of shipping
a deprecated default — is itself the engineering signal.**

**Tradeoff (temperature 0.4).** Low enough for consistent judgement and stable
JSON, high enough that the spoken lines don't sound robotic. Pure 0.0 would make
follow-ups feel canned; high temp risks rubric-scoring drift and malformed JSON.

---

## Stage 6 — Orchestrator / state machine (`interview.py`)

**Problem.** Keep the agent on-script: the right number of questions, a bounded
number of follow-ups, no wandering, no infinite loops.

**Approach.** `InterviewSession` **owns the control state** the model is not
allowed to: the question pointer (`idx`), the per-question follow-up budget, the
phase, and two transcripts. Each turn it:
1. injects the reference for the current question **fresh** (never stored),
2. tells the LLM how many follow-ups remain and whether this is the last
   question, and in which `mode`,
3. takes the model's requested `action` and **decides**: advance the pointer (and
   detect "finished") or spend one follow-up. When the budget is gone it sends
   `must_advance` and the LLM layer forces `advance`.

**Two transcripts, on purpose.** `history` (clean spoken turns) is what's fed
back to the LLM for context; `transcript` is for display. **Grounding/rubric is
in neither.** That keeps the model's context small (cheaper + faster) and means a
past turn's rubric can't bleed into a later answer.

**Why the orchestrator owns the pointer instead of the LLM.** If you let the
model self-manage "what question are we on," it *will* eventually skip one, loop,
or invent a topic — and you can't unit-test that. Putting the pointer and budget
in code makes the behaviour deterministic and testable. I validated it with a
mock LLM: with a 2-follow-up cap it probes twice then is **forced** to advance,
walks Q1→Q2→Q3, and finishes on the last advance. The model never had a chance to
go off the rails because the rails aren't its to move.

---

## Stage 7 — End-of-interview feedback (`feedback.py`)

**Problem.** Produce structured feedback: what they did well, what to improve.

**Approach.** The orchestrator already accumulated, per question, the answers and
the interviewer's **last private judgement** (score + covered/missing points). So
feedback is **one synthesis call**, not one-call-per-question: cheaper, faster,
and it lets the model write a coherent whole-interview narrative. It returns
structured JSON (overall summary, strengths, top-3 improvements, per-question
notes, plus a short spoken summary for optional TTS), written in the interview
language.

**The key tradeoff: who owns the number.** The **overall score is averaged in
code** from the per-turn scores; the model only writes prose. Models are
inconsistent and often generous at summarisation time — so the qualitative report
is the model's, but the headline number is deterministic and reproducible. (Same
principle as everywhere else: judgement to the model, arithmetic to the code.) I
also stitch the model's per-question prose back onto the **authoritative**
per-question scores by id, so the numbers in the report can't drift from what
was actually assessed during the interview.

---

## Stage 8 — The voice shell (`app.py`, Streamlit)

**Problem.** Wire mic → STT → session → TTS into a UI, and survive Streamlit's
rerun model.

**Approach.** `app.py` is the *only* component that touches audio/UI. Per turn:
record → STT → `session.submit_answer` → TTS → render transcript and autoplay the
reply. Sidebar selects language (locked once started) and offers Start/Reset.
"End interview" jumps to feedback any time.

**The rerun gotcha (a bug I designed around up front).** Streamlit reruns the
whole script on every interaction, and `st.audio_input` returns the **same**
recording object on each rerun until a new one is recorded. Naively transcribing
"whatever it returns" reprocesses the same audio repeatedly (extra latency, extra
cost, duplicate turns). The fix: remember the processed recording's **`file_id`**
in `st.session_state` and only act when a genuinely new id appears. Everything
that must survive a rerun (the session object, transcript, last audio blob,
feedback) lives in `session_state`; clients are built with `@st.cache_resource`
so we don't reconnect every rerun.

**Graceful degradation.** Voice is best-effort around a text core: if TTS fails,
show the text; if STT fails, keep the turn and ask the candidate to repeat. A
transient API hiccup never hard-crashes the interview.

---

## Stage 9 — The CLI (`cli.py`)

**Problem.** Try and debug the logic without a microphone.

**Approach.** Same orchestrator + LLM + feedback, answers typed instead of
spoken. This exists because (a) it makes the core trivially testable and
reviewable in seconds, and (b) it proves the voice/logic separation is real — the
exact same session code runs with zero audio involved.

---

## Cross-cutting decisions (the ones worth defending)

- **Model currency over memory.** I verified provider docs and used current
  models because Groq deprecated the Llama 3.x defaults on 17 Jun 2026. Shipping
  a deprecated model name is the kind of thing that quietly breaks a demo.
- **One multilingual model per stage**, not three per-language stacks — fewer
  moving parts, one config switch for en/hi/de.
- **Structured output as a safety mechanism, not just formatting.** Surfacing a
  single field is what makes "don't leak the answer" a *guarantee* rather than a
  *request*.
- **Ephemeral grounding** — small context, lower latency, no cross-turn leak.
- **No premature infra** — no vector DB at N≈10, no streaming complexity in a
  turn-based prototype (but the path to both is written down).
- **Provider interfaces (Protocols)** for STT/TTS/LLM so swapping Deepgram/Azure
  later is a one-class change, not a refactor.

---

## Anticipated interviewer questions (with answers)

**Retrieval**

- **"Why not just use a vector database / proper RAG?"** Because the dominant
  operation is exact, not fuzzy — the agent already knows which question it
  asked, so it's an O(1) id lookup, and a vector DB only pays off around 10k+
  vectors. I still built a multilingual semantic index (numpy cosine) for
  free-form navigation and scale-readiness, with a BM25 fallback. Reaching for
  Pinecone at N≈10 would be over-engineering; choosing not to is the point.
- **"How do you chunk the reference answers?"** Each question is its own chunk —
  the natural unit of grounding — so there's no token-window splitting. That's
  also why "update without code changes" is just a YAML edit.
- **"What if the candidate asks their own question / goes off-script?"** The
  semantic `match()` path maps free-form text to the nearest canonical question,
  and the orchestrator still owns the pointer, so we can route or politely steer
  back without losing our place.
- **"Why embed `topic. question` and not the ideal answer?"** Topic adds signal
  and disambiguates terse questions; the ideal answer is private and embedding it
  would risk surfacing it through a match.

**Keeping it an interviewer / not leaking**

- **"How do you *guarantee* it won't reveal the ideal answer?"** I don't rely on
  the prompt alone. The model returns JSON that separates private judgement from
  the single `spoken_response`, and the app forwards only `spoken_response` to
  the candidate/TTS. So even if the model reasons about the rubric, it can't leak
  through the channel they hear. Prompt + output-contract + orchestration =
  three independent layers.
- **"What stops it asking 12 questions, or looping on one?"** The orchestrator,
  not the model, owns the pointer and a per-question follow-up budget; when the
  budget's spent we force `advance`. The model requests; the code decides.
- **"What if the model returns broken JSON?"** A tolerant parser (fence-strip →
  `json.loads` → balanced-brace extraction → safe fallback turn) means a bad
  response degrades to a sensible follow-up, never a crash. And JSON mode makes
  breakage rare in the first place.
- **"Why is the final score computed in code?"** Determinism. Models drift and
  are generous at summarisation; the prose is theirs, the number is the code's
  average of the per-turn scores — reproducible and not mood-dependent.

**Latency**

- **"Where's the time going, and what would you cut first?"** Retrieval is free;
  STT and TTS and the LLM dominate. First win is pipelining instead of buffering:
  stream LLM tokens and start TTS on the first finished sentence while the rest
  generates, and stream STT partials — that overlaps the slow stages. Then
  `gpt-oss-20b`, warm clients (already cached), short replies, and barge-in.
- **"Why flash TTS and low reasoning effort?"** The brief flags responsiveness,
  so the fast TTS model and `reasoning_effort=low` are deliberate latency
  choices; both have one-line higher-quality swaps when latency isn't the
  priority.

**Models & stack**

- **"Why these specific models?"** Whisper-turbo: one multilingual ASR, ~216×
  real-time, managed. gpt-oss-120b: Groq's current successor after the Llama 3.x
  deprecation, reasoning + JSON mode, fast. flash_v2_5: ~75 ms, 32 languages.
  Each is a single config line to swap.
- **"What happens when one of these models gets deprecated too?"** Change one
  line in `config.yaml`. The `_chat` fallback already tolerates a model that
  rejects JSON-mode/reasoning params, and the provider sits behind a Protocol, so
  even switching vendor is a one-class change.
- **"Why Streamlit and not a 'real' web app?"** Fastest path to a working voice
  prototype with built-in mic capture and audio playback, and the brief says use
  whatever you're fastest in. The core is UI-agnostic, so a FastAPI + browser
  front-end later reuses `interview.py`/`llm.py`/`feedback.py` untouched.

**General engineering**

- **"How is this testable given it calls LLMs and microphones?"** The interview
  core is pure text-in/text-out with no audio dependency, so control flow runs
  offline against a mock LLM. I verified the follow-up cap, the Q1→Q2→Q3
  progression, the finish condition, and the feedback math (deterministic 4.0
  from per-turn 4s) without a single API call.
- **"What would you do before production?"** Streaming pipeline + barge-in,
  per-question latency/score telemetry, retries/backoff on the providers, native
  voices per language, and a small eval set to track that guidance stays helpful
  without leaking as the prompt evolves.
