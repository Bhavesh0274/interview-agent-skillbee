# 🎙️ Voice Interview Agent

A voice-based mock-interview agent. A candidate **speaks** with it; it conducts a
realistic interview — asking questions, listening, asking natural follow-ups,
guiding when an answer is weak (without leaking the ideal answer), and producing
**structured feedback** at the end. It is grounded in a fixed reference Q&A set so
its judgement stays consistent, while an LLM makes it behave like a real,
adaptive interviewer.

**Pipeline:** `🎤 speech → STT → grounding (reference Q&A) → LLM interviewer → TTS → 🔊 speech` → end-of-interview feedback.

| Stage | Provider / model | Why |
|---|---|---|
| Speech-to-text | Groq `whisper-large-v3-turbo` | One multilingual model for en/hi/de, ~216× real-time |
| Interviewer LLM | Groq `openai/gpt-oss-120b` | Fast on Groq, reasoning + JSON mode; current (post-Llama-3.3 deprecation) |
| Text-to-speech | **Edge TTS** (free, default) · ElevenLabs (optional upgrade) | Free, no key, native en/hi/de voices; ElevenLabs `eleven_flash_v2_5` for extra polish |
| Embeddings (optional) | `paraphrase-multilingual-MiniLM-L12-v2` | Tiny multilingual model for semantic navigation/scale |



## Quick start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd interview-agent
python -m venv .venv && source .venv/bin/activate    # optional but recommended
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
# then open .env and paste your Groq key:
#   GROQ_API_KEY=...          (free at https://console.groq.com/keys)
```

That's the **only** key you need — the default TTS (Edge) is free and keyless.
`ELEVENLABS_API_KEY` is optional and only needed if you switch to ElevenLabs in
`config.yaml` for higher-quality voices. `.env` is git-ignored, so keys never
get committed.

### 3a. Run the voice app (the main experience)

```bash
streamlit run app.py
```

Then in the browser: you'll land on a **"Choose your interview language"**
screen — pick **English / हिन्दी / Deutsch** and the interview starts in that
language. Record your answer, stop the recording, and the agent transcribes it,
replies by voice, and follows up. Use **Start over** in the sidebar to pick a
different language, or **End interview & get feedback** for the scored report.


## How to customise (no core-code changes)

Everything you'd normally want to change lives in two files:

**`data/questions.yaml` — the reference Q&A set.** Add, edit, or remove a
question by editing this YAML; the core logic re-reads it on the next run. Each
entry is:

```yaml
- id: q11_my_new_question        # unique id
  topic: Caching                 # short label (shown in feedback)
  difficulty: medium             # easy | medium | hard
  question: "Canonical English question — the agent rephrases it naturally."
  ideal_answer: |                # PRIVATE rubric — never read to the candidate
    - key point one
    - key point two
  follow_up_hints:               # safe nudges the agent may use if they're stuck
    - "What happens under high concurrency?"
```

No code references question ids directly — the orchestrator just walks the list
in order — so adding a question is purely a data edit.

**`config.yaml` — behaviour & models.** Language, follow-up budget, which
STT/LLM/TTS models to use, voices, retrieval mode. For example:

- `language: en | hi | de` — interview language (also switchable live in the app).
- `interview.max_follow_ups_per_question` — how hard the agent probes before moving on.
- `llm.model` — swap `openai/gpt-oss-120b` → `openai/gpt-oss-20b` for lower latency.
- `tts.provider` — `edge` (free, default) or `elevenlabs` (needs a key, higher quality).
- `retrieval.mode: id | semantic` — see ARCHITECTURE.

---

## Project layout

```
interview-agent/
├── app.py                  # Streamlit voice UI (the only place audio/UI lives)
├── cli.py                  # text-mode runner (no mic) for testing/debugging
├── config.yaml             # behaviour + model config (no secrets)
├── .env.example            # template for API keys -> copy to .env
├── requirements.txt
├── data/
│   └── questions.yaml      # the reference Q&A set (edit freely)
├── src/
│   ├── config.py           # loads config.yaml + .env into a typed AppConfig
│   ├── reference_store.py  # grounding layer: id lookup + semantic/BM25 search
│   ├── stt.py              # Groq Whisper speech-to-text
│   ├── tts.py              # ElevenLabs text-to-speech
│   ├── llm.py              # the interviewer brain (grounded, non-leaking, JSON)
│   ├── interview.py        # orchestrator / state machine (owns the script)
│   └── feedback.py         # end-of-interview structured report
├── ARCHITECTURE.md         # 1-2 page design note (retrieval, non-leak, latency)
└── ENGINEERING_NOTES.md    # stage-by-stage build + tradeoff walkthrough
```



  feel real-time.
- **Lightweight install option:** the interview runs on the exact-lookup
  retrieval path, so if you set `retrieval.mode: id` you can skip
  `sentence-transformers`/`torch` entirely and still run everything.
- Model names are current as of June 2026 and are all one-line config swaps if a
  provider deprecates one.
