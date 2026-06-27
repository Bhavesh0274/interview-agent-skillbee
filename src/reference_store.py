"""
reference_store.py
==================
The grounding layer. Loads the reference Q&A set and serves it to the agent.

Why this shape?
---------------
The interview is AGENT-DRIVEN: the agent chooses which question to ask next,
so the dominant retrieval operation is "give me the reference for question X".
That is an exact lookup by id — O(1), deterministic, and immune to the
embedding drift / false-match problems that semantic search can introduce.
With only ~10 questions, that exact path is genuinely the right primary design.

We ALSO build a semantic index (multilingual embeddings + brute-force cosine)
so that free-form text can be matched to the nearest canonical question. That
is useful for (a) flexible navigation ("can we go back to the caching one?")
and (b) scaling the bank to hundreds of questions later. We deliberately do
NOT use a vector DB (FAISS/Pinecone): for N in the tens, a numpy dot-product
over the matrix is faster than any index (no build/query overhead) and far
simpler to operate. A vector DB earns its keep at ~10k+ vectors, not 10.

If the embedding model can't be loaded (missing dependency / constrained
host), we fall back to a BM25 lexical match so the store ALWAYS works.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


# --------------------------------------------------------------------------- #
# Data model                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Question:
    id: str
    topic: str
    difficulty: str
    question: str
    ideal_answer: str
    follow_up_hints: list[str]


@dataclass
class MatchResult:
    question: Question
    score: float
    method: str   # "semantic" | "lexical"


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# Store                                                                        #
# --------------------------------------------------------------------------- #
class ReferenceStore:
    def __init__(
        self,
        dataset_path: str | Path,
        retrieval_mode: str = "semantic",
        embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.dataset_path = Path(dataset_path)
        self.retrieval_mode = retrieval_mode
        self.embedding_model_name = embedding_model

        data = yaml.safe_load(self.dataset_path.read_text(encoding="utf-8"))
        self.domain: str = data.get("domain", "Interview")
        self.default_language: str = data.get("default_language", "en")

        self._questions: list[Question] = [
            Question(
                id=str(q["id"]),
                topic=q.get("topic", ""),
                difficulty=q.get("difficulty", "medium"),
                question=q["question"].strip(),
                ideal_answer=q.get("ideal_answer", "").strip(),
                follow_up_hints=list(q.get("follow_up_hints", []) or []),
            )
            for q in data["questions"]
        ]
        if not self._questions:
            raise ValueError(f"No questions found in {self.dataset_path}")

        self._by_id: dict[str, Question] = {q.id: q for q in self._questions}

        # Lazily-initialised semantic index (built on first use).
        self._embedder = None
        self._embeddings = None            # np.ndarray [N, dim], L2-normalised
        self._semantic_ready: Optional[bool] = None  # None=untried, False=failed

    # ----- exact retrieval (PRIMARY path) ---------------------------------- #
    @property
    def questions(self) -> list[Question]:
        return list(self._questions)

    def __len__(self) -> int:
        return len(self._questions)

    def get(self, question_id: str) -> Question:
        """Exact lookup by id. This is what the running interview uses."""
        if question_id not in self._by_id:
            raise KeyError(f"No question with id '{question_id}'")
        return self._by_id[question_id]

    def get_by_index(self, index: int) -> Question:
        return self._questions[index]

    # ----- semantic / lexical retrieval (SECONDARY path) ------------------- #
    def match(self, query: str, top_k: int = 1) -> list[MatchResult]:
        """
        Map a free-form string to the nearest canonical question(s).
        Tries embeddings first; falls back to BM25 lexical matching.
        """
        if self.retrieval_mode == "semantic" and self._ensure_semantic():
            return self._semantic_match(query, top_k)
        return self._lexical_match(query, top_k)

    # -- embedding backend --
    def _ensure_semantic(self) -> bool:
        """Build the embedding index once. Return False if unavailable."""
        if self._semantic_ready is not None:
            return self._semantic_ready
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            self._np = np
            self._embedder = SentenceTransformer(self.embedding_model_name)
            # Index on question text + topic so a paraphrase still matches.
            corpus = [f"{q.topic}. {q.question}" for q in self._questions]
            emb = self._embedder.encode(
                corpus, normalize_embeddings=True, convert_to_numpy=True
            )
            self._embeddings = emb
            self._semantic_ready = True
        except Exception as exc:  # missing dep, OOM, download blocked, ...
            print(f"[reference_store] semantic index unavailable "
                  f"({exc.__class__.__name__}: {exc}); using lexical fallback.")
            self._semantic_ready = False
        return self._semantic_ready

    def _semantic_match(self, query: str, top_k: int) -> list[MatchResult]:
        np = self._np
        q_emb = self._embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        # cosine == dot product because everything is L2-normalised
        scores = self._embeddings @ q_emb
        order = np.argsort(-scores)[:top_k]
        return [
            MatchResult(self._questions[i], float(scores[i]), "semantic")
            for i in order
        ]

    # -- lexical fallback (BM25; pure-python, no heavy deps) --
    def _lexical_match(self, query: str, top_k: int) -> list[MatchResult]:
        from rank_bm25 import BM25Okapi

        tokenized_corpus = [
            _tokenize(f"{q.topic} {q.question}") for q in self._questions
        ]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            MatchResult(self._questions[i], float(scores[i]), "lexical")
            for i in ranked
        ]
