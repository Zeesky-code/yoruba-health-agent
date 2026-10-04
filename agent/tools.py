"""The agent's five tools: three that gather evidence, two terminal ones that end a run.

search_corpus -> rerank -> (translate_query) -> answer | refuse
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path
from typing import Literal

import cohere
import httpx
import numpy as np
from dotenv import load_dotenv

EMBED_MODEL = "embed-multilingual-v3.0"
RERANK_MODEL = "rerank-multilingual-v3.0"
CHAT_MODEL = "command-a-03-2025"
TRANSLATE_MODEL = "tiny-aya-global"  # best EN->YO round-trip of 4 Cohere models tried

ANSWER_MODEL = CHAT_MODEL

CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "corpus.jsonl"


@dataclass(frozen=True)
class Chunk:
    id: str
    topic: str
    area: str
    url: str
    text: str
    score: float = 0.0


@dataclass(frozen=True)
class Citation:
    start: int
    end: int
    text: str
    chunk_ids: list[str]


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)  # passages the model was given


RefusalReason = Literal["out_of_scope", "personal_medical", "low_confidence"]

REFUSAL_MESSAGES: dict[str, str] = {
    "out_of_scope": "Ẹ má bínú, ìbéèrè nípa ìlera nìkan ni mo lè dáhùn.",
    "personal_medical": (
        "Ẹ má bínú, mi ò lè fún yín ní ìmọ̀ràn ìtọ́jú fún ara yín. Ẹ jọ̀wọ́ ẹ lọ rí dókítà "
        "tàbí òṣìṣẹ́ ìlera. Tí ó bá jẹ́ pàjáwìrì, ẹ lọ sí ilé ìwòsàn lẹ́sẹ̀kẹsẹ̀."
    ),
    "low_confidence": (
        "Ẹ má bínú, mi ò rí ìdáhùn tó dájú sí ìbéèrè yìí nínú àwọn ìwé ìlera tí mo ní."
    ),
}


@dataclass(frozen=True)
class Refusal:
    reason: RefusalReason
    text: str


@cache
def client() -> cohere.ClientV2:
    load_dotenv()
    key = os.environ.get("COHERE_API_KEY")
    if not key:
        raise RuntimeError("COHERE_API_KEY is not set (see .env.example)")
    return cohere.ClientV2(api_key=key)


def call_with_rate_limit(fn, *args, attempts: int = 6, wait_s: float = 15.0, **kwargs):
    """Call a Cohere client method, waiting out 429s and dropped connections.

    Trial keys allow 20 calls/minute, so a long batch job hits 429s by design.
    """
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except (cohere.errors.TooManyRequestsError, httpx.TransportError):
            if attempt == attempts - 1:
                raise
            time.sleep(wait_s)


def embed(texts: list[str], input_type: str) -> np.ndarray:
    """Embed texts in batches of 96 (the API limit). Returns L2-normalised rows."""
    rows: list[list[float]] = []
    for i in range(0, len(texts), 96):
        resp = call_with_rate_limit(
            client().embed,
            model=EMBED_MODEL,
            input_type=input_type,
            texts=texts[i : i + 96],
            embedding_types=["float"],
        )
        rows.extend(resp.embeddings.float_)
    vecs = np.asarray(rows, dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def chat_text(model: str, prompt: str, **kwargs) -> str:
    """Single-turn chat returning the text part (reasoning models also return thinking)."""
    messages = [{"role": "user", "content": prompt}]
    resp = call_with_rate_limit(client().chat, model=model, messages=messages, **kwargs)
    return next(c.text for c in resp.message.content if c.type == "text").strip()


@cache
def load_corpus(path: Path = CORPUS_PATH) -> tuple[list[Chunk], np.ndarray]:
    chunks, vecs = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            vecs.append(rec.pop("embedding"))
            rec.pop("n_tokens", None)
            chunks.append(Chunk(**rec))
    matrix = np.asarray(vecs, dtype=np.float32)
    return chunks, matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def top_k(query_vec: np.ndarray, chunks: list[Chunk], matrix: np.ndarray, k: int) -> list[Chunk]:
    scores = matrix @ query_vec
    order = np.argsort(-scores)[:k]
    return [replace(chunks[i], score=float(scores[i])) for i in order]


def search_corpus(query: str, k: int = 50) -> list[Chunk]:
    chunks, matrix = load_corpus()
    query_vec = embed([query], input_type="search_query")[0]
    return top_k(query_vec, chunks, matrix, k)


def document_text(chunk: Chunk) -> str:
    return f"{chunk.topic}\n\n{chunk.text}"


def rerank(query: str, chunks: list[Chunk], top_n: int = 5) -> list[Chunk]:
    """Reorder chunks by cross-encoder relevance; score becomes the rerank score (0-1)."""
    if not chunks:
        return []
    resp = call_with_rate_limit(
        client().rerank,
        model=RERANK_MODEL,
        query=query,
        documents=[document_text(c) for c in chunks],
        top_n=min(top_n, len(chunks)),
    )
    return [replace(chunks[r.index], score=float(r.relevance_score)) for r in resp.results]


TRANSLATE_QUERY_PROMPT = """Translate this health question into {target}. Keep medical terms
precise. Output only the translation.

{text}"""

LANGUAGE_NAMES = {"en": "English", "yo": "Yorùbá"}


def translate_query(text: str, target: str = "en") -> str:
    """Fallback for when retrieval on the original-language query scores low.

    Uses tiny-aya-global: command-a-03-2025 invented a different question for 4 of 40
    Yorùbá eval questions (vaccines -> "secondhand smoke"), tiny-aya-global for none.
    """
    prompt = TRANSLATE_QUERY_PROMPT.format(target=LANGUAGE_NAMES[target], text=text)
    return chat_text(TRANSLATE_MODEL, prompt, temperature=0)


ANSWER_SYSTEM_PROMPT = """You answer health questions for Yorùbá speakers using ONLY the
provided documents, which are English MedlinePlus pages.

- Always reply in Yorùbá with full tone marks, in plain everyday language.
- Use only facts from the documents. If they do not answer the question, say so.
- Give general information only: no diagnoses, no drug doses, no advice about one
  person's own treatment. Suggest seeing a health worker where it helps.
- Keep it short: three to six sentences."""


def answer(passages: list[Chunk], question: str) -> Answer:
    """Terminal tool: grounded answer in Yorùbá, with Cohere's native citations."""
    resp = call_with_rate_limit(
        client().chat,
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        documents=[
            {"id": c.id, "data": {"title": c.topic, "snippet": c.text, "url": c.url}}
            for c in passages
        ],
        temperature=0,
    )
    text = next(c.text for c in resp.message.content if c.type == "text").strip()
    citations = [
        Citation(
            start=c.start,
            end=c.end,
            text=c.text,
            chunk_ids=sorted({s.id for s in (c.sources or []) if s.id}),
        )
        for c in (resp.message.citations or [])
    ]
    return Answer(text=text, citations=citations, chunk_ids=[c.id for c in passages])


def refuse(reason: RefusalReason) -> Refusal:
    """Terminal tool: a fixed Yorùbá message per reason, so refusals cost nothing."""
    return Refusal(reason=reason, text=REFUSAL_MESSAGES[reason])


if __name__ == "__main__":
    import sys

    for c in search_corpus(" ".join(sys.argv[1:]), k=5):
        print(f"{c.score:.3f}  {c.id:<10} {c.topic}  |  {c.text[:90]!r}")
