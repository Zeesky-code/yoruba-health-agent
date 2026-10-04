"""The agent's tools."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

import cohere
import numpy as np
from dotenv import load_dotenv

EMBED_MODEL = "embed-multilingual-v3.0"
RERANK_MODEL = "rerank-multilingual-v3.0"
CHAT_MODEL = "command-a-03-2025"
TRANSLATE_MODEL = "command-a-translate-08-2025"

CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "corpus.jsonl"


@dataclass(frozen=True)
class Chunk:
    id: str
    topic: str
    area: str
    url: str
    text: str
    score: float = 0.0


@cache
def client() -> cohere.ClientV2:
    load_dotenv()
    key = os.environ.get("COHERE_API_KEY")
    if not key:
        raise RuntimeError("COHERE_API_KEY is not set (see .env.example)")
    return cohere.ClientV2(api_key=key)


def call_with_rate_limit(fn, *args, attempts: int = 6, wait_s: float = 15.0, **kwargs):
    """Call a Cohere client method, waiting out 429s (trial keys allow 20 calls/minute)."""
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except cohere.errors.TooManyRequestsError:
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


if __name__ == "__main__":
    import sys

    for c in search_corpus(" ".join(sys.argv[1:]), k=5):
        print(f"{c.score:.3f}  {c.id:<10} {c.topic}  |  {c.text[:90]!r}")
