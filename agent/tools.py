"""The agent's five tools: three that gather evidence, two terminal ones that end a run.

search_corpus -> rerank -> (translate_query) -> answer | refuse
"""

from __future__ import annotations

import json
import os
import re
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
    text_en: str = ""  # the grounded English answer before translation
    citations_en: list[Citation] = field(default_factory=list)


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


ANSWER_SYSTEM_PROMPT = """You answer health questions from Yorùbá speakers using ONLY the
provided documents, which are English MedlinePlus pages. The question may be in Yorùbá.

- Reply in plain, simple English. Your answer will be translated into Yorùbá, so use
  short sentences and everyday words.
- Use only facts from the documents. If they do not answer the question, say so.
- Give general information only: no diagnoses, no drug doses, no advice about one
  person's own treatment. Suggest seeing a health worker where it helps.
- Keep it short: three to six sentences. No markdown, no lists."""

# Terms the translator got wrong without help (e.g. mosquito -> ọlọ́wọ́, "rich person").
GLOSSARY = {
    "malaria": "ibà",
    "mosquito": "ẹ̀fọn",
    "mosquito net": "àwọ̀n ẹ̀fọn",
    "tuberculosis (TB)": "ikọ́ ẹ̀gbẹ",
    "high blood pressure": "ẹ̀jẹ̀ ríru",
    "diabetes": "àrùn ṣúgà",
    "blood sugar": "ṣúgà inú ẹ̀jẹ̀",
    "vaccine": "abẹ́rẹ́ àjẹsára",
    "germs / bacteria": "kòkòrò àrùn",
    "lungs": "ẹ̀dọ̀fóró",
    "kidneys": "kíndìnrín",
    "stroke": "àrùn ẹ̀gbà",
    "pregnancy": "oyún",
    "fever": "ibà / ara gbígbóná",
    "medicine": "oògùn",
    "health care provider / doctor": "dókítà / òṣìṣẹ́ ìlera",
}

TRANSLATE_ANSWER_PROMPT = """Translate each numbered English sentence into natural, everyday
Yorùbá with full tone marks. Keep medicine names and numbers unchanged. Use these terms:
{glossary}

Reply with exactly {n} lines, each starting with its number, and nothing else. No markdown.

{numbered}"""


def strip_markdown(text: str) -> str:
    return re.sub(r"[*_`#]+", "", text).strip()


def split_sentences(text: str, citations: list[Citation]) -> list[tuple[str, list[str]]]:
    """Split an answer into sentences and list items, each with the chunk IDs cited in it."""
    sentences = []
    for m in re.finditer(r"[^.!?\n]+[.!?]*", text):
        sentence = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", m.group()).strip()
        if not sentence:
            continue
        ids = sorted(
            {
                cid
                for c in citations
                if c.start < m.end() and c.end > m.start()
                for cid in c.chunk_ids
            }
        )
        sentences.append((sentence, ids))
    return sentences


def parse_numbered(text: str, n: int) -> list[str] | None:
    """Parse "1. ..." lines; None unless exactly lines 1..n are present."""
    found = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.+)", line)
        if m:
            found[int(m.group(1))] = m.group(2).strip()
    if sorted(found) != list(range(1, n + 1)):
        return None
    return [found[i] for i in range(1, n + 1)]


def translate_sentences(sentences: list[str]) -> list[str]:
    """One call for all sentences; sentence-by-sentence if the numbering comes back wrong."""
    glossary = "\n".join(f"- {en}: {yo}" for en, yo in GLOSSARY.items())
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    prompt = TRANSLATE_ANSWER_PROMPT.format(glossary=glossary, n=len(sentences), numbered=numbered)
    raw = strip_markdown(chat_text(TRANSLATE_MODEL, prompt, temperature=0))
    out = parse_numbered(raw, len(sentences))
    if out is not None:
        return out
    singles = []
    for sentence in sentences:
        prompt = TRANSLATE_ANSWER_PROMPT.format(glossary=glossary, n=1, numbered=f"1. {sentence}")
        raw = strip_markdown(chat_text(TRANSLATE_MODEL, prompt, temperature=0))
        singles.append((parse_numbered(raw, 1) or [raw])[0])
    return singles


def answer(passages: list[Chunk], question: str) -> Answer:
    """Terminal tool: grounded answer in Yorùbá with per-sentence citations.

    No Cohere model tested both writes good Yorùbá and stays grounded: command-a cites
    correctly but garbles Yorùbá, tiny-aya-global writes fluent Yorùbá but ignores the
    documents. So command-a answers in English with native citations, then
    tiny-aya-global translates sentence by sentence and each sentence keeps its sources.
    """
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
    text_en = next(c.text for c in resp.message.content if c.type == "text").strip()
    citations_en = [
        Citation(
            start=c.start,
            end=c.end,
            text=c.text,
            chunk_ids=sorted({s.id for s in (c.sources or []) if s.id}),
        )
        for c in (resp.message.citations or [])
    ]

    sentences = split_sentences(text_en, citations_en)
    translated = translate_sentences([s for s, _ in sentences])
    parts, citations, pos = [], [], 0
    for yo, (_, ids) in zip(translated, sentences, strict=True):
        if ids:
            citations.append(Citation(start=pos, end=pos + len(yo), text=yo, chunk_ids=ids))
        parts.append(yo)
        pos += len(yo) + 1
    return Answer(
        text=" ".join(parts),
        citations=citations,
        chunk_ids=[c.id for c in passages],
        text_en=text_en,
        citations_en=citations_en,
    )


def refuse(reason: RefusalReason) -> Refusal:
    """Terminal tool: a fixed Yorùbá message per reason, so refusals cost nothing."""
    return Refusal(reason=reason, text=REFUSAL_MESSAGES[reason])


if __name__ == "__main__":
    import sys

    for c in search_corpus(" ".join(sys.argv[1:]), k=5):
        print(f"{c.score:.3f}  {c.id:<10} {c.topic}  |  {c.text[:90]!r}")
