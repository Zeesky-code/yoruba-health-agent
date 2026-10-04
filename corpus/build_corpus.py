"""Fetch MedlinePlus health topics, chunk them, embed them, write corpus.jsonl.

Source is the official MedlinePlus XML dump (https://medlineplus.gov/xml.html), not
scraped pages. NLM keeps only the last few days of dumps online, so the date is pinned
and the built corpus.jsonl is committed: the eval gold labels point at its chunk IDs.

    uv run python -m corpus.build_corpus [--date YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path

import httpx

from agent.tools import EMBED_MODEL, client, embed

HERE = Path(__file__).resolve().parent
TOPICS_FILE = HERE / "topics.txt"
RAW_DIR = HERE / "raw"
OUT_FILE = HERE / "corpus.jsonl"

DUMP_DATE = "2026-10-03"
DUMP_URL = "https://medlineplus.gov/xml/mplus_topics_{date}.xml"

CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50

BLOCK_TAGS = {"p", "ul", "ol", "li", "h2", "h3", "h4", "div", "br"}


def read_topics(path: Path = TOPICS_FILE) -> dict[str, str]:
    """Return {title: area} from topics.txt."""
    topics: dict[str, str] = {}
    area = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            area = line[1:-1]
            continue
        if area is None:
            raise ValueError(f"topic {line!r} appears before any [area] header")
        topics[line] = area
    return topics


def fetch_dump(date: str) -> Path:
    path = RAW_DIR / f"mplus_topics_{date}.xml"
    if not path.exists():
        RAW_DIR.mkdir(exist_ok=True)
        resp = httpx.get(DUMP_URL.format(date=date), timeout=120, follow_redirects=True)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(html: str) -> str:
    """Flatten a MedlinePlus summary to plain text, one paragraph or list item per line."""
    parser = _TextExtractor()
    parser.feed(html)
    lines = (re.sub(r"\s+", " ", line).strip() for line in "".join(parser.parts).split("\n"))
    return "\n".join(line for line in lines if line and line != "-")


def parse_topics(xml_path: Path, wanted: dict[str, str]) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    found = {}
    for t in root.iter("health-topic"):
        title = t.get("title")
        if t.get("language") == "English" and title in wanted:
            found[title] = {
                "topic_id": t.get("id"),
                "topic": title,
                "area": wanted[title],
                "url": t.get("url"),
                "text": html_to_text(t.findtext("full-summary") or ""),
            }
    missing = sorted(set(wanted) - set(found))
    if missing:
        raise ValueError(f"topics not in dump: {missing}")
    empty = sorted(title for title, t in found.items() if not t["text"])
    if empty:
        raise ValueError(f"topics with an empty summary: {empty}")
    return [found[title] for title in wanted]


def split_units(text: str) -> list[str]:
    """Paragraphs and list items, with long paragraphs split into sentences.

    Each unit keeps its trailing separator (a space inside a paragraph, a newline at
    its end) so that joining units reproduces the original text.
    """
    units = []
    for line in text.split("\n"):
        sentences = [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if s]
        if not sentences:
            continue
        units.extend(f"{s} " for s in sentences[:-1])
        units.append(f"{sentences[-1]}\n")
    return units


def chunk_text(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int = CHUNK_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[str]:
    """Greedily pack sentence-level units into chunks of at most max_tokens.

    Each new chunk starts with the trailing units of the previous one, up to `overlap`
    tokens, so a fact split across a boundary is still retrievable from either side.
    Splitting on sentences rather than raw token windows keeps chunks readable when
    they are shown back as citations.
    """
    units = [(u, count_tokens(u)) for u in split_units(text)]
    chunks: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    size = 0
    for unit, n in units:
        if current and size + n > max_tokens:
            chunks.append(current)
            carry: list[tuple[str, int]] = []
            carried = 0
            for prev in reversed(current):
                if carried + prev[1] > overlap:
                    break
                carry.insert(0, prev)
                carried += prev[1]
            current, size = carry, carried
        current.append((unit, n))
        size += n
    if current:
        chunks.append(current)
    return ["".join(u for u, _ in chunk).strip() for chunk in chunks]


def cohere_token_counter() -> Callable[[str], int]:
    def count(text: str) -> int:
        return len(client().tokenize(text=text, model=EMBED_MODEL, offline=True).tokens)

    return count


def build(date: str) -> list[dict]:
    topics = parse_topics(fetch_dump(date), read_topics())
    count = cohere_token_counter()
    records = []
    for t in topics:
        for i, text in enumerate(chunk_text(t["text"], count)):
            records.append(
                {
                    "id": f"{t['topic_id']}-{i}",
                    "topic": t["topic"],
                    "area": t["area"],
                    "url": t["url"],
                    "text": text,
                    "n_tokens": count(text),
                }
            )
    # The title is prepended for embedding only: chunks deep inside a page often never
    # name the condition they are about.
    vecs = embed([f"{r['topic']}\n\n{r['text']}" for r in records], "search_document")
    for r, v in zip(records, vecs, strict=True):
        r["embedding"] = [round(float(x), 6) for x in v]
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", default=DUMP_DATE, help="MedlinePlus XML dump date")
    args = ap.parse_args()

    records = build(args.date)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_topics = len({r["topic"] for r in records})
    print(f"wrote {len(records)} chunks from {n_topics} topics to {OUT_FILE}")


if __name__ == "__main__":
    main()
