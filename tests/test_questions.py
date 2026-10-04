"""Shape checks on the committed eval set, whichever version (synthetic or hand-written)."""

import json
from collections import Counter
from pathlib import Path

import pytest

from agent.tools import load_corpus

QUESTIONS = Path(__file__).resolve().parent.parent / "evals" / "questions.jsonl"

pytestmark = pytest.mark.skipif(not QUESTIONS.exists(), reason="eval set not built yet")


@pytest.fixture(scope="module")
def rows() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]


def test_counts(rows):
    assert Counter(r["type"] for r in rows) == {
        "answerable": 40,
        "out_of_scope": 10,
        "personal_medical": 10,
    }


def test_ids_unique(rows):
    assert len({r["id"] for r in rows}) == len(rows)


def test_gold_labels(rows):
    corpus_ids = {c.id for c in load_corpus()[0]}
    for r in rows:
        if r["type"] == "answerable":
            assert 1 <= len(r["gold_chunk_ids"]) <= 3, r["id"]
            assert set(r["gold_chunk_ids"]) <= corpus_ids, r["id"]
        else:
            assert r["gold_chunk_ids"] == [], r["id"]


def test_every_area_covered(rows):
    areas = {r["area"] for r in rows if r["type"] == "answerable"}
    assert len(areas) == 9


def test_questions_present(rows):
    for r in rows:
        assert r["question_yo"].strip() and r["question_en"].strip(), r["id"]
