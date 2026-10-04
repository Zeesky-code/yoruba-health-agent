"""Run evaluations and write raw results under evals/results/ for report.py to score.

    uv run python -m evals.run_eval retrieval

retrieval: for every question, under three query conditions
  yo     the Yorùbá question as written
  yo_mt  the Yorùbá question machine-translated to English by translate_query
  en     the English question (ceiling: what retrieval could do without a language gap)
records the embedding top-10 and the rerank top-10 (reranking the embedding top-50).
Refusal questions are included so their top rerank scores can calibrate the
pipeline's low-confidence threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.tools import embed, load_corpus, rerank, top_k, translate_query

EVALS_DIR = Path(__file__).resolve().parent
QUESTIONS = EVALS_DIR / "questions.jsonl"
RESULTS_DIR = EVALS_DIR / "results"
RETRIEVAL_OUT = RESULTS_DIR / "retrieval.jsonl"

CANDIDATES = 50
KEEP = 10


def load_questions() -> list[dict]:
    return [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]


def _done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return {(r["id"], r["condition"]) for r in rows}


def run_retrieval() -> None:
    """Resumable: rows already in retrieval.jsonl are skipped (trial keys are slow)."""
    questions = load_questions()
    RESULTS_DIR.mkdir(exist_ok=True)
    done = _done_keys(RETRIEVAL_OUT)

    translated = {}
    for q in questions:
        if (q["id"], "yo_mt") not in done:
            translated[q["id"]] = translate_query(q["question_yo"])

    conditions = {
        "yo": {q["id"]: q["question_yo"] for q in questions},
        "yo_mt": translated,
        "en": {q["id"]: q["question_en"] for q in questions},
    }
    chunks, matrix = load_corpus()
    for condition, queries in conditions.items():
        todo = {qid: text for qid, text in queries.items() if (qid, condition) not in done}
        if not todo:
            continue
        vecs = embed(list(todo.values()), input_type="search_query")
        for (qid, text), vec in zip(todo.items(), vecs, strict=True):
            candidates = top_k(vec, chunks, matrix, CANDIDATES)
            reranked = rerank(text, candidates, top_n=KEEP)
            row = {
                "id": qid,
                "condition": condition,
                "query": text,
                "embed": [[c.id, round(c.score, 4)] for c in candidates[:KEEP]],
                "rerank": [[c.id, round(c.score, 4)] for c in reranked],
            }
            with RETRIEVAL_OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {condition:<6} {qid}  top: {reranked[0].id} ({reranked[0].score:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["retrieval"])
    args = ap.parse_args()
    if args.what == "retrieval":
        run_retrieval()


if __name__ == "__main__":
    main()
