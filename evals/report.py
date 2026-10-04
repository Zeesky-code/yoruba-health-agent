"""Score raw eval results and print the README tables as markdown.

uv run python -m evals.report
"""

from __future__ import annotations

import json
import math
from statistics import mean

from evals.run_eval import RETRIEVAL_OUT, load_questions

CONDITION_LABELS = {
    "yo": "Yorùbá query",
    "yo_mt": "Yorùbá → English (translate_query)",
    "en": "English query (ceiling)",
}


def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold)


def ndcg_at(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, cid in enumerate(ranked[:k]) if cid in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal


def retrieval_table() -> str:
    gold = {
        q["id"]: set(q["gold_chunk_ids"]) for q in load_questions() if q["type"] == "answerable"
    }
    rows = [json.loads(line) for line in RETRIEVAL_OUT.read_text(encoding="utf-8").splitlines()]

    lines = [
        f"| Query | Ranking | recall@5 | nDCG@10 |  (n={len(gold)} answerable)",
        "|---|---|---|---|",
    ]
    for condition, label in CONDITION_LABELS.items():
        scored = [r for r in rows if r["condition"] == condition and r["id"] in gold]
        for ranking, name in [("embed", "Embed only"), ("rerank", "Embed + Rerank")]:
            ranked = [([cid for cid, _ in r[ranking]], gold[r["id"]]) for r in scored]
            r5 = mean(recall_at(ids, g, 5) for ids, g in ranked)
            n10 = mean(ndcg_at(ids, g, 10) for ids, g in ranked)
            lines.append(f"| {label} | {name} | {r5:.3f} | {n10:.3f} |")
    return "\n".join(lines)


def rerank_score_table() -> str:
    """Top rerank score by question type: the signal a low-confidence threshold uses."""
    types = {q["id"]: q["type"] for q in load_questions()}
    rows = [json.loads(line) for line in RETRIEVAL_OUT.read_text(encoding="utf-8").splitlines()]
    lines = ["| Query | answerable | out_of_scope | personal_medical |", "|---|---|---|---|"]
    for condition, label in CONDITION_LABELS.items():
        cells = []
        for t in ["answerable", "out_of_scope", "personal_medical"]:
            tops = sorted(
                r["rerank"][0][1]
                for r in rows
                if r["condition"] == condition and types[r["id"]] == t
            )
            cells.append(f"median {tops[len(tops) // 2]:.3f} (min {tops[0]:.3f})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    print("## Retrieval\n")
    print(retrieval_table())
    print("\n## Top rerank score by question type\n")
    print(rerank_score_table())


if __name__ == "__main__":
    main()
