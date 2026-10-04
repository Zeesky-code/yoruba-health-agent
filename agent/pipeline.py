"""Fixed-path baseline: the same tools as the agent, called in a fixed order.

translate_query -> search_corpus -> rerank -> answer, or refuse("low_confidence") when the
best rerank score is under the threshold. Every query is translated first because
rerank-multilingual-v3.0 scores Yorùbá queries near zero (see the README's retrieval
table). The pipeline cannot tell an out-of-scope or personal-medical question from a
real one except through that threshold, which is the gap the agent harness should close.

    uv run python -m agent.pipeline "Báwo ni ibà ṣe ń ràn?"
"""

from __future__ import annotations

from agent.tools import Answer, Refusal, answer, refuse, rerank, search_corpus, translate_query

TOP_N = 5
# Calibrated on the eval set's top rerank scores (evals.report): see README.
MIN_RERANK_SCORE = 0.1


def run(question: str) -> Answer | Refusal:
    query = translate_query(question, target="en")
    passages = rerank(query, search_corpus(query, k=50), top_n=TOP_N)
    if not passages or passages[0].score < MIN_RERANK_SCORE:
        return refuse("low_confidence")
    return answer(passages, question)


if __name__ == "__main__":
    import sys

    result = run(" ".join(sys.argv[1:]))
    print(result.text)
    if isinstance(result, Answer):
        print("\nsources:", sorted({cid for c in result.citations for cid in c.chunk_ids}))
