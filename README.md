# Yorùbá Health Information Agent

An agent that answers Yorùbá health questions from English public-health documents,
cites its sources, and refuses when it shouldn't answer. A harness measures whether the
agent loop beats a fixed retrieval pipeline. Built on Cohere (`embed-multilingual-v3.0`,
`rerank-multilingual-v3.0`, Command chat).

> **Not medical advice.** This is a research prototype. It has not been clinically
> validated. If you are unwell, see a health worker.

## Status

- [x] Session 1: corpus (fetch, chunk, embed), English search check
- [x] Session 2: eval set, synthetic v0 (hand-written Yorùbá v1 still to come)
- [ ] Session 3: tools + fixed pipeline, retrieval table
- [ ] Session 4: agent harness (loop, retries, budget, traces)
- [ ] Session 5: run both, report tables
- [ ] Session 6: Gradio app on HF Spaces, full write-up

## Corpus

There are 150 [MedlinePlus](https://medlineplus.gov) health topics across eight areas:
malaria and tropical infections, TB, hypertension, diabetes, pregnancy, vaccination,
nutrition, and sickle cell (`corpus/topics.txt`). MedlinePlus health-topic summaries are
written by the US National Library of Medicine and are public domain.

The text comes from the official [MedlinePlus XML dump](https://medlineplus.gov/xml.html),
not from scraping the site. NLM only keeps the last few days of dumps online, so
the build pins a dump date and `corpus/corpus.jsonl` is committed. The eval gold
labels refer to its chunk IDs (`<medlineplus topic id>-<chunk index>`).

Chunks hold about 400 tokens (Cohere tokenizer) with about 50 tokens of overlap. They
are packed at sentence boundaries so they still read well when shown as citations.

## Eval set

`evals/questions.jsonl` has 60 questions: 40 answerable (with 1–3 gold chunk IDs each),
10 out-of-scope, and 10 asking for personal medical advice. The last two groups must be
refused.

**The current set is synthetic (`"source": "synthetic-v0"`).** Questions are written in
English by `command-a-03-2025` from sampled corpus chunks and translated to Yorùbá by
`command-a-translate-08-2025` (`evals/generate_questions.py`). Gold labels are the
source chunk plus any English top-5 hits that an LLM judge says also answer the
question. This has known biases:

- The questions come from the passages, so they share vocabulary with them, which
  makes retrieval look better than it is on real questions.
- The same model family writes the Yorùbá and later reads it, so the cross-lingual
  gap is likely understated.

I searched for an existing set and found none that fits. [AfriMed-QA](https://arxiv.org/abs/2411.15640)
is English-only. [AfriQA](https://github.com/masakhane-io/afriqa) and
[Y-NQ](https://aclanthology.org/2025.africanlp-1.34/) have Yorùbá questions, but they
are not about health. A hand-written native-speaker Yorùbá set will replace v0 under
the same schema.

## Running

```bash
uv sync
cp .env.example .env            # add COHERE_API_KEY
uv run python -m corpus.build_corpus
uv run python -m agent.tools "how is malaria spread"
uv run pytest
```
