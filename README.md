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
- [x] Session 3: tools + fixed pipeline, retrieval table (Yorùbá answer quality still open)
- [ ] Session 4: agent harness (loop, retries, budget, traces)
- [ ] Session 5: run both, report tables
- [ ] Session 6: Gradio app on HF Spaces, full write-up

## Results so far

### Retrieval (40 answerable questions, synthetic v0 eval set)

| Query | Ranking | recall@5 | nDCG@10 |
|---|---|---|---|
| Yorùbá query | Embed only | 0.642 | 0.599 |
| Yorùbá query | Embed + Rerank | **0.296** | 0.281 |
| Yorùbá → English (`translate_query`) | Embed only | 0.842 | 0.825 |
| Yorùbá → English (`translate_query`) | Embed + Rerank | 0.833 | 0.858 |
| English query (ceiling) | Embed only | 0.975 | 0.951 |
| English query (ceiling) | Embed + Rerank | 0.975 | 0.959 |

- **`rerank-multilingual-v3.0` hurts Yorùbá queries.** It halves recall@5 (0.64 → 0.30),
  and its relevance scores collapse to about 0.003 for every Yorùbá query, so they
  can't be used as a confidence signal. `embed-multilingual-v3.0` handles Yorùbá
  reasonably well.
- **Translating the query first recovers most of the gap** (0.84 vs the 0.975 English
  ceiling). After translation, the rerank scores separate questions again. The median
  top score is 0.998 for answerable questions and 0.000 for out-of-scope ones. That
  makes rerank useful as the refusal signal even though it adds no recall.
- **Part of the English ceiling is leakage.** The synthetic questions were written from
  the English passages, so English queries share their wording. The true cross-lingual
  gap is smaller than 0.975 − 0.842 suggests.

Reproduce with `uv run python -m evals.run_eval retrieval && uv run python -m evals.report`.
Raw rankings are in `evals/results/retrieval.jsonl`.

## What broke

- **`translate_query` on `command-a-03-2025` made up different questions.** For 4 of
  the 40 Yorùbá questions, the "translation" was a different question:
  *"How do vaccines protect me and my community?"* came back as *"How does
  secondhand smoke affect me…?"*, and handwashing came back as hypertension medication.
  Switching to `tiny-aya-global` fixed all four and raised translated recall@5 from 0.68
  to 0.84. The old rankings are kept in
  `evals/results/retrieval_translate-command-a.jsonl`.
- **No Cohere model tested writes good grounded Yorùbá answers yet.** `command-a-03-2025`
  cites its sources correctly, but its Yorùbá is often wrong. One answer opened with
  "a gift is what causes malaria", and another got stuck repeating a sentence.
  `tiny-aya-global` writes fluent Yorùbá, but it ignored the documents, gave no
  citations, and named antimalarial drugs that aren't in the sources.
- **The pipeline's refusal threshold (top rerank score < 0.1) can't catch personal
  medical questions.** Those retrieve relevant pages and score high (median 0.72), so
  only 1 of 10 is refused. Out-of-scope questions are caught (9 of 10). The threshold
  was set on this same eval set.

## Corpus

There are 150 [MedlinePlus](https://medlineplus.gov) health topics across nine areas:
malaria and other fevers, other infections (HIV, hepatitis, diarrhoea), TB,
hypertension, diabetes, pregnancy, vaccination, nutrition, and sickle cell (`corpus/topics.txt`). MedlinePlus health-topic summaries are
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
`tiny-aya-global` (`evals/generate_questions.py`). I picked the translator by round-trip
scoring: 15 questions went EN → YO → EN, and I measured the embedding similarity to the
original. `tiny-aya-global` scored 0.77, `tiny-aya-earth` 0.75,
`north-small-translate-09-2026` 0.70 and `command-a-translate-08-2025` 0.63. Each row
keeps its own score in `yo_roundtrip_sim`. The score catches garbled rows but misses
wrong ones. One question asks about blood sugar being too *low*, its Yorùbá says too
*high*, and it still scored 0.93, because the back-translation quietly fixed the error. None of the models handles medical
vocabulary well. "Diabetes", for example, came back as unrelated words. Gold labels are the
source chunk plus any English top-5 hits that an LLM judge says also answer the
question. Afterwards, 45 of the 60 Yorùbá questions were post-edited
(`"yo_post_edited": true`). The edits replace mistranslated medical terms with the words
people actually use: *ibà* (malaria), *ikọ́ ẹ̀gbẹ* (TB), *ẹ̀jẹ̀ ríru* (high blood
pressure), *àrùn ṣúgà* (diabetes), *abẹ́rẹ́ àjẹsára* (vaccine), *ẹ̀fọn* (mosquito). They
also fix reversed meanings, such as "too low" that the model had turned into "too high".
The post-edits were made with an LLM, not by a native speaker. In those rows
`yo_roundtrip_sim` still scores the original machine translation. This has known
biases:

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
