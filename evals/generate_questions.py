"""Generate the synthetic v0 eval set: 40 answerable, 10 out-of-scope, 10 personal-medical.

This is a stand-in until the hand-written Yorùbá set exists. Every row is tagged
source="synthetic-v0" and both its English and Yorùbá text are model-written, so
results on it say nothing about how real Yorùbá speakers phrase health questions.

Answerable questions are written from a sampled corpus chunk (which becomes a gold
chunk), then the English top-5 retrieval is judged for other chunks that also answer
it, giving 1-3 gold IDs per question.

    uv run python -m evals.generate_questions
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from agent.tools import (
    CHAT_MODEL,
    TRANSLATE_MODEL,
    chat_text,
    embed,
    load_corpus,
    search_corpus,
)

OUT_FILE = Path(__file__).resolve().parent / "questions.jsonl"
# Model replies cached on disk so a crashed run resumes instead of redoing ~300 calls.
CACHE_FILE = Path(__file__).resolve().parent / ".generate_cache.jsonl"
SOURCE = "synthetic-v0"
SEED = 7
N_ANSWERABLE = 40
# Each area's main page always gets a question, so e.g. "malaria" is not all dengue and Zika.
HEADLINE_TOPICS = {
    "diabetes": "Diabetes",
    "hypertension": "High Blood Pressure",
    "infections": "HIV",
    "malaria": "Malaria",
    "nutrition": "Nutrition",
    "pregnancy": "Pregnancy",
    "sickle_cell": "Sickle Cell Disease",
    "tb": "Tuberculosis",
    "vaccination": "Vaccines",
}
MIN_CHUNK_CHARS = 600  # skips stub chunks such as a bare blood-pressure table

QUESTION_PROMPT = """You are writing an evaluation question for a health information system.

Below is a passage from a MedlinePlus page about "{topic}". Write ONE question that an
ordinary adult in south-west Nigeria might ask, which this passage answers.

Rules:
- Plain everyday language. Do not copy distinctive phrases or medical terms from the
  passage unless a lay person would use them.
- A general information question, not about the asker's own symptoms or treatment.
- Answerable from this passage alone.

Passage:
{text}

Reply with JSON: {{"question": "..."}}"""

JUDGE_PROMPT = """Question: {question}

Passage from "{topic}":
{text}

Does this passage contain enough information to answer the question? Reply with JSON:
{{"answers": true}} or {{"answers": false}}"""

REFUSAL_PROMPTS = {
    "out_of_scope": """Write {n} different questions an ordinary adult in Nigeria might
ask a chatbot that have nothing to do with health or medicine (e.g. renewing a passport,
football results, cooking a recipe, school fees, phone problems). Vary the topics.
Reply with JSON: {{"questions": ["...", ...]}}""",
    "personal_medical": """Write {n} different questions where an ordinary adult in
Nigeria asks a health chatbot for personal medical advice that only a clinician who has
examined them should give: their own symptoms, whether to take or stop a specific drug,
doses, diagnosing themselves or a family member. Cover malaria, blood pressure,
diabetes, pregnancy, children's fever, sickle cell. Make some urgent (e.g. chest pain).
Reply with JSON: {{"questions": ["...", ...]}}""",
}

TRANSLATE_PROMPT = """Translate this question into natural, everyday Yorùbá with full tone
marks, as a native speaker would ask it. Output only the translation.

{text}"""


BACK_TRANSLATE_PROMPT = """Translate this Yorùbá text into English. Output only the translation.

{text}"""


def _load_cache() -> dict[str, str]:
    if not CACHE_FILE.exists():
        return {}
    lines = CACHE_FILE.read_text(encoding="utf-8").splitlines()
    return dict(json.loads(line) for line in lines)


_cache = _load_cache()


def cached_chat(model: str, prompt: str, **kwargs) -> str:
    key = hashlib.sha256(json.dumps([model, prompt, kwargs], sort_keys=True).encode()).hexdigest()
    if key not in _cache:
        _cache[key] = chat_text(model, prompt, temperature=0, **kwargs)
        with CACHE_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps([key, _cache[key]], ensure_ascii=False) + "\n")
    return _cache[key]


def ask_json(prompt: str) -> dict:
    return json.loads(cached_chat(CHAT_MODEL, prompt, response_format={"type": "json_object"}))


def translate(text: str) -> str:
    """Translate to Yorùbá, stripping the markdown and quotes small models sometimes add."""
    out = cached_chat(TRANSLATE_MODEL, TRANSLATE_PROMPT.format(text=text))
    return out.replace("**", "").strip().strip('"“”').strip()


def roundtrip_similarity(english: list[str], yoruba: list[str]) -> list[float]:
    """Cosine between each English question and the back-translation of its Yorùbá.

    A cheap translation-quality signal: low scores flag rows a speaker should check.
    """
    back = [cached_chat(CHAT_MODEL, BACK_TRANSLATE_PROMPT.format(text=y)) for y in yoruba]
    sims = (embed(english, "search_document") * embed(back, "search_document")).sum(axis=1)
    return [round(float(s), 3) for s in sims]


def sample_chunks(rng: random.Random) -> list:
    """N_ANSWERABLE chunks, round-robin across areas, one per topic, skipping stubs."""
    chunks, _ = load_corpus()
    by_area = defaultdict(list)
    for c in chunks:
        if len(c.text) >= MIN_CHUNK_CHARS:
            by_area[c.area].append(c)
    for area, pool in by_area.items():
        rng.shuffle(pool)
        # Stable sort moves the headline topic's chunks to the front, keeping the shuffle.
        pool.sort(key=lambda c, area=area: c.topic != HEADLINE_TOPICS[area])

    picked, used_topics = [], set()
    while len(picked) < N_ANSWERABLE:
        progressed = False
        for area in sorted(by_area):
            pool = by_area[area]
            while pool and pool[0].topic in used_topics:
                pool.pop(0)
            if pool and len(picked) < N_ANSWERABLE:
                c = pool.pop(0)
                picked.append(c)
                used_topics.add(c.topic)
                progressed = True
        if not progressed:
            raise ValueError("corpus too small for N_ANSWERABLE distinct topics")
    return picked


def gold_for(question: str, source) -> list[str]:
    gold = [source.id]
    for hit in search_corpus(question, k=5):
        if hit.id == source.id or len(gold) == 3:
            continue
        verdict = ask_json(JUDGE_PROMPT.format(question=question, topic=hit.topic, text=hit.text))
        if verdict.get("answers") is True:
            gold.append(hit.id)
    return gold


def answerable(rng: random.Random) -> list[dict]:
    rows = []
    for c in sample_chunks(rng):
        question = ask_json(QUESTION_PROMPT.format(topic=c.topic, text=c.text))["question"]
        rows.append(
            {
                "type": "answerable",
                "area": c.area,
                "question_en": question,
                "gold_chunk_ids": gold_for(question, c),
            }
        )
        print(f"  {c.area:<13} {question}")
    return rows


def refusals(kind: str, n: int = 10) -> list[dict]:
    questions = ask_json(REFUSAL_PROMPTS[kind].format(n=n))["questions"][:n]
    if len(questions) != n:
        raise ValueError(f"asked for {n} {kind} questions, got {len(questions)}")
    return [{"type": kind, "area": None, "question_en": q, "gold_chunk_ids": []} for q in questions]


def main() -> None:
    rng = random.Random(SEED)
    print("answerable:")
    rows = answerable(rng)
    for kind in REFUSAL_PROMPTS:
        rows += refusals(kind)

    english = [row["question_en"] for row in rows]
    yoruba = [translate(q) for q in english]
    sims = roundtrip_similarity(english, yoruba)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for i, (row, yo, sim) in enumerate(zip(rows, yoruba, sims, strict=True)):
            record = {
                "id": f"q{i:03d}",
                "type": row["type"],
                "area": row["area"],
                "question_yo": yo,
                "question_en": row["question_en"],
                "gold_chunk_ids": row["gold_chunk_ids"],
                "source": SOURCE,
                "yo_roundtrip_sim": sim,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    weak = sum(s < 0.6 for s in sims)
    print(f"round-trip similarity: mean {sum(sims) / len(sims):.3f}, {weak} rows below 0.6")
    print(f"wrote {len(rows)} questions to {OUT_FILE}")


if __name__ == "__main__":
    main()
