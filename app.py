"""Gradio demo: ask in Yorùbá, get a cited answer or a refusal, and see every agent step.

uv run python app.py
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date
from pathlib import Path

import cohere
import gradio as gr

from agent.harness import run
from agent.tools import Answer, load_corpus

# The demo runs on a trial Cohere key (20 calls/minute, ~9 calls per answered question),
# so it caps usage per visitor session and per day across all visitors.
PER_SESSION_LIMIT = 5
DAILY_LIMIT = 60

QUESTIONS = Path(__file__).resolve().parent / "evals" / "questions.jsonl"

DISCLAIMER = (
    "**Kì í ṣe ìmọ̀ràn ìṣègùn / Not medical advice.** This is a research prototype that "
    "answers general health questions from US National Library of Medicine (MedlinePlus) "
    "pages. It has not been clinically validated. If you are unwell, see a health worker."
)

_daily = {"day": date.today(), "count": 0}
_daily_lock = threading.Lock()


def _take_daily_slot() -> bool:
    with _daily_lock:
        if _daily["day"] != date.today():
            _daily.update(day=date.today(), count=0)
        if _daily["count"] >= DAILY_LIMIT:
            return False
        _daily["count"] += 1
        return True


def _examples() -> list[str]:
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]
    picks = ["q003", "q007", "q027", "q053", "q042"]
    by_id = {r["id"]: r["question_yo"] for r in rows}
    return [by_id[p] for p in picks if p in by_id]


def _sources_markdown(output: Answer) -> str:
    chunks = {c.id: c for c in load_corpus()[0]}
    cited = sorted({cid for c in output.citations for cid in c.chunk_ids})
    if not cited:
        return "_No citations._"
    lines = []
    for cid in cited:
        c = chunks[cid]
        lines.append(f"- [{c.topic}]({c.url}) `{cid}`")
    return "\n".join(lines)


def _trace_rows(trace: list[dict]) -> list[list]:
    rows = []
    for t in trace:
        args = json.dumps(t.get("args", {}), ensure_ascii=False)
        rows.append(
            [
                t["step"],
                t["tool"],
                args[:120],
                t.get("top_score", ""),
                t.get("latency_ms", ""),
                t.get("cost_usd", ""),
                t.get("error") or "",
            ]
        )
    return rows


def ask(question: str, used: int):
    """Returns (answer, sources, english, trace_rows, used, status) for the UI."""

    def notice(msg: str, used_after: int = used):
        return msg, "", "", [], used_after, ""

    question = (question or "").strip()
    if not question:
        return notice("Ẹ jọ̀wọ́, ẹ kọ ìbéèrè yín. (Please type a question.)")
    if used >= PER_SESSION_LIMIT:
        return notice(f"This demo allows {PER_SESSION_LIMIT} questions per visit.")
    if not _take_daily_slot():
        return notice("The demo has reached its daily limit. Please try again tomorrow.")

    try:
        result = run(question)
    except cohere.errors.TooManyRequestsError:
        return notice(
            "The demo's API key is busy right now. Please try again in a minute.", used + 1
        )

    out = result.output
    status = (
        f"{result.steps} steps · ${result.cost_usd:.4f} · "
        f"{result.latency_ms / 1000:.1f}s · stopped by: {result.stopped_by}"
    )
    if isinstance(out, Answer):
        sources, english = _sources_markdown(out), out.text_en
    else:
        sources, english = f"_Refused: `{out.reason}`._", ""
    return out.text, sources, english, _trace_rows(result.trace), used + 1, status


def build() -> gr.Blocks:
    with gr.Blocks(title="Yorùbá Health Agent") as demo:
        gr.Markdown(
            "# Olùrànlọ́wọ́ Ìlera ní Èdè Yorùbá\n"
            "Ask a health question in Yorùbá. The agent searches English MedlinePlus pages, "
            "answers in Yorùbá with sources, and refuses questions that are off-topic or "
            "ask for personal medical advice.\n\n" + DISCLAIMER
        )
        used = gr.State(0)
        question = gr.Textbox(label="Ìbéèrè rẹ (your question, in Yorùbá)", lines=2)
        submit = gr.Button("Béèrè (Ask)", variant="primary")
        gr.Examples(_examples(), inputs=question)

        answer = gr.Textbox(label="Ìdáhùn (answer)", lines=6)
        status = gr.Markdown()
        sources = gr.Markdown(label="Sources")
        with gr.Accordion("English answer (before translation)", open=False):
            english = gr.Textbox(show_label=False, lines=6)
        with gr.Accordion("Agent trace", open=False):
            trace = gr.Dataframe(
                headers=["step", "tool", "args", "top score", "latency ms", "cost $", "error"],
                wrap=True,
            )
        gr.Markdown(
            "[Code, eval set and results on GitHub](https://github.com/Zeesky-code/yoruba-health-agent)"
        )

        outputs = [answer, sources, english, trace, used, status]
        submit.click(ask, [question, used], outputs)
        question.submit(ask, [question, used], outputs)
    return demo


demo = build()

if __name__ == "__main__":
    # One request at a time: the trial key cannot serve concurrent agent runs.
    # Render sets PORT and needs the server on all interfaces.
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("HOST", "127.0.0.1"),
        server_port=int(os.environ.get("PORT", 7860)),
    )
