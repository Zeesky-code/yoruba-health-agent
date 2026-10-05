"""Gradio demo: ask in Yorùbá, get a cited answer or a refusal, and see every agent step.

uv run python app.py
"""

from __future__ import annotations

import html
import json
import os
import threading
from datetime import date
from pathlib import Path

import cohere
import gradio as gr

from agent.harness import RunResult, run
from agent.tools import Answer, Chunk, Refusal, load_corpus

# The demo runs on a trial Cohere key (20 calls/minute, ~9 calls per answered question),
# so it caps usage per visitor session and per day across all visitors.
PER_SESSION_LIMIT = 5
DAILY_LIMIT = 60

QUESTIONS = Path(__file__).resolve().parent / "evals" / "questions.jsonl"
GITHUB_URL = "https://github.com/Zeesky-code/yoruba-health-agent"

# (button label, eval question id): three answerable, two the agent should decline.
EXAMPLES = [
    ("Malaria while travelling", "q003"),
    ("How TB spreads", "q007"),
    ("Signs of low blood sugar", "q027"),
    ("“Should I double my BP pills?”", "q053"),
    ("Jollof rice recipe", "q042"),
]

REFUSAL_LABELS = {
    "out_of_scope": "Not a health question",
    "personal_medical": "Personal medical advice: please see a health worker",
    "low_confidence": "No reliable source found",
}

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


def _example_questions() -> dict[str, str]:
    rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r["question_yo"] for r in rows}
    return {label: by_id[qid] for label, qid in EXAMPLES if qid in by_id}


# ---------- rendering (pure functions, covered by tests/test_app.py) ----------


def render_answer(answer: Answer, chunks: dict[str, Chunk]) -> str:
    """Answer text with superscript citation numbers, plus a numbered source list."""
    numbers: dict[str, int] = {}
    parts, pos = [], 0
    for c in sorted(answer.citations, key=lambda c: c.start):
        parts.append(html.escape(answer.text[pos : c.end]))
        refs = sorted({numbers.setdefault(cid, len(numbers) + 1) for cid in c.chunk_ids})
        parts.append(f"<sup class='cite'>{','.join(str(n) for n in refs)}</sup>")
        pos = c.end
    parts.append(html.escape(answer.text[pos:]))

    items = "".join(
        f"<li><a href='{html.escape(chunks[cid].url)}' target='_blank'>"
        f"{html.escape(chunks[cid].topic)}</a> <span class='cid'>{cid}</span></li>"
        for cid in numbers
        if cid in chunks
    )
    sources = (
        f"<div class='sources'><div class='label'>Àwọn orísun · Sources</div><ol>{items}</ol></div>"
        if items
        else ""
    )
    return f"<div class='card answer'><p>{''.join(parts)}</p>{sources}</div>"


def render_refusal(refusal: Refusal) -> str:
    label = REFUSAL_LABELS.get(refusal.reason, refusal.reason)
    return (
        f"<div class='card refusal'><span class='badge'>{html.escape(label)}</span>"
        f"<p>{html.escape(refusal.text)}</p></div>"
    )


def render_notice(message: str) -> str:
    return f"<div class='card notice'><p>{html.escape(message)}</p></div>"


def render_meta(result: RunResult) -> str:
    stopped = "" if result.stopped_by == "terminal" else f" · stopped by {result.stopped_by}"
    return (
        f"<div class='meta'>{result.steps} step{'s' if result.steps != 1 else ''} · "
        f"${result.cost_usd:.4f} · "
        f"{result.latency_ms / 1000:.1f}s{stopped}</div>"
    )


def describe_step(row: dict) -> str:
    """One plain-language line per trace row."""
    if row.get("error"):
        return f"error: {row['error']}"
    tool, args = row["tool"], row.get("args", {})
    if tool == "translate_query":
        return f"→ “{row.get('translation', '')}”"
    if tool == "search_corpus":
        n, top = row.get("n_results", 0), row.get("top_score") or 0
        return f"{n} passages · top cosine {top:.2f}"
    if tool == "rerank":
        return f"top relevance {row.get('top_score') or 0:.2f}"
    if tool == "answer":
        ids = ", ".join(args.get("chunk_ids", []))
        return f"from {ids} · {row.get('n_citations', 0)} cited sentences"
    if tool == "refuse":
        why = f" (stopped by {row['stopped_by']})" if row.get("stopped_by") else ""
        return f"reason: {args.get('reason', '')}{why}"
    return json.dumps(args, ensure_ascii=False)


def render_trace(trace: list[dict]) -> str:
    """The run's trace as a compact table: step, tool, what happened, latency, cost."""
    if not trace:
        return ""
    rows = []
    for t in trace:
        cost = f"{t['cost_usd']:.4f}" if "cost_usd" in t else ""
        rows.append(
            f"<tr><td>{t['step']}</td><td><code>{html.escape(t['tool'])}</code></td>"
            f"<td>{html.escape(describe_step(t))}</td>"
            f"<td class='num'>{t.get('latency_ms', '')}</td><td class='num'>{cost}</td></tr>"
        )
    return (
        "<table class='trace'><thead><tr><th>#</th><th>tool</th><th>what happened</th>"
        "<th class='num'>ms</th><th class='num'>$</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


# ---------- request handling ----------


def ask(question: str, used: int):
    """Returns (answer_html, meta_html, english_panel, english, trace_html, used) for the UI."""

    def notice(msg: str, used_after: int = used):
        return render_notice(msg), "", gr.update(visible=False), "", "", used_after

    question = (question or "").strip()
    if not question:
        return notice("Ẹ jọ̀wọ́, ẹ kọ ìbéèrè yín. Please type a question, or pick an example.")
    if used >= PER_SESSION_LIMIT:
        return notice(f"This demo allows {PER_SESSION_LIMIT} questions per visit.")
    if not _take_daily_slot():
        return notice("The demo has reached its daily limit. Please try again tomorrow.")

    try:
        result = run(question)
    except cohere.errors.TooManyRequestsError:
        return notice("The demo's API key is busy. Please try again in a minute.", used + 1)

    out = result.output
    if isinstance(out, Answer):
        chunks = {c.id: c for c in load_corpus()[0]}
        body, english = render_answer(out, chunks), out.text_en
    else:
        body, english = render_refusal(out), ""
    english_panel = gr.update(visible=bool(english))
    return body, render_meta(result), english_panel, english, render_trace(result.trace), used + 1


# ---------- layout ----------

CSS = """
.gradio-container { max-width: 820px !important; margin: 0 auto !important; }
#hero h1 { font-size: 2rem; margin: 0.5rem 0 0.25rem; }
#hero .sub { color: var(--body-text-color-subdued); margin: 0; }
#disclaimer { font-size: 0.85rem; padding: 0.6rem 0.9rem; border-radius: 10px;
  background: var(--color-accent-soft); color: var(--body-text-color); }
.card { border: 1px solid var(--border-color-primary); border-radius: 12px;
  padding: 1rem 1.2rem; background: var(--block-background-fill); line-height: 1.7; }
.card p { margin: 0; font-size: 1.05rem; }
sup.cite { color: var(--color-accent); font-weight: 600; margin-left: 1px; }
.sources { margin-top: 0.9rem; padding-top: 0.7rem;
  border-top: 1px dashed var(--border-color-primary); }
.sources .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--body-text-color-subdued); }
.sources ol { margin: 0.3rem 0 0 1.2rem; padding: 0; font-size: 0.9rem; }
.cid { font-family: var(--font-mono); font-size: 0.75rem; color: var(--body-text-color-subdued); }
.refusal .badge { display: inline-block; font-size: 0.75rem; font-weight: 600;
  margin-bottom: 0.5rem; padding: 0.15rem 0.6rem; border-radius: 999px;
  background: #fef3c7; color: #92400e; }
.dark .refusal .badge { background: #451a03; color: #fcd34d; }
.meta { font-size: 0.8rem; color: var(--body-text-color-subdued); }
#examples { flex-wrap: wrap; gap: 0.4rem; }
#examples button { flex: 0 0 auto; }
.card a, #footer a { color: var(--color-accent); }
table.trace { width: 100%; border-collapse: collapse; font-size: 0.85rem; border: none; }
table.trace th, table.trace td { border-left: none !important; border-right: none !important;
  border-top: none !important; }
table.trace th { text-align: left; font-weight: 600; color: var(--body-text-color-subdued);
  border-bottom: 1px solid var(--border-color-primary); padding: 0.35rem 0.5rem; }
table.trace td { padding: 0.35rem 0.5rem; border-bottom: 1px solid var(--border-color-primary);
  vertical-align: top; }
table.trace .num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
table.trace code { font-size: 0.8rem; white-space: nowrap; }
#footer { text-align: center; font-size: 0.8rem; color: var(--body-text-color-subdued);
  margin-top: 1rem; }
"""

THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="amber",
    neutral_hue="stone",
    font=(gr.themes.GoogleFont("Noto Sans"), "ui-sans-serif", "system-ui", "sans-serif"),
)


def build() -> gr.Blocks:
    examples = _example_questions()
    with gr.Blocks(title="Yorùbá Health Agent") as demo:
        gr.HTML(
            "<div id='hero'><h1>Olùrànlọ́wọ́ Ìlera</h1>"
            "<p class='sub'>Ask a health question in Yorùbá. A Cohere agent reads English "
            "MedlinePlus pages, answers in Yorùbá with sources, and declines off-topic "
            "questions or requests for personal medical advice.</p></div>"
        )
        gr.HTML(
            "<div id='disclaimer'><b>Kì í ṣe ìmọ̀ràn ìṣègùn · Not medical advice.</b> "
            "A research prototype that has not been clinically validated. "
            "If you are unwell, see a health worker.</div>"
        )

        used = gr.State(0)
        with gr.Group():
            question = gr.Textbox(
                label="Ìbéèrè rẹ · Your question",
                placeholder="Bí àpẹẹrẹ: Báwo ni ibà ṣe ń ràn?",
                lines=2,
            )
            submit = gr.Button("Béèrè · Ask", variant="primary")
        with gr.Row(elem_id="examples"):
            for label, text in examples.items():
                gr.Button(label, size="sm", variant="secondary", scale=0, min_width=0).click(
                    lambda t=text: t, outputs=question
                )

        answer = gr.HTML()
        meta = gr.HTML()
        with gr.Accordion(
            "English answer (before translation)", open=False, visible=False
        ) as english_panel:
            english = gr.Textbox(show_label=False, lines=5, interactive=False)
        with gr.Accordion("What the agent did", open=False):
            trace = gr.HTML()
        gr.HTML(
            f"<div id='footer'><a href='{GITHUB_URL}' target='_blank'>Code, eval set and "
            "results</a> · Content from <a href='https://medlineplus.gov' target='_blank'>"
            "MedlinePlus</a> (US National Library of Medicine) · Built with Cohere</div>"
        )

        outputs = [answer, meta, english_panel, english, trace, used]
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
        theme=THEME,
        css=CSS,
        footer_links=[],
    )
