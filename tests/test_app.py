from agent.tools import Answer, Chunk, Citation, Refusal
from app import describe_step, render_answer, render_refusal, render_trace

CHUNKS = {
    "41-0": Chunk(id="41-0", topic="Tuberculosis", area="tb", url="https://m/tb", text="..."),
    "41-2": Chunk(id="41-2", topic="Tuberculosis", area="tb", url="https://m/tb", text="..."),
}


def test_answer_numbers_sources_in_order_of_first_citation():
    text = "Ọ̀kan. Èjì. Ẹ̀ta."
    cites = [
        Citation(start=0, end=6, text="Ọ̀kan.", chunk_ids=["41-2"]),
        Citation(start=7, end=11, text="Èjì.", chunk_ids=["41-0", "41-2"]),
    ]
    out = render_answer(Answer(text=text, citations=cites), CHUNKS)
    assert "Ọ̀kan.<sup class='cite'>1</sup> Èjì.<sup class='cite'>1,2</sup> Ẹ̀ta." in out
    assert out.index("41-2</span>") < out.index("41-0</span>")


def test_answer_escapes_html():
    out = render_answer(Answer(text="<script>x</script>"), CHUNKS)
    assert "<script>" not in out and "Sources" not in out


def test_refusal_shows_a_readable_badge():
    out = render_refusal(Refusal(reason="personal_medical", text="Ẹ lọ rí dókítà."))
    assert "Personal medical advice" in out and "Ẹ lọ rí dókítà." in out


def test_describe_step_in_plain_words():
    row = {"tool": "rerank", "args": {"query": "q"}, "top_score": 0.9987, "error": None}
    assert describe_step(row) == "top relevance 1.00"
    assert describe_step({"tool": "search_corpus", "args": {}, "error": "boom"}) == "error: boom"


def test_trace_table_has_a_row_per_step():
    trace = [
        {
            "step": 0,
            "tool": "refuse",
            "args": {"reason": "out_of_scope"},
            "latency_ms": 600,
            "cost_usd": 0.0016,
            "error": None,
        },
    ]
    out = render_trace(trace)
    assert out.count("<tr>") == 2 and "reason: out_of_scope" in out and "0.0016" in out
