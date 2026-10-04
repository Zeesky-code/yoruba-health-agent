"""Loop mechanics with scripted decisions and fake tools: no API calls."""

import json

import pytest

from agent import harness
from agent.harness import Decision, InvalidArgs, State, ToolError, dispatch, run
from agent.tools import Answer, Refusal


def scripted(*decisions):
    it = iter(decisions)

    def decide(state: State) -> Decision:
        return next(it)

    return decide


def fake_execute(errors_on: set[int] | None = None):
    calls = []

    def execute(tool, args, state):
        calls.append(tool)
        if errors_on and len(calls) in errors_on:
            raise ToolError("boom")
        if tool == "answer":
            return Answer(text="ìdáhùn"), None
        if tool == "refuse":
            return Refusal(reason=args["reason"], text="rárá"), None
        return [], {"ok": True}

    execute.calls = calls
    return execute


def test_answer_path_stops_at_terminal_tool(tmp_path):
    trace = tmp_path / "t.jsonl"
    decide = scripted(
        Decision("translate_query", {"text": "q"}),
        Decision("search_corpus", {"query": "q"}),
        Decision("rerank", {"query": "q"}),
        Decision("answer", {"chunk_ids": ["1-0"]}),
    )
    res = run("q", decide=decide, execute=fake_execute(), trace_path=trace)
    assert isinstance(res.output, Answer)
    assert (res.steps, res.stopped_by, res.tool_errors) == (4, "terminal", 0)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [r["tool"] for r in rows] == ["translate_query", "search_corpus", "rerank", "answer"]
    assert {"run_id", "step", "args", "latency_ms", "tokens_in", "cost_usd", "error"} <= set(
        rows[0]
    )


def test_tool_error_is_logged_and_the_loop_continues(tmp_path):
    trace = tmp_path / "t.jsonl"
    decide = scripted(
        Decision("search_corpus", {"query": "q"}),
        Decision("search_corpus", {"query": "q"}),
        Decision("refuse", {"reason": "low_confidence"}),
    )
    res = run("q", decide=decide, execute=fake_execute(errors_on={1}), trace_path=trace)
    assert isinstance(res.output, Refusal)
    assert res.tool_errors == 1 and res.steps == 3
    assert json.loads(trace.read_text().splitlines()[0])["error"] == "boom"


def test_budget_exhaustion_degrades_to_refusal():
    def decide(state):
        state.spent += 1.0  # pretend each decision is expensive
        return Decision("search_corpus", {"query": "q"})

    res = run("q", budget_usd=0.5, decide=decide, execute=fake_execute())
    assert isinstance(res.output, Refusal) and res.output.reason == "low_confidence"
    assert res.stopped_by == "budget" and res.steps == 1


def test_max_steps_degrades_to_refusal():
    def decide(state):
        return Decision("search_corpus", {"query": "q"})

    res = run("q", max_steps=3, decide=decide, execute=fake_execute())
    assert res.stopped_by == "max_steps" and res.output.reason == "low_confidence"


def test_dispatch_retries_transient_errors(monkeypatch):
    attempts = []

    def flaky(tool, args, state):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("503")
        return "ok", {}

    monkeypatch.setattr(harness, "_execute", flaky)
    monkeypatch.setattr(harness.time, "sleep", lambda s: None)
    assert dispatch("search_corpus", {}, State("q"), retries=2) == ("ok", {})
    assert len(attempts) == 3


def test_dispatch_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr(harness, "_execute", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(harness.time, "sleep", lambda s: None)
    with pytest.raises(ToolError):
        dispatch("search_corpus", {}, State("q"), retries=2)


def test_invalid_args_are_not_retried(monkeypatch):
    attempts = []

    def bad(tool, args, state):
        attempts.append(1)
        raise InvalidArgs("unknown chunk id")

    monkeypatch.setattr(harness, "_execute", bad)
    with pytest.raises(InvalidArgs):
        dispatch("answer", {}, State("q"))
    assert len(attempts) == 1


def test_answer_rejects_ids_not_seen_in_rerank():
    with pytest.raises(InvalidArgs):
        harness._execute("answer", {"chunk_ids": ["315-0"]}, State("q"))
