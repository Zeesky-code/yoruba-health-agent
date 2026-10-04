"""The agent loop: a model picks one tool per step until a terminal tool ends the run.

Retries with backoff, a hard budget that degrades to a refusal instead of looping, and a
JSONL trace line per step (agent/trace.py).

    uv run python -m agent.harness "Báwo ni ibà ṣe ń ràn?"
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cohere
import httpx

from agent import tools
from agent.tools import Answer, Chunk, Refusal, call_with_rate_limit, client, track_usage
from agent.trace import Trace, cost_usd

DECIDE_MODEL = tools.CHAT_MODEL
TERMINAL = {"answer", "refuse"}
REFUSAL_REASONS = ["out_of_scope", "personal_medical", "low_confidence"]

# translate -> search -> rerank -> answer is 4 steps; one rephrased search adds 2.
MAX_STEPS = 6
BUDGET_USD = 0.02

SYSTEM_PROMPT = """You control a health-information assistant for Yorùbá speakers. Its only
knowledge is a set of English MedlinePlus pages. Call exactly one tool per turn.

1. Decide first whether to refuse:
   - Not about health or medicine -> refuse(reason="out_of_scope").
   - Asks for advice about the asker's own (or a family member's) symptoms, diagnosis,
     medicines or doses, e.g. "I have chest pain, should I take aspirin?" ->
     refuse(reason="personal_medical"). General questions such as "how is malaria
     spread?" or "what foods contain vitamin A?" are fine to answer.
2. Otherwise: translate_query the question to English, search_corpus with that English
   query, then rerank with the same English query.
3. If the best rerank score is at least 0.1 and the passages answer the question, call
   answer with the IDs of the relevant reranked passages (1 to 5 IDs).
4. If the best score is below 0.1, you may search once more with a rephrased English
   query and rerank again. If it is still low, refuse(reason="low_confidence").
Never pass answer an ID that did not appear in a rerank result."""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "translate_query",
            "description": "Translate the user's question into English for retrieval.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Embedding search over the MedlinePlus corpus. Returns the top "
            "passages with cosine scores. Search in English.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rerank",
            "description": "Rerank the results of the most recent search_corpus call. "
            "Scores are 0-1; below 0.1 means nothing relevant was found.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "top_n": {"type": "integer"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Final step: answer the user's question in Yorùbá from these "
            "passages, with citations.",
            "parameters": {
                "type": "object",
                "properties": {"chunk_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["chunk_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refuse",
            "description": "Final step: decline to answer.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string", "enum": REFUSAL_REASONS}},
                "required": ["reason"],
            },
        },
    },
]


class ToolError(Exception):
    """A tool call failed after its retries, or its arguments were invalid."""


class InvalidArgs(ToolError):
    """Bad arguments from the model: retrying the same call cannot help."""


@dataclass
class State:
    question: str
    messages: list[dict] = field(default_factory=list)
    last_search: list[Chunk] = field(default_factory=list)
    seen_reranked: dict[str, Chunk] = field(default_factory=dict)
    spent: float = 0.0


@dataclass
class RunResult:
    output: Answer | Refusal
    run_id: str
    steps: int
    cost_usd: float
    latency_ms: float  # wall time minus rate-limit waits
    tool_errors: int
    tool_calls: int
    stopped_by: str  # "terminal" | "budget" | "max_steps"


@dataclass
class Decision:
    tool: str
    args: dict
    call_id: str | None = None
    raw_message: dict | None = None


def decide_next(state: State) -> Decision:
    """One Command A tool-use call: which tool next, with what arguments."""
    if not state.messages:
        state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": state.question},
        ]
    try:
        resp = call_with_rate_limit(
            client().chat,
            model=DECIDE_MODEL,
            messages=state.messages,
            tools=TOOL_SCHEMAS,
            tool_choice="REQUIRED",
            temperature=0,
        )
    except (cohere.core.ApiError, httpx.HTTPError) as e:
        raise ToolError(f"decide_next failed: {e!r}") from e
    calls = resp.message.tool_calls or []
    if not calls:
        raise InvalidArgs("model returned no tool call")
    call = calls[0]  # one tool per step, by design
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError as e:
        raise InvalidArgs(f"tool arguments are not valid JSON: {e}") from e
    raw = {
        "role": "assistant",
        "tool_plan": resp.message.tool_plan or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
        ],
    }
    return Decision(tool=call.function.name, args=args, call_id=call.id, raw_message=raw)


def _execute(tool: str, args: dict, state: State):
    """Run one tool and return (python result, compact observation for the model)."""
    if tool == "translate_query":
        text = tools.translate_query(args["text"], target="en")
        return text, {"translation": text}
    if tool == "search_corpus":
        hits = tools.search_corpus(args["query"], k=int(args.get("k", 50)))
        state.last_search = hits
        top = [{"id": c.id, "topic": c.topic, "score": round(c.score, 3)} for c in hits[:10]]
        return hits, {"n_results": len(hits), "top_10": top}
    if tool == "rerank":
        if not state.last_search:
            raise InvalidArgs("rerank called before any search_corpus")
        ranked = tools.rerank(args["query"], state.last_search, top_n=int(args.get("top_n", 5)))
        state.seen_reranked.update({c.id: c for c in ranked})
        results = [
            {"id": c.id, "topic": c.topic, "score": round(c.score, 3), "text": c.text[:300]}
            for c in ranked
        ]
        return ranked, {"results": results}
    if tool == "answer":
        ids = args.get("chunk_ids") or []
        unknown = [i for i in ids if i not in state.seen_reranked]
        if not ids or unknown:
            raise InvalidArgs(f"answer needs IDs from a rerank result; unknown: {unknown}")
        result = tools.answer([state.seen_reranked[i] for i in ids], state.question)
        return result, None
    if tool == "refuse":
        if args.get("reason") not in REFUSAL_REASONS:
            raise InvalidArgs(f"reason must be one of {REFUSAL_REASONS}")
        return tools.refuse(args["reason"]), None
    raise InvalidArgs(f"unknown tool {tool!r}")


def dispatch(tool: str, args: dict, state: State, retries: int = 2, backoff: float = 1.5):
    """Run a tool, retrying transient failures with exponential backoff.

    Invalid arguments fail immediately: the same call would fail again, so the error goes
    back to the model instead. Rate limits are already waited out in call_with_rate_limit.
    """
    for attempt in range(retries + 1):
        try:
            return _execute(tool, args, state)
        except InvalidArgs:
            raise
        except (KeyError, TypeError, ValueError) as e:
            raise InvalidArgs(f"bad arguments for {tool}: {e!r}") from e
        except Exception as e:
            if attempt == retries:
                raise ToolError(f"{tool} failed after {retries + 1} attempts: {e!r}") from e
            time.sleep(backoff**attempt)


def _summary(tool: str, result) -> dict:
    """Trace fields describing a tool's result (the full result is not logged)."""
    if tool in ("search_corpus", "rerank"):
        top = round(result[0].score, 4) if result else None
        return {"n_results": len(result), "top_score": top}
    if tool == "translate_query":
        return {"translation": result}
    if tool == "answer":
        return {"n_citations": len(result.citations)}
    if tool == "refuse":
        return {"reason": result.reason}
    return {}


def run(
    question: str,
    max_steps: int = MAX_STEPS,
    budget_usd: float = BUDGET_USD,
    trace_path: Path | None = None,
    decide: Callable[[State], Decision] = decide_next,
    execute: Callable = dispatch,
) -> RunResult:
    run_id = uuid.uuid4().hex[:12]
    trace = Trace(run_id, trace_path)
    state = State(question=question)
    started = time.perf_counter()
    waited_ms = 0.0
    errors = calls = 0

    def finish(output, steps: int, stopped_by: str) -> RunResult:
        latency = (time.perf_counter() - started) * 1000 - waited_ms
        return RunResult(output, run_id, steps, state.spent, latency, errors, calls, stopped_by)

    for step in range(max_steps):
        if state.spent > budget_usd:
            trace.log(
                step,
                "refuse",
                {"reason": "low_confidence"},
                stopped_by="budget",
                spent_usd=round(state.spent, 6),
                error=None,
            )
            return finish(tools.refuse("low_confidence"), step, "budget")

        step_start = time.perf_counter()
        with track_usage() as usage:
            try:
                decision = decide(state)
            except ToolError as e:
                decision, decide_error = None, str(e)
            else:
                decide_error = None
            result, error, obs = None, decide_error, None
            if decision is not None:
                calls += 1
                try:
                    result, obs = execute(decision.tool, decision.args, state)
                except ToolError as e:
                    error = str(e)

        step_cost = cost_usd(usage)
        step_wait = sum(u["wait_ms"] for u in usage)
        state.spent += step_cost
        waited_ms += step_wait
        tool = decision.tool if decision else "decide_next"
        trace.log(
            step,
            tool,
            decision.args if decision else {},
            **(_summary(tool, result) if error is None and decision else {}),
            latency_ms=round((time.perf_counter() - step_start) * 1000 - step_wait),
            rate_limit_wait_ms=step_wait,
            tokens_in=sum(u["tokens_in"] for u in usage),
            tokens_out=sum(u["tokens_out"] for u in usage),
            cost_usd=round(step_cost, 6),
            error=error,
        )

        if error is not None:
            errors += 1
            if decision is not None and decision.raw_message is not None:
                state.messages.append(decision.raw_message)
                state.messages.append(_tool_message(decision.call_id, {"error": error}))
            continue  # the error is now an observation; the model gets to recover

        if decision.tool in TERMINAL:
            return finish(result, step + 1, "terminal")

        if decision.raw_message is not None:
            state.messages.append(decision.raw_message)
            state.messages.append(_tool_message(decision.call_id, obs))

    trace.log(
        max_steps,
        "refuse",
        {"reason": "low_confidence"},
        stopped_by="max_steps",
        spent_usd=round(state.spent, 6),
        error=None,
    )
    return finish(tools.refuse("low_confidence"), max_steps, "max_steps")


def _tool_message(call_id: str | None, obs: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": [
            {"type": "document", "document": {"data": json.dumps(obs, ensure_ascii=False)}}
        ],
    }


if __name__ == "__main__":
    import sys

    res = run(" ".join(sys.argv[1:]), trace_path=Path("traces/adhoc.jsonl"))
    print(res.output.text)
    print(
        f"\n[{res.stopped_by}] steps={res.steps} cost=${res.cost_usd:.4f} "
        f"latency={res.latency_ms / 1000:.1f}s errors={res.tool_errors} run_id={res.run_id}"
    )
