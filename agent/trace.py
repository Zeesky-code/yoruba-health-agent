"""Per-step JSONL traces and cost accounting.

One line per harness step, so every reported number can be recomputed from traces/.
"""

from __future__ import annotations

import json
from pathlib import Path

TRACES_DIR = Path(__file__).resolve().parent.parent / "traces"

# USD list prices, checked 2026-10-04. Cohere's pricing page no longer lists these models,
# so Command A and Rerank come from third-party trackers (pricepertoken.com,
# aipricing.guru). Tiny Aya has no published price: it is assumed to cost the same as
# Aya Expanse, the only Aya model Cohere publishes a price for.
PER_M_TOKENS_IN = {
    "command-a-03-2025": 2.50,
    "tiny-aya-global": 0.50,  # assumed, see above
    "embed-multilingual-v3.0": 0.10,
}
PER_M_TOKENS_OUT = {
    "command-a-03-2025": 10.00,
    "tiny-aya-global": 1.50,  # assumed, see above
}
PER_SEARCH_UNIT = {"rerank-multilingual-v3.0": 0.002}


def cost_usd(records: list[dict]) -> float:
    total = 0.0
    for r in records:
        model = r["model"]
        if r["tokens_in"] and model not in PER_M_TOKENS_IN:
            raise KeyError(f"no input price for {model}")
        total += r["tokens_in"] * PER_M_TOKENS_IN.get(model, 0) / 1e6
        total += r["tokens_out"] * PER_M_TOKENS_OUT.get(model, 0) / 1e6
        total += r["search_units"] * PER_SEARCH_UNIT.get(model, 0)
    return total


class Trace:
    def __init__(self, run_id: str, path: Path | None) -> None:
        self.run_id = run_id
        self.path = path
        self.steps: list[dict] = []

    def log(self, step: int, tool: str, args: dict, **fields) -> dict:
        row = {"run_id": self.run_id, "step": step, "tool": tool, "args": args, **fields}
        self.steps.append(row)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row
