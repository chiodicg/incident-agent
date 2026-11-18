"""Run the agent once against a live service and print a readable trace.

    uv run python -m agent.cli --alert alerts/db_down.json --model anthropic:claude-opus-5-5

The service must already be in the scenario's state (fault active, traffic recorded). The
CLI never touches /admin: it has the same read-only access as the agent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent.loop import run_agent
from agent.models import Alert
from agent.prompts import CONDITIONS
from agent.tools import ToolBox
from agent.trace import Trace
from llm.base import create_client

RESULT_PREVIEW_CHARS = 600


def format_trace(trace: Trace, preview_chars: int = RESULT_PREVIEW_CHARS) -> str:
    """Human-readable view of a run: one block per step, then the outcome."""
    lines = [
        f"model={trace.model_id}  condition={trace.condition}  budget={trace.budget}",
        f"alert: [{trace.alert.severity}] {trace.alert.name}: {trace.alert.summary}",
        "",
    ]
    for step in trace.steps:
        args = json.dumps(step.arguments) if step.arguments else ""
        lines.append(f"── step {step.step}/{trace.budget}: {step.tool}({args})")
        if step.reasoning:
            lines.append(_indent("reasoning: " + step.reasoning))
        if step.assistant_text:
            lines.append(_indent("said: " + step.assistant_text))
        result = step.result
        if len(result) > preview_chars:
            result = result[:preview_chars] + f" … [{len(step.result)} chars]"
        lines.append(_indent(("ERROR " if step.is_error else "") + "result: " + result))
    if trace.events:
        lines += ["", "events:"] + [f"  - {e}" for e in trace.events]

    usage = trace.usage
    lines += [
        "",
        f"outcome: {trace.outcome}  steps used: {trace.steps_used}/{trace.budget}  "
        f"model calls: {len(trace.model_calls)}",
        f"tokens: in={usage.input_tokens} cache_read={usage.cache_read_tokens} "
        f"cache_write={usage.cache_write_tokens} out={usage.output_tokens}",
    ]
    if trace.diagnosis is not None:
        lines += ["", "diagnosis:"]
        if trace.submission_reasoning:
            lines.append(_indent("reasoning: " + trace.submission_reasoning))
        lines.append(trace.diagnosis.model_dump_json(indent=2))
    return "\n".join(lines)


def _indent(text: str) -> str:
    return "\n".join("   " + line for line in text.splitlines())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent.cli", description=__doc__.splitlines()[0])
    parser.add_argument("--alert", type=Path, required=True, help="alert JSON file")
    parser.add_argument("--model", required=True, help="provider:model, e.g. openai:gpt-6-luna")
    parser.add_argument("--condition", choices=CONDITIONS, default="unguided")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--service-url", default=None, help="default: $SERVICE_URL")
    parser.add_argument("--trace-out", type=Path, help="also write the full trace as JSON")
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    alert = Alert.model_validate_json(args.alert.read_text())
    service_url = args.service_url or os.environ.get("SERVICE_URL", "http://localhost:8000")

    tools = ToolBox.for_url(service_url)
    try:
        trace = await run_agent(
            alert,
            client=create_client(args.model),
            tools=tools,
            condition=args.condition,
            budget=args.budget,
        )
    finally:
        await tools.aclose()

    print(format_trace(trace))
    if args.trace_out:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        args.trace_out.write_text(trace.model_dump_json(indent=2))
        print(f"\ntrace written to {args.trace_out}")
    return 0 if trace.diagnosis is not None else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
