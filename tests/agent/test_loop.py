"""Agent loop behavior, driven by a scripted LLM and fake tools (no network, no API)."""

import json
from datetime import UTC, datetime

import pytest

from agent.loop import run_agent
from agent.models import Alert
from agent.prompts import BASE, CONDITIONS, FORCE_SUBMIT, GUIDANCE, NUDGE
from agent.tools import SUBMIT_TOOL
from llm.base import LLMProviderError
from llm.scripted import ScriptedLLMClient, ScriptExhaustedError
from llm.types import (
    Completion,
    ForceTool,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)

ALERT = Alert(
    name="HighErrorRate",
    severity="critical",
    summary="5xx rate > 20% for 5m",
    fired_at=datetime(2026, 11, 2, 9, 30, tzinfo=UTC),
)
DIAGNOSIS = {
    "root_cause_category": "dependency_down_db",
    "explanation": "The orders DB refuses connections.",
    "evidence": ["/health: db down", "logs: connection refused"],
    "proposed_fix": "Restore DB connectivity; check the DB host.",
    "fix_risk": "low",
    "confidence": 0.9,
}
USAGE = Usage(input_tokens=100, output_tokens=10)
FORCED = ForceTool(name=SUBMIT_TOOL)


class FakeTools:
    def __init__(self) -> None:
        self.ran: list[str] = []

    async def run(self, call: ToolCallPart) -> ToolResultPart:
        self.ran.append(call.name)
        return ToolResultPart(call_id=call.id, content=f"{call.name} output")


_ids = iter(range(10_000))


def tool(name: str, text: str = "", **arguments) -> Completion:
    parts = [TextPart(text=text)] if text else []
    parts.append(ToolCallPart(id=f"c{next(_ids)}", name=name, arguments=arguments))
    return Completion(parts=parts, stop_reason="tool_use", usage=USAGE)


def tools(*names: str) -> Completion:
    parts = [ToolCallPart(id=f"c{next(_ids)}", name=n) for n in names]
    return Completion(parts=parts, stop_reason="tool_use", usage=USAGE)


def submit(**overrides) -> Completion:
    return tool(SUBMIT_TOOL, **{**DIAGNOSIS, **overrides})


def say(text: str) -> Completion:
    return Completion(parts=[TextPart(text=text)], stop_reason="end_turn", usage=USAGE)


async def run(script, budget: int = 8, condition: str = "unguided", fake: FakeTools | None = None):
    client = ScriptedLLMClient(script)
    trace = await run_agent(
        ALERT, client=client, tools=fake or FakeTools(), condition=condition, budget=budget
    )
    return trace, client


# --- Happy path ------------------------------------------------------------------------------


async def test_investigates_then_submits() -> None:
    fake = FakeTools()
    trace, client = await run(
        [tool("check_health", text="Health first."), tool("get_logs", level="ERROR"), submit()],
        fake=fake,
    )

    assert trace.outcome == "submitted"
    assert trace.diagnosis is not None
    assert trace.diagnosis.root_cause_category == "dependency_down_db"
    assert trace.tool_sequence == ["check_health", "get_logs"] == fake.ran
    assert trace.steps[0].assistant_text == "Health first."
    assert trace.steps[1].arguments == {"level": "ERROR"}
    assert len(client.calls) == 3


async def test_conversation_shape_and_budget_footer() -> None:
    trace, _ = await run([tool("check_health"), submit()], budget=4)

    assert [m.role for m in trace.messages] == ["user", "assistant", "tool", "assistant"]
    result = trace.messages[2].parts[0]
    assert result.content == "check_health output\n[steps remaining: 3]"
    assert trace.steps[0].result == "check_health output"  # trace keeps the raw result


async def test_first_message_is_the_rendered_alert() -> None:
    _, client = await run([submit()], budget=4)
    assert client.calls[0].messages[0].parts[0].text == (
        "ALERT [critical] HighErrorRate, fired 2026-11-02T09:30:00Z\n"
        "5xx rate > 20% for 5m\n"
        "Step budget: 4 diagnostic tool calls."
    )


async def test_every_call_disables_parallel_tools_and_sends_all_specs() -> None:
    _, client = await run([tool("check_health"), submit()])
    for call in client.calls:
        assert call.parallel_tool_calls is False
        assert [t.name for t in call.tools][-1] == SUBMIT_TOOL
        assert len(call.tools) == 5


async def test_submitting_immediately_costs_no_steps() -> None:
    trace, _ = await run([submit()])
    assert trace.outcome == "submitted"
    assert trace.steps_used == 0


async def test_usage_is_summed_across_calls() -> None:
    trace, _ = await run([tool("check_health"), tool("get_logs"), submit()])
    assert trace.usage == Usage(input_tokens=300, output_tokens=30)


# --- Budget ----------------------------------------------------------------------------------


async def test_budget_exhaustion_forces_submit() -> None:
    fake = FakeTools()
    trace, client = await run(
        [tool("get_deploys"), tool("check_health"), submit()], budget=2, fake=fake
    )

    assert fake.ran == ["get_deploys", "check_health"]
    assert [c.tool_choice for c in client.calls] == ["auto", "auto", FORCED]
    assert (
        trace.messages[-2]
        .parts[0]
        .content.endswith("[steps remaining: 0. Call submit_diagnosis now.]")
    )
    assert trace.outcome == "submitted_after_force"


async def test_diagnostic_call_while_forced_is_not_executed() -> None:
    """Some models ignore tool_choice; the budget must still hold."""
    fake = FakeTools()
    trace, _ = await run([tool("get_deploys"), tool("get_logs"), submit()], budget=1, fake=fake)

    assert fake.ran == ["get_deploys"]
    assert trace.steps_used == 1
    refused = trace.messages[4].parts[0]
    assert refused.is_error
    assert "submit_diagnosis" in refused.content
    assert trace.outcome == "submitted_after_force"


# --- One tool per turn -----------------------------------------------------------------------


async def test_extra_calls_in_one_turn_are_refused_at_no_cost() -> None:
    fake = FakeTools()
    trace, _ = await run([tools("check_health", "get_logs", "get_metrics"), submit()], fake=fake)

    assert fake.ran == ["check_health"]
    assert trace.steps_used == 1
    results = trace.messages[2].parts
    assert len(results) == 3  # every call gets a result, as providers require
    assert not results[0].is_error
    assert all(r.is_error and "one tool call per turn" in r.content for r in results[1:])
    assert any("not executed" in e for e in trace.events)


# --- Text-only replies -----------------------------------------------------------------------


async def test_text_only_reply_gets_one_free_nudge() -> None:
    fake = FakeTools()
    trace, client = await run([say("Hmm."), tool("check_health"), submit()], fake=fake)

    assert client.calls[1].messages[-1].parts[0].text == NUDGE
    assert client.calls[1].tool_choice == "auto"
    assert trace.steps_used == 1
    assert trace.outcome == "submitted"


async def test_second_text_only_reply_forces_submit() -> None:
    trace, client = await run([say("Hmm."), say("Still thinking."), submit()])

    assert client.calls[2].messages[-1].parts[0].text == FORCE_SUBMIT
    assert client.calls[2].tool_choice == FORCED
    assert trace.outcome == "submitted_after_force"


async def test_empty_reply_is_treated_as_text_only() -> None:
    empty = Completion(parts=[], stop_reason="end_turn", usage=USAGE)
    trace, _ = await run([empty, submit()])

    assert [m.role for m in trace.messages][:2] == ["user", "user"]  # no empty assistant turn
    assert trace.outcome == "submitted"


# --- Submission validation -------------------------------------------------------------------


async def test_invalid_submission_gets_one_retry() -> None:
    trace, client = await run([submit(confidence=7), submit()])

    retry_prompt = client.calls[1].messages[-1].parts[0]
    assert retry_prompt.is_error
    assert "confidence" in retry_prompt.content
    assert client.calls[1].tool_choice == FORCED
    assert trace.outcome == "submitted"
    assert trace.diagnosis is not None


async def test_two_invalid_submissions_is_invalid_output() -> None:
    trace, _ = await run([submit(root_cause_category="gremlins"), submit(evidence=[])])
    assert trace.outcome == "invalid_output"
    assert trace.diagnosis is None


async def test_model_that_never_submits_ends_as_no_submission() -> None:
    trace, client = await run([say("a"), say("b"), say("c")])
    assert trace.outcome == "no_submission"
    assert len(client.calls) == 3


# --- Conditions ------------------------------------------------------------------------------


async def test_conditions_change_only_the_system_prompt_guidance() -> None:
    recorded = {}
    for condition in CONDITIONS:
        trace, client = await run([tool("check_health"), submit()], condition=condition)
        assert client.calls[0].system == trace.system_prompt
        recorded[condition] = client.calls[0]

    tool_json = {
        c: json.dumps([t.model_dump() for t in call.tools], sort_keys=True)
        for c, call in recorded.items()
    }
    assert len(set(tool_json.values())) == 1  # byte-identical tools across conditions

    assert recorded["unguided"].system == BASE
    for condition in ("soft_prior", "hard_procedure"):
        assert recorded[condition].system == f"{BASE}\n\n{GUIDANCE[condition]}"


async def test_trace_records_run_metadata() -> None:
    trace, _ = await run([submit()], budget=4, condition="soft_prior")
    assert (trace.model_id, trace.condition, trace.budget) == ("scripted:test", "soft_prior", 4)
    assert trace.client_config == {"scripted": True}
    assert trace.alert == ALERT


async def test_rejects_nonpositive_budget() -> None:
    with pytest.raises(ValueError):
        await run([submit()], budget=0)


async def test_provider_error_returns_partial_trace() -> None:
    def explode(call):
        raise LLMProviderError("provider down")

    trace, _ = await run([tool("check_health"), explode], budget=4)

    assert trace.outcome == "error"
    assert trace.diagnosis is None
    assert trace.events[-1] == "error: LLMProviderError: provider down"
    assert [m.role for m in trace.messages] == ["user", "assistant", "tool"]
    assert trace.tool_sequence == ["check_health"]


async def test_bugs_are_not_swallowed_as_provider_errors() -> None:
    with pytest.raises(ScriptExhaustedError):
        await run([tool("check_health")])  # script ends before the agent submits


async def test_refusal_ends_the_run() -> None:
    refusal = Completion(parts=[], stop_reason="refusal", usage=USAGE)
    trace, client = await run([tool("check_health"), refusal])
    assert trace.outcome == "refused"
    assert trace.diagnosis is None
    assert len(client.calls) == 2


async def test_reasoning_summary_is_recorded_per_step() -> None:
    from llm.types import ReasoningPart

    thinking = ReasoningPart(provider="anthropic", text="DB errors likely.", data={"x": 1})
    step = Completion(
        parts=[thinking, ToolCallPart(id="r1", name="check_health")],
        stop_reason="tool_use",
        usage=USAGE,
    )
    trace, client = await run([step, submit()])
    assert trace.steps[0].reasoning == "DB errors likely."
    assert client.calls[1].messages[1].parts[0] == thinking  # replayed unchanged
