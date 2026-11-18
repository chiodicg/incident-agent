"""The hand-rolled agent loop: investigate an alert under a step budget, then submit.

Rules (SPEC.md "Agent loop behavior"):
- one diagnostic tool call per turn; each costs one step; submit_diagnosis is free
- the remaining budget is appended to every tool result
- a text-only reply gets one free nudge; a second one forces submit_diagnosis
- when the budget is spent, submit_diagnosis is forced via tool_choice
- an invalid submission gets one retry, then the run is recorded as invalid_output
- infrastructure errors end the run with outcome=error and the partial trace kept
"""

from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from agent.models import Alert, Diagnosis
from agent.prompts import (
    FORCE_SUBMIT,
    NUDGE,
    Condition,
    render_alert,
    steps_footer,
    system_prompt,
)
from agent.tools import SUBMIT_TOOL, format_validation_error, tool_specs
from agent.trace import ModelCall, ToolStep, Trace
from llm.base import LLMClient, LLMProviderError
from llm.types import Completion, ForceTool, Message, ToolCallPart, ToolChoice, ToolResultPart

# Safety net against a model that loops without progressing (e.g. ignoring forced submit).
EXTRA_MODEL_CALLS = 6

NOT_EXECUTED_PARALLEL = "Not executed: one tool call per turn. Call it again on its own."
NOT_EXECUTED_FORCED = "Not executed: no more diagnostic calls. Call submit_diagnosis now."


class Tools(Protocol):
    async def run(self, call: ToolCallPart) -> ToolResultPart: ...


async def run_agent(
    alert: Alert,
    *,
    client: LLMClient,
    tools: Tools,
    condition: Condition = "unguided",
    budget: int = 8,
) -> Trace:
    """Investigate `alert` using at most `budget` diagnostic tool calls."""
    if budget < 1:
        raise ValueError("budget must be at least 1")

    trace = Trace(
        model_id=client.model_id,
        client_config=client.config,
        condition=condition,
        budget=budget,
        alert=alert,
        system_prompt=system_prompt(condition),
    )
    try:
        await _Run(trace, client, tools).execute()
    except LLMProviderError as exc:  # keep the partial trace; the eval decides on retries
        trace.outcome = "error"
        trace.diagnosis = None
        trace.events.append(f"error: {type(exc).__name__}: {exc}")
    return trace


class _Run:
    """State for one run. Kept as a small class so each rule reads as one method."""

    def __init__(self, trace: Trace, client: LLMClient, tools: Tools) -> None:
        self.trace = trace
        self.client = client
        self.tools = tools
        self.specs = tool_specs()
        # The trace holds the live history, so a run that crashes mid-way keeps what happened.
        self.messages: list[Message] = trace.messages
        self.messages.append(Message.user(render_alert(trace.alert, trace.budget)))
        self.force_reason: str | None = None  # "budget" | "no_tool_call": counts as forced
        self.retrying_submit = False  # forced too, but doesn't mark the outcome as forced
        self.nudged = False

    @property
    def forced(self) -> bool:
        return self.force_reason is not None or self.retrying_submit

    async def execute(self) -> None:
        max_calls = self.trace.budget + EXTRA_MODEL_CALLS
        while len(self.trace.model_calls) < max_calls:
            completion = await self._call_model()
            if completion.stop_reason == "refusal":
                self.trace.outcome = "refused"
                self.trace.events.append("refused: provider declined the request")
                break
            if completion.tool_calls:
                done = await self._handle_tool_calls(completion)
            else:
                done = self._handle_text_only()
            if done:
                break
        else:
            self.trace.events.append(f"stopped: model call limit ({max_calls}) reached")

    async def _call_model(self) -> Completion:
        choice: ToolChoice = ForceTool(name=SUBMIT_TOOL) if self.forced else "auto"
        completion = await self.client.complete(
            self.messages,
            system=self.trace.system_prompt,
            tools=self.specs,
            tool_choice=choice,
            parallel_tool_calls=False,
        )
        self.trace.model_calls.append(
            ModelCall(
                tool_choice=choice if isinstance(choice, str) else f"force:{choice.name}",
                stop_reason=completion.stop_reason,
                usage=completion.usage,
            )
        )
        if completion.parts:  # an empty reply can't be a message; treat it as text-only
            self.messages.append(completion.to_message())
        return completion

    def _handle_text_only(self) -> bool:
        """Nudge once, then force submit. Returns True when the run is over."""
        if self.forced:
            self.trace.events.append("no submission: model ignored forced submit_diagnosis")
            return True
        if not self.nudged:
            self.nudged = True
            self.trace.events.append("nudge: reply had no tool call")
            self.messages.append(Message.user(NUDGE))
        else:
            self.force_reason = "no_tool_call"
            self.trace.events.append("forced submit: second reply without a tool call")
            self.messages.append(Message.user(FORCE_SUBMIT))
        return False

    async def _handle_tool_calls(self, completion: Completion) -> bool:
        """Act on the first call, refuse any extras. Returns True when the run is over."""
        first, *extra = completion.tool_calls

        if first.name == SUBMIT_TOOL:
            result, done = self._accept_submission(first, completion)
            if done:
                return True
        elif self.forced:
            self.trace.events.append(f"not executed: {first.name} called while submit was forced")
            result = _refuse(first, NOT_EXECUTED_FORCED)
        else:
            result = await self._run_diagnostic(first, completion)

        results = [result]
        for call in extra:  # every call needs a result, or providers reject the next request
            self.trace.events.append(f"not executed: extra call {call.name} in the same turn")
            results.append(_refuse(call, NOT_EXECUTED_PARALLEL))
        self.messages.append(Message(role="tool", parts=results))
        return False

    async def _run_diagnostic(self, call: ToolCallPart, completion: Completion) -> ToolResultPart:
        result = await self.tools.run(call)
        self.trace.steps.append(
            ToolStep(
                step=self.trace.steps_used + 1,
                tool=call.name,
                arguments=call.arguments,
                result=result.content,
                is_error=result.is_error,
                assistant_text=completion.text,
                reasoning=completion.reasoning,
            )
        )
        remaining = self.trace.budget - self.trace.steps_used
        if remaining == 0:
            self.force_reason = "budget"
            self.trace.events.append("forced submit: step budget spent")
        # New part rather than mutation: the raw result stays in the ToolStep above.
        return result.model_copy(update={"content": f"{result.content}\n{steps_footer(remaining)}"})

    def _accept_submission(
        self, call: ToolCallPart, completion: Completion
    ) -> tuple[ToolResultPart, bool]:
        """Returns (result for the model, run is over)."""
        try:
            diagnosis = Diagnosis.model_validate(call.arguments)
        except ValidationError as exc:
            if self.retrying_submit:
                self.trace.outcome = "invalid_output"
                self.trace.events.append("invalid submission (retry also failed)")
                return _refuse(call, "Invalid submission."), True
            self.retrying_submit = True
            self.trace.events.append("invalid submission: retrying once")
            message = f"Invalid submission: {format_validation_error(exc)}. "
            return _refuse(call, message + "Fix it and call submit_diagnosis again."), False

        self.trace.diagnosis = diagnosis
        self.trace.submission_reasoning = completion.reasoning
        self.trace.outcome = "submitted_after_force" if self.force_reason else "submitted"
        return ToolResultPart(call_id=call.id, content="Diagnosis recorded."), True


def _refuse(call: ToolCallPart, message: str) -> ToolResultPart:
    return ToolResultPart(call_id=call.id, content=message, is_error=True)
