"""The record of one agent run: everything the eval scores and the README shows."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from agent.models import Alert, Diagnosis
from llm.types import Message, StopReason, Usage

Outcome = Literal[
    "submitted",  # agent submitted on its own
    "submitted_after_force",  # submitted only once submit_diagnosis was forced
    "invalid_output",  # submission failed validation twice
    "no_submission",  # never produced a submission (ignored forcing / call limit hit)
    "refused",  # the provider's safety system declined the request
    "error",  # infrastructure failure (e.g. provider API error); details in events
]


class ToolStep(BaseModel):
    """One executed diagnostic tool call: the unit of the step budget."""

    step: int  # 1-based
    tool: str
    arguments: dict[str, Any]
    result: str
    is_error: bool
    assistant_text: str  # what the model said in the same turn
    reasoning: str = ""  # reasoning summary from the same turn, if the provider returns one


class ModelCall(BaseModel):
    tool_choice: str
    stop_reason: StopReason
    usage: Usage


class Trace(BaseModel):
    model_id: str
    client_config: dict[str, Any]  # effective model settings (e.g. reasoning)
    condition: str
    budget: int
    alert: Alert
    system_prompt: str
    messages: list[Message] = []
    steps: list[ToolStep] = []
    model_calls: list[ModelCall] = []
    events: list[str] = []  # nudges, forcing, dropped calls, validation failures
    diagnosis: Diagnosis | None = None
    outcome: Outcome = "no_submission"

    @property
    def steps_used(self) -> int:
        return len(self.steps)

    @property
    def tool_sequence(self) -> list[str]:
        return [s.tool for s in self.steps]

    @property
    def usage(self) -> Usage:
        total = Usage(input_tokens=0, output_tokens=0)
        for call in self.model_calls:
            total = total + call.usage
        return total
