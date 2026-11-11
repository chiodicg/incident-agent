"""A fake LLMClient that replays a script, for testing the loop and eval without API calls."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from llm.base import DEFAULT_MAX_TOKENS
from llm.types import Completion, Message, ToolChoice, ToolSpec


class RecordedCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: list[Message]
    system: str
    tools: list[ToolSpec]
    tool_choice: ToolChoice = "auto"


ScriptStep = Completion | Callable[[RecordedCall], Completion]


class ScriptExhaustedError(AssertionError):
    """The code under test made more LLM calls than the script expected."""


class ScriptedLLMClient:
    def __init__(self, script: Sequence[ScriptStep], model_id: str = "scripted:test") -> None:
        self.model_id = model_id
        self._script = list(script)
        self.calls: list[RecordedCall] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str,
        tools: list[ToolSpec],
        tool_choice: ToolChoice = "auto",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        call = RecordedCall(
            messages=list(messages), system=system, tools=list(tools), tool_choice=tool_choice
        )
        self.calls.append(call)
        if len(self.calls) > len(self._script):
            raise ScriptExhaustedError(
                f"call #{len(self.calls)} but the script has {len(self._script)} steps"
            )
        step = self._script[len(self.calls) - 1]
        return step(call) if callable(step) else step
