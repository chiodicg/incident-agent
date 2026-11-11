"""Provider-neutral conversation types.

Everything outside llm/ speaks only these types. Adapters translate them to and from each
provider's wire format, so provider quirks never leak into the agent or the eval.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

# Both Anthropic and OpenAI accept tool names matching this pattern.
TOOL_NAME_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"


class _Frozen(BaseModel):
    # Immutable: a recorded trace can't be altered after the fact. Note this is shallow:
    # fields can't be reassigned, but list/dict contents can. Convention: never mutate a
    # message in place; build a new one.
    model_config = ConfigDict(frozen=True)


# --- Parts -----------------------------------------------------------------------------------


class TextPart(_Frozen):
    type: Literal["text"] = "text"
    text: str


class ToolCallPart(_Frozen):
    type: Literal["tool_call"] = "tool_call"
    id: str  # provider-issued; tool results refer back to it
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultPart(_Frozen):
    type: Literal["tool_result"] = "tool_result"
    call_id: str
    content: str  # tools return text; structured data is serialized (e.g. JSON)
    is_error: bool = False


Part = Annotated[TextPart | ToolCallPart | ToolResultPart, Field(discriminator="type")]
OutputPart = Annotated[TextPart | ToolCallPart, Field(discriminator="type")]

Role = Literal["user", "assistant", "tool"]

ALLOWED_PARTS: dict[str, tuple[type[BaseModel], ...]] = {
    "user": (TextPart,),
    "assistant": (TextPart, ToolCallPart),
    "tool": (ToolResultPart,),
}


# --- Messages --------------------------------------------------------------------------------


class Message(_Frozen):
    """One turn. Tool results get their own `tool` role so traces separate what the agent
    was told from what the tools returned."""

    role: Role
    parts: list[Part] = Field(min_length=1)

    @model_validator(mode="after")
    def _parts_match_role(self) -> Self:
        allowed = ALLOWED_PARTS[self.role]
        bad = [p.type for p in self.parts if not isinstance(p, allowed)]
        if bad:
            raise ValueError(f"{self.role!r} message cannot contain {bad} parts")
        return self

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", parts=[TextPart(text=text)])


# --- Tools -----------------------------------------------------------------------------------


class ToolSpec(_Frozen):
    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments object


class ForceTool(_Frozen):
    """Require the model to call this specific tool (e.g. submit_diagnosis at budget end)."""

    name: str = Field(pattern=TOOL_NAME_PATTERN)


ToolChoice = Literal["auto", "required"] | ForceTool


# --- Completions -----------------------------------------------------------------------------

StopReason = Literal["tool_use", "end_turn", "max_tokens", "other"]


class Usage(_Frozen):
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class Completion(_Frozen):
    parts: list[OutputPart]
    stop_reason: StopReason
    usage: Usage

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.parts if isinstance(p, ToolCallPart)]

    def to_message(self) -> Message:
        """The assistant turn to append to the conversation history."""
        return Message(role="assistant", parts=list(self.parts))
