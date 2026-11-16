"""Anthropic Messages API adapter.

Translation rules worth knowing:
- `tool` messages become `user` messages of tool_result blocks (Anthropic has no tool role).
- Thinking blocks come back as ReasoningPart and are replayed unchanged: the API requires an
  append-only history with thinking blocks intact.
- Claude Opus 5.5 / Sonnet 5.5 / Fable 5.1 / Mythos 5.1 reject forced tool choice (400). For
  them ForceTool/"required" degrade to "auto"; the agent loop already says in text which tool
  to call and refuses anything else, so the budget still holds.
- Reasoning runs at provider defaults (adaptive thinking, default effort). `display:
  "summarized"` only makes the reasoning visible in traces; it doesn't change behavior.
- No server-side refusal fallbacks: they would silently swap models mid-eval.
"""

from __future__ import annotations

from typing import Any

import anthropic
from anthropic.types import Message as SDKMessage

from llm.base import DEFAULT_MAX_TOKENS, LLMProviderError, ModelRef
from llm.types import (
    Completion,
    ForceTool,
    Message,
    ReasoningPart,
    StopReason,
    TextPart,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    ToolSpec,
    Usage,
)

PROVIDER = "anthropic"
MAX_RETRIES = 4  # SDK retries 408/409/429/5xx and connection errors with backoff

THINKING = {"type": "adaptive", "display": "summarized"}
FORCED_TOOL_CHOICE_UNSUPPORTED = frozenset(
    {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"}
)
# Informational only (recorded in traces); we never send `effort`.
DEFAULT_EFFORT = {
    "claude-opus-5-5": "medium",
    "claude-sonnet-5-5": "high",
    "claude-haiku-5-5": "medium",
}

STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


class AnthropicClient:
    def __init__(self, model: str, *, sdk: Any | None = None) -> None:
        self.model_id = f"{PROVIDER}:{model}"
        self._model = model
        self._sdk = sdk or anthropic.AsyncAnthropic(max_retries=MAX_RETRIES)
        self._native_forced = model not in FORCED_TOOL_CHOICE_UNSUPPORTED
        self.config: dict[str, Any] = {
            "thinking": THINKING,
            "effort": f"provider default ({DEFAULT_EFFORT.get(model, 'unknown')})",
            "forced_tool_choice": "native" if self._native_forced else "emulated (auto + prompt)",
            "prompt_caching": "automatic",
            "server_side_fallbacks": False,
            "max_retries": MAX_RETRIES,
        }

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str,
        tools: list[ToolSpec],
        tool_choice: ToolChoice = "auto",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        parallel_tool_calls: bool = False,
    ) -> Completion:
        request = build_request(
            self._model,
            messages,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            parallel_tool_calls=parallel_tool_calls,
            native_forced_tool_choice=self._native_forced,
        )
        try:
            response = await self._sdk.messages.create(**request)
        except anthropic.APIConnectionError as exc:  # includes timeouts
            raise LLMProviderError(f"{self.model_id}: {type(exc).__name__}: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:  # still failing after retries
                raise LLMProviderError(f"{self.model_id}: HTTP {exc.status_code}: {exc}") from exc
            raise  # 4xx: our request or credentials are wrong; fail loudly
        return parse_response(response)


def create(ref: ModelRef) -> AnthropicClient:
    return AnthropicClient(ref.model)


# --- Request ---------------------------------------------------------------------------------


def build_request(
    model: str,
    messages: list[Message],
    *,
    system: str,
    tools: list[ToolSpec],
    tool_choice: ToolChoice,
    max_tokens: int,
    parallel_tool_calls: bool,
    native_forced_tool_choice: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": to_anthropic_messages(messages),
        "tools": [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ],
        "tool_choice": to_tool_choice(
            tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            native_forced=native_forced_tool_choice,
        ),
        "thinking": THINKING,
        "cache_control": {"type": "ephemeral"},  # system + tools + history repeat every turn
    }


def to_tool_choice(
    choice: ToolChoice, *, parallel_tool_calls: bool, native_forced: bool
) -> dict[str, Any]:
    if isinstance(choice, ForceTool):
        out: dict[str, Any] = (
            {"type": "tool", "name": choice.name} if native_forced else {"type": "auto"}
        )
    elif choice == "required":
        out = {"type": "any"} if native_forced else {"type": "auto"}
    else:
        out = {"type": "auto"}
    if not parallel_tool_calls:
        out["disable_parallel_tool_use"] = True
    return out


def to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = "assistant" if msg.role == "assistant" else "user"
        blocks = [b for part in msg.parts if (b := _to_block(part)) is not None]
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})
    return out


def _to_block(part: Any) -> dict[str, Any] | None:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ToolCallPart):
        return {"type": "tool_use", "id": part.id, "name": part.name, "input": part.arguments}
    if isinstance(part, ToolResultPart):
        return {
            "type": "tool_result",
            "tool_use_id": part.call_id,
            "content": part.content,
            "is_error": part.is_error,
        }
    if isinstance(part, ReasoningPart):
        return dict(part.data) if part.provider == PROVIDER else None
    raise TypeError(f"unsupported part {type(part).__name__}")


# --- Response --------------------------------------------------------------------------------


def parse_response(response: SDKMessage) -> Completion:
    parts: list[TextPart | ToolCallPart | ReasoningPart] = []
    for block in response.content:
        if block.type == "text":
            parts.append(TextPart(text=block.text))
        elif block.type == "tool_use":
            parts.append(ToolCallPart(id=block.id, name=block.name, arguments=dict(block.input)))
        else:  # thinking, redacted_thinking, and anything newer: keep verbatim for replay
            parts.append(
                ReasoningPart(
                    provider=PROVIDER,
                    text=getattr(block, "thinking", "") or "",
                    data=block.model_dump(mode="json", exclude_none=True),
                )
            )

    usage = response.usage
    return Completion(
        parts=parts,
        stop_reason=STOP_REASONS.get(response.stop_reason or "", "other"),
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
        ),
    )
