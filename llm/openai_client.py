"""OpenAI Responses API adapter (also used for OpenAI-compatible endpoints via base_url).

Why the Responses API and not Chat Completions: it is the only OpenAI API that returns
reasoning items we can replay on later turns. Chat Completions drops a reasoning model's
reasoning between calls, which would handicap GPT models against Claude (whose thinking
blocks we replay) and confound the model comparison.

Translation rules worth knowing:
- Stateless: `store=False`, full history sent every call. Reasoning items come back with
  `encrypted_content` and are replayed unchanged as ReasoningPart(provider="openai").
- `tool` messages become `function_call_output` items. The API has no error flag, so error
  results are prefixed with ERROR_PREFIX.
- Function tools are sent with `strict=False`: our schemas have optional arguments, which
  strict mode disallows. Arguments are validated by the tools anyway.
- Reasoning effort stays at the provider default. `summary: "auto"` only makes reasoning
  summaries visible in traces.
- OpenAI-compatible servers (Ollama, vLLM, OpenRouter, ...) must implement /v1/responses.
  OpenAI-only options (reasoning summaries, encrypted reasoning) are not sent to them.
"""

from __future__ import annotations

import json
import os
from typing import Any

import openai
from openai.types.responses import Response

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

PROVIDER = "openai"
COMPAT_PROVIDER = "openai-compat"
MAX_RETRIES = 4  # SDK retries 408/409/429/5xx and connection errors with backoff
ERROR_PREFIX = "ERROR: "
REASONING = {"summary": "auto"}  # visibility only; effort is left at the provider default
COMPAT_DEFAULT_KEY = "not-needed"  # local servers ignore the key, but the SDK requires one


class OpenAIClient:
    def __init__(
        self,
        model: str,
        *,
        provider: str = PROVIDER,
        base_url: str | None = None,
        sdk: Any | None = None,
    ) -> None:
        self.model_id = f"{provider}:{model}" + (f"@{base_url}" if base_url else "")
        self._model = model
        self._openai_only = provider == PROVIDER
        if sdk is None:
            api_key = (
                None
                if self._openai_only
                else os.environ.get("OPENAI_COMPAT_API_KEY") or COMPAT_DEFAULT_KEY
            )
            sdk = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=MAX_RETRIES)
        self._sdk = sdk
        self.config: dict[str, Any] = {
            "api": "responses",
            "reasoning": REASONING if self._openai_only else "not sent",
            "effort": "provider default",
            "store": False,
            "forced_tool_choice": "native",
            "prompt_caching": "automatic",
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
            openai_only_options=self._openai_only,
        )
        try:
            response = await self._sdk.responses.create(**request)
        except openai.APIConnectionError as exc:  # includes timeouts
            raise LLMProviderError(f"{self.model_id}: {type(exc).__name__}: {exc}") from exc
        except openai.APIStatusError as exc:
            if exc.status_code == 429 or exc.status_code >= 500:  # still failing after retries
                raise LLMProviderError(f"{self.model_id}: HTTP {exc.status_code}: {exc}") from exc
            raise  # 4xx: our request or credentials are wrong; fail loudly
        return parse_response(response)


def create(ref: ModelRef) -> OpenAIClient:
    return OpenAIClient(ref.model, provider=ref.provider, base_url=ref.base_url)


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
    openai_only_options: bool = True,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "instructions": system,
        "input": to_input_items(messages),
        "tools": [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": False,
            }
            for t in tools
        ],
        "tool_choice": to_tool_choice(tool_choice),
        "parallel_tool_calls": parallel_tool_calls,
        "max_output_tokens": max_tokens,
        "store": False,
    }
    if openai_only_options:
        request["reasoning"] = REASONING
        request["include"] = ["reasoning.encrypted_content"]  # needed to replay when store=False
    return request


def to_tool_choice(choice: ToolChoice) -> Any:
    if isinstance(choice, ForceTool):
        return {"type": "function", "name": choice.name}
    return choice  # "auto" and "required" have the same names in the Responses API


def to_input_items(messages: list[Message]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for msg in messages:
        for part in msg.parts:
            item = _to_item(msg.role, part)
            if item is not None:
                items.append(item)
    return items


def _to_item(role: str, part: Any) -> dict[str, Any] | None:
    if isinstance(part, TextPart):
        return {"role": role, "content": part.text}
    if isinstance(part, ToolCallPart):
        return {
            "type": "function_call",
            "call_id": part.id,
            "name": part.name,
            "arguments": json.dumps(part.arguments),
        }
    if isinstance(part, ToolResultPart):
        content = f"{ERROR_PREFIX}{part.content}" if part.is_error else part.content
        return {"type": "function_call_output", "call_id": part.call_id, "output": content}
    if isinstance(part, ReasoningPart):
        return dict(part.data) if part.provider == PROVIDER else None
    raise TypeError(f"unsupported part {type(part).__name__}")


# --- Response --------------------------------------------------------------------------------


def parse_response(response: Response) -> Completion:
    parts: list[TextPart | ToolCallPart | ReasoningPart] = []
    refused = False
    for item in response.output:
        if item.type == "message":
            for content in item.content:
                if content.type == "output_text":
                    parts.append(TextPart(text=content.text))
                elif content.type == "refusal":
                    refused = True
                    parts.append(TextPart(text=content.refusal))
        elif item.type == "function_call":
            parts.append(
                ToolCallPart(id=item.call_id, name=item.name, arguments=_parse_arguments(item))
            )
        elif item.type == "reasoning":
            parts.append(
                ReasoningPart(
                    provider=PROVIDER,
                    text="\n".join(s.text for s in item.summary),
                    data=item.model_dump(mode="json", exclude_none=True),
                )
            )
        # Other item types are server-side tools we never enable; ignore them.

    return Completion(
        parts=parts,
        stop_reason=_stop_reason(response, refused, parts),
        usage=_usage(response),
    )


def _parse_arguments(item: Any) -> dict[str, Any]:
    """Arguments arrive as a JSON string. Malformed JSON is passed on as a single key so the
    tool's own validation reports it to the model, instead of crashing the run."""
    try:
        parsed = json.loads(item.arguments or "{}")
    except json.JSONDecodeError:
        return {"_unparseable_arguments": item.arguments}
    return parsed if isinstance(parsed, dict) else {"_unparseable_arguments": item.arguments}


def _stop_reason(response: Response, refused: bool, parts: list[Any]) -> StopReason:
    reason = response.incomplete_details.reason if response.incomplete_details else None
    if refused or reason == "content_filter":
        return "refusal"
    if reason == "max_output_tokens":
        return "max_tokens"
    if response.status != "completed":
        return "other"
    if any(isinstance(p, ToolCallPart) for p in parts):
        return "tool_use"
    return "end_turn"


def _usage(response: Response) -> Usage:
    u = response.usage
    if u is None:
        return Usage(input_tokens=0, output_tokens=0)
    details = u.input_tokens_details
    cached = details.cached_tokens if details else 0
    written = getattr(details, "cache_write_tokens", 0) or 0
    # OpenAI's input_tokens includes cached tokens; ours counts uncached input only.
    return Usage(
        input_tokens=max(u.input_tokens - cached - written, 0),
        output_tokens=u.output_tokens,
        cache_read_tokens=cached,
        cache_write_tokens=written,
    )
