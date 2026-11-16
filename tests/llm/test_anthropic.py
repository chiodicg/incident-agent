"""Anthropic adapter: neutral <-> Messages API translation. No network except `-m live`."""

import os
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message as SDKMessage

from llm.anthropic_client import (
    AnthropicClient,
    build_request,
    parse_response,
    to_anthropic_messages,
    to_tool_choice,
)
from llm.base import LLMProviderError, create_client
from llm.types import (
    ForceTool,
    Message,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
    Usage,
)

TOOLS = [
    ToolSpec(
        name="check_health",
        description="Read the health endpoint.",
        parameters={"type": "object", "properties": {}},
    )
]
THINKING_BLOCK = {"type": "thinking", "thinking": "Check health first.", "signature": "sig-abc"}


def sdk_message(content: list[dict], stop_reason: str = "tool_use", **usage: int) -> SDKMessage:
    return SDKMessage.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("input", 100),
                "output_tokens": usage.get("output", 20),
                "cache_read_input_tokens": usage.get("cache_read", 0),
                "cache_creation_input_tokens": usage.get("cache_write", 0),
            },
        }
    )


# --- Request building ------------------------------------------------------------------------


def test_build_request_shape() -> None:
    req = build_request(
        "claude-opus-5-5",
        [Message.user("alert")],
        system="You are on call.",
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=16000,
        parallel_tool_calls=False,
        native_forced_tool_choice=False,
    )
    assert req["model"] == "claude-opus-5-5"
    assert req["system"] == "You are on call."
    assert req["max_tokens"] == 16000
    assert req["tools"] == [
        {
            "name": "check_health",
            "description": "Read the health endpoint.",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]
    assert req["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert req["cache_control"] == {"type": "ephemeral"}
    assert "output_config" not in req  # effort stays at the provider default
    assert req["messages"] == [{"role": "user", "content": [{"type": "text", "text": "alert"}]}]


@pytest.mark.parametrize(
    "choice, native, parallel, expected",
    [
        ("auto", True, False, {"type": "auto", "disable_parallel_tool_use": True}),
        ("auto", True, True, {"type": "auto"}),
        ("required", True, False, {"type": "any", "disable_parallel_tool_use": True}),
        (
            ForceTool(name="submit_diagnosis"),
            True,
            False,
            {"type": "tool", "name": "submit_diagnosis", "disable_parallel_tool_use": True},
        ),
        # Opus 5.5 / Sonnet 5.5 reject forced choice: degrade to auto; the loop steers by prompt
        (
            ForceTool(name="submit_diagnosis"),
            False,
            False,
            {"type": "auto", "disable_parallel_tool_use": True},
        ),
        ("required", False, False, {"type": "auto", "disable_parallel_tool_use": True}),
    ],
)
def test_tool_choice(choice: Any, native: bool, parallel: bool, expected: dict) -> None:
    assert to_tool_choice(choice, parallel_tool_calls=parallel, native_forced=native) == expected


def test_assistant_and_tool_messages_translate() -> None:
    history = [
        Message.user("alert"),
        Message(
            role="assistant",
            parts=[
                ReasoningPart(
                    provider="anthropic", text="Check health first.", data=THINKING_BLOCK
                ),
                TextPart(text="Checking health."),
                ToolCallPart(id="toolu_1", name="check_health", arguments={}),
            ],
        ),
        Message(
            role="tool",
            parts=[ToolResultPart(call_id="toolu_1", content="db down", is_error=False)],
        ),
    ]
    assert to_anthropic_messages(history)[1:] == [
        {
            "role": "assistant",
            "content": [
                THINKING_BLOCK,  # replayed byte-for-byte
                {"type": "text", "text": "Checking health."},
                {"type": "tool_use", "id": "toolu_1", "name": "check_health", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "db down",
                    "is_error": False,
                }
            ],
        },
    ]


def test_reasoning_from_other_providers_is_skipped() -> None:
    msg = Message(
        role="assistant",
        parts=[
            ReasoningPart(provider="openai", data={"type": "reasoning", "id": "rs_1"}),
            TextPart(text="hi"),
        ],
    )
    assert to_anthropic_messages([Message.user("x"), msg])[1]["content"] == [
        {"type": "text", "text": "hi"}
    ]


def test_consecutive_same_role_messages_are_merged() -> None:
    """The loop can emit user,user (e.g. empty reply then a nudge)."""
    merged = to_anthropic_messages([Message.user("alert"), Message.user("nudge")])
    assert merged == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "alert"}, {"type": "text", "text": "nudge"}],
        }
    ]


# --- Response parsing ------------------------------------------------------------------------


def test_parse_response_with_thinking_text_and_tool_call() -> None:
    completion = parse_response(
        sdk_message(
            [
                THINKING_BLOCK,
                {"type": "text", "text": "Looking."},
                {"type": "tool_use", "id": "toolu_9", "name": "check_health", "input": {}},
            ],
            cache_read=40,
            cache_write=7,
        )
    )
    assert completion.stop_reason == "tool_use"
    assert completion.text == "Looking."
    assert completion.reasoning == "Check health first."
    assert completion.parts[0].data == THINKING_BLOCK
    assert completion.tool_calls == [ToolCallPart(id="toolu_9", name="check_health", arguments={})]
    assert completion.usage == Usage(
        input_tokens=100, output_tokens=20, cache_read_tokens=40, cache_write_tokens=7
    )


@pytest.mark.parametrize(
    "sdk_reason, neutral",
    [
        ("end_turn", "end_turn"),
        ("tool_use", "tool_use"),
        ("max_tokens", "max_tokens"),
        ("refusal", "refusal"),
        ("stop_sequence", "other"),
        ("pause_turn", "other"),
    ],
)
def test_stop_reasons_are_normalized(sdk_reason: str, neutral: str) -> None:
    content = [{"type": "text", "text": "x"}]
    assert parse_response(sdk_message(content, stop_reason=sdk_reason)).stop_reason == neutral


def test_parsed_completion_round_trips_into_next_request() -> None:
    """Append-only history: what came back is sent back unchanged."""
    content = [
        THINKING_BLOCK,
        {"type": "tool_use", "id": "toolu_9", "name": "check_health", "input": {}},
    ]
    completion = parse_response(sdk_message(content))
    sent = to_anthropic_messages([Message.user("a"), completion.to_message()])[1]["content"]
    assert sent == content


# --- Client ----------------------------------------------------------------------------------


class FakeMessages:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.kwargs: dict | None = None

    async def create(self, **kwargs: Any) -> SDKMessage:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeSDK:
    def __init__(self, result: Any) -> None:
        self.messages = FakeMessages(result)


async def test_client_sends_request_and_parses_response() -> None:
    sdk = FakeSDK(sdk_message([{"type": "text", "text": "done"}], stop_reason="end_turn"))
    client = AnthropicClient("claude-opus-5-5", sdk=sdk)

    completion = await client.complete(
        [Message.user("alert")], system="s", tools=TOOLS, tool_choice=ForceTool(name="check_health")
    )

    assert completion.text == "done"
    assert sdk.messages.kwargs["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert client.model_id == "anthropic:claude-opus-5-5"


def test_client_config_records_effective_settings() -> None:
    config = AnthropicClient("claude-opus-5-5", sdk=FakeSDK(None)).config
    assert config["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert config["effort"] == "provider default (medium)"
    assert config["forced_tool_choice"] == "emulated (auto + prompt)"
    assert config["server_side_fallbacks"] is False


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(cls: type, status: int) -> Exception:
    return cls("boom", response=httpx2.Response(status, request=_request()), body=None)


@pytest.mark.parametrize(
    "error",
    [
        _status_error(anthropic.RateLimitError, 429),
        _status_error(anthropic.InternalServerError, 500),
        _status_error(anthropic.APIStatusError, 529),
        anthropic.APIConnectionError(request=_request()),
        anthropic.APITimeoutError(request=_request()),
    ],
    ids=["429", "500", "529-overloaded", "connection", "timeout"],
)
async def test_infrastructure_errors_become_provider_errors(error: Exception) -> None:
    client = AnthropicClient("claude-opus-5-5", sdk=FakeSDK(error))
    with pytest.raises(LLMProviderError):
        await client.complete([Message.user("x")], system="s", tools=TOOLS)


@pytest.mark.parametrize(
    "error",
    [
        _status_error(anthropic.BadRequestError, 400),
        _status_error(anthropic.AuthenticationError, 401),
        _status_error(anthropic.NotFoundError, 404),
    ],
    ids=["400", "401", "404"],
)
async def test_configuration_errors_propagate_unchanged(error: Exception) -> None:
    """A malformed request or bad key is a bug to fix, not a run outcome."""
    client = AnthropicClient("claude-opus-5-5", sdk=FakeSDK(error))
    with pytest.raises(type(error)):
        await client.complete([Message.user("x")], system="s", tools=TOOLS)


def test_create_client_resolves_the_adapter() -> None:
    client = create_client("anthropic:claude-sonnet-5-5")
    assert isinstance(client, AnthropicClient)
    assert client.model_id == "anthropic:claude-sonnet-5-5"


# --- Live smoke test (opt-in: uv run pytest -m live -k anthropic) ----------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_live_opus_calls_a_tool() -> None:
    client = create_client("anthropic:claude-opus-5-5")
    completion = await client.complete(
        [Message.user("The service is alerting. Check its health using the tool.")],
        system="You are an on-call assistant. Use the tools provided.",
        tools=TOOLS,
    )
    assert completion.stop_reason == "tool_use"
    assert [c.name for c in completion.tool_calls] == ["check_health"]
    assert completion.usage.output_tokens > 0
