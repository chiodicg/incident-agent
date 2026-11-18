"""OpenAI adapter: neutral <-> Responses API translation. No network except `-m live`."""

import os
from typing import Any

import httpx2
import openai
import pytest
from openai.types.responses import Response

from llm.base import LLMProviderError, create_client
from llm.openai_client import (
    ERROR_PREFIX,
    OpenAIClient,
    build_request,
    parse_response,
    to_input_items,
    to_tool_choice,
)
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
REASONING_ITEM = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": "Check health first."}],
    "encrypted_content": "enc-abc",
}
CALL_ITEM = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_9",
    "name": "check_health",
    "arguments": '{"verbose": true}',
    "status": "completed",
}


def text_item(text: str) -> dict:
    return {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def sdk_response(
    output: list[dict], status: str = "completed", incomplete: str | None = None, **usage: int
) -> Response:
    return Response.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-6-luna",
            "output": output,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "status": status,
            "incomplete_details": {"reason": incomplete} if incomplete else None,
            "usage": {
                "input_tokens": usage.get("input", 100),
                "output_tokens": usage.get("output", 20),
                "total_tokens": 0,
                "input_tokens_details": {
                    "cached_tokens": usage.get("cached", 0),
                    "cache_write_tokens": usage.get("written", 0),
                },
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


# --- Request building ------------------------------------------------------------------------


def test_build_request_shape() -> None:
    req = build_request(
        "gpt-6-luna",
        [Message.user("alert")],
        system="You are on call.",
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=16000,
        parallel_tool_calls=False,
    )
    assert req["model"] == "gpt-6-luna"
    assert req["instructions"] == "You are on call."
    assert req["input"] == [{"role": "user", "content": "alert"}]
    assert req["tools"] == [
        {
            "type": "function",
            "name": "check_health",
            "description": "Read the health endpoint.",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]
    assert req["tool_choice"] == "auto"
    assert req["parallel_tool_calls"] is False
    assert req["max_output_tokens"] == 16000
    assert req["store"] is False
    assert req["include"] == ["reasoning.encrypted_content"]
    assert req["reasoning"] == {"summary": "auto"}  # no effort: provider default


def test_compat_endpoints_get_no_openai_only_options() -> None:
    req = build_request(
        "llama3.1",
        [Message.user("alert")],
        system="s",
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=100,
        parallel_tool_calls=False,
        openai_only_options=False,
    )
    assert "reasoning" not in req
    assert "include" not in req


@pytest.mark.parametrize(
    "choice, expected",
    [
        ("auto", "auto"),
        ("required", "required"),
        (ForceTool(name="submit_diagnosis"), {"type": "function", "name": "submit_diagnosis"}),
    ],
)
def test_tool_choice(choice: Any, expected: Any) -> None:
    assert to_tool_choice(choice) == expected


def test_assistant_and_tool_messages_translate() -> None:
    history = [
        Message.user("alert"),
        Message(
            role="assistant",
            parts=[
                ReasoningPart(provider="openai", text="Check health first.", data=REASONING_ITEM),
                TextPart(text="Checking health."),
                ToolCallPart(id="call_9", name="check_health", arguments={"verbose": True}),
            ],
        ),
        Message(
            role="tool",
            parts=[
                ToolResultPart(call_id="call_9", content="db down"),
                ToolResultPart(call_id="call_10", content="bad args", is_error=True),
            ],
        ),
    ]
    assert to_input_items(history)[1:] == [
        REASONING_ITEM,  # replayed unchanged, before the call it led to
        {"role": "assistant", "content": "Checking health."},
        {
            "type": "function_call",
            "call_id": "call_9",
            "name": "check_health",
            "arguments": '{"verbose": true}',
        },
        {"type": "function_call_output", "call_id": "call_9", "output": "db down"},
        {
            "type": "function_call_output",
            "call_id": "call_10",
            "output": f"{ERROR_PREFIX}bad args",
        },
    ]


def test_reasoning_from_other_providers_is_skipped() -> None:
    msg = Message(
        role="assistant",
        parts=[
            ReasoningPart(provider="anthropic", data={"type": "thinking", "signature": "s"}),
            TextPart(text="hi"),
        ],
    )
    assert to_input_items([msg]) == [{"role": "assistant", "content": "hi"}]


# --- Response parsing ------------------------------------------------------------------------


def test_parse_response_with_reasoning_text_and_tool_call() -> None:
    completion = parse_response(
        sdk_response([REASONING_ITEM, text_item("Looking."), CALL_ITEM], cached=40, written=7)
    )
    assert completion.stop_reason == "tool_use"
    assert completion.text == "Looking."
    assert completion.reasoning == "Check health first."
    assert completion.parts[0].data == REASONING_ITEM
    assert completion.tool_calls == [
        ToolCallPart(id="call_9", name="check_health", arguments={"verbose": True})
    ]
    # OpenAI's input_tokens includes cached ones; ours is uncached only
    assert completion.usage == Usage(
        input_tokens=53, output_tokens=20, cache_read_tokens=40, cache_write_tokens=7
    )


def test_text_only_response_is_end_turn() -> None:
    assert parse_response(sdk_response([text_item("done")])).stop_reason == "end_turn"


def test_truncated_response_is_max_tokens() -> None:
    response = sdk_response([text_item("par")], status="incomplete", incomplete="max_output_tokens")
    assert parse_response(response).stop_reason == "max_tokens"


@pytest.mark.parametrize("via", ["refusal_part", "content_filter"])
def test_refusals_are_normalized(via: str) -> None:
    if via == "refusal_part":
        item = text_item("")
        item["content"] = [{"type": "refusal", "refusal": "I can't help with that."}]
        response = sdk_response([item])
    else:
        response = sdk_response([], status="incomplete", incomplete="content_filter")
    assert parse_response(response).stop_reason == "refusal"


def test_malformed_arguments_reach_the_tool_instead_of_crashing() -> None:
    bad = {**CALL_ITEM, "arguments": '{"level": "ERR'}
    call = parse_response(sdk_response([bad])).tool_calls[0]
    assert call.arguments == {"_unparseable_arguments": '{"level": "ERR'}


def test_parsed_completion_round_trips_into_next_request() -> None:
    """Append-only history: reasoning and the call go back in the order they came."""
    completion = parse_response(sdk_response([REASONING_ITEM, CALL_ITEM]))
    sent = to_input_items([completion.to_message()])
    assert sent[0] == REASONING_ITEM
    assert sent[1]["type"] == "function_call"
    assert sent[1]["call_id"] == "call_9"


# --- Client ----------------------------------------------------------------------------------


class FakeResponses:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.kwargs: dict | None = None

    async def create(self, **kwargs: Any) -> Response:
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeSDK:
    def __init__(self, result: Any) -> None:
        self.responses = FakeResponses(result)


async def test_client_sends_request_and_parses_response() -> None:
    sdk = FakeSDK(sdk_response([text_item("done")]))
    client = OpenAIClient("gpt-6-luna", sdk=sdk)

    completion = await client.complete(
        [Message.user("alert")], system="s", tools=TOOLS, tool_choice=ForceTool(name="check_health")
    )

    assert completion.text == "done"
    assert sdk.responses.kwargs["tool_choice"] == {"type": "function", "name": "check_health"}
    assert sdk.responses.kwargs["parallel_tool_calls"] is False
    assert client.model_id == "openai:gpt-6-luna"
    assert client.config["api"] == "responses"


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.openai.com/v1/responses")


def _status_error(cls: type, status: int) -> Exception:
    return cls("boom", response=httpx2.Response(status, request=_request()), body=None)


@pytest.mark.parametrize(
    "error",
    [
        _status_error(openai.RateLimitError, 429),
        _status_error(openai.InternalServerError, 500),
        _status_error(openai.APIStatusError, 503),
        openai.APIConnectionError(request=_request()),
        openai.APITimeoutError(request=_request()),
    ],
    ids=["429", "500", "503", "connection", "timeout"],
)
async def test_infrastructure_errors_become_provider_errors(error: Exception) -> None:
    client = OpenAIClient("gpt-6-luna", sdk=FakeSDK(error))
    with pytest.raises(LLMProviderError):
        await client.complete([Message.user("x")], system="s", tools=TOOLS)


@pytest.mark.parametrize(
    "error",
    [
        _status_error(openai.BadRequestError, 400),
        _status_error(openai.AuthenticationError, 401),
        _status_error(openai.NotFoundError, 404),
    ],
    ids=["400", "401", "404"],
)
async def test_configuration_errors_propagate_unchanged(error: Exception) -> None:
    client = OpenAIClient("gpt-6-luna", sdk=FakeSDK(error))
    with pytest.raises(type(error)):
        await client.complete([Message.user("x")], system="s", tools=TOOLS)


def test_create_client_resolves_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = create_client("openai:gpt-6-luna")
    assert isinstance(client, OpenAIClient)
    assert client.model_id == "openai:gpt-6-luna"


def test_create_client_resolves_compat_with_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    client = create_client("openai-compat:llama3.1:8b@http://localhost:11434/v1")
    assert isinstance(client, OpenAIClient)
    assert client.model_id == "openai-compat:llama3.1:8b@http://localhost:11434/v1"
    assert str(client._sdk.base_url).startswith("http://localhost:11434/v1")
    assert client.config["reasoning"] == "not sent"


# --- Live smoke test (opt-in: uv run pytest -m live -k openai) -------------------------------


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_live_luna_calls_a_tool_and_accepts_replayed_reasoning() -> None:
    client = create_client("openai:gpt-6-luna")
    history = [Message.user("The service is alerting. Check its health using the tool.")]
    system = "You are an on-call assistant. Use the tools provided."

    first = await client.complete(history, system=system, tools=TOOLS)
    assert first.stop_reason == "tool_use"
    assert [c.name for c in first.tool_calls] == ["check_health"]

    # Second turn replays reasoning items statelessly (store=False); the API must accept them.
    call = first.tool_calls[0]
    history += [
        first.to_message(),
        Message(role="tool", parts=[ToolResultPart(call_id=call.id, content='{"status":"ok"}')]),
    ]
    second = await client.complete(history, system=system, tools=TOOLS)
    assert second.stop_reason in ("end_turn", "tool_use")
    assert second.usage.output_tokens > 0
