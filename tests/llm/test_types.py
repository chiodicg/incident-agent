"""Contract for llm/types.py, the provider-neutral data model.

Everything outside llm/ (agent loop, eval, traces) speaks only these types; adapters translate
them to and from each provider's wire format.
"""

import pytest
from pydantic import ValidationError

from llm.types import (
    Completion,
    ForceTool,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ToolSpec,
    Usage,
)

CALL = ToolCallPart(id="call_1", name="check_health", arguments={})
RESULT = ToolResultPart(call_id="call_1", content='{"status": "ok"}')


# --- Messages and parts ----------------------------------------------------------------------


def test_parts_are_discriminated_by_type_when_parsing_dicts() -> None:
    msg = Message.model_validate(
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "text": "Checking health first."},
                {"type": "tool_call", "id": "c1", "name": "check_health", "arguments": {}},
            ],
        }
    )
    assert isinstance(msg.parts[0], TextPart)
    assert isinstance(msg.parts[1], ToolCallPart)


def test_tool_call_arguments_default_to_empty_dict() -> None:
    assert ToolCallPart(id="c1", name="check_health").arguments == {}


def test_tool_result_is_not_an_error_by_default() -> None:
    assert RESULT.is_error is False


def test_user_helper_builds_a_single_text_part() -> None:
    assert Message.user("alert!") == Message(role="user", parts=[TextPart(text="alert!")])


def test_conversation_round_trips_through_json() -> None:
    """Traces are stored as JSON, so a message must survive serialization unchanged."""
    convo = [
        Message.user("p95 latency > 2s on /orders"),
        Message(role="assistant", parts=[TextPart(text="Let me look."), CALL]),
        Message(role="tool", parts=[RESULT]),
    ]
    assert [Message.model_validate_json(m.model_dump_json()) for m in convo] == convo


# --- Role invariants: tool results live only in `tool` messages ------------------------------


def test_tool_message_holds_tool_results() -> None:
    msg = Message(role="tool", parts=[RESULT, RESULT.model_copy(update={"call_id": "call_2"})])
    assert len(msg.parts) == 2


@pytest.mark.parametrize(
    "role, part",
    [
        ("user", RESULT),  # tool results must not hide inside user messages
        ("assistant", RESULT),
        ("tool", TextPart(text="hi")),  # tool messages carry only results
        ("tool", CALL),
        ("user", CALL),  # only the assistant calls tools
    ],
)
def test_parts_must_match_role(role: str, part: object) -> None:
    with pytest.raises(ValidationError):
        Message(role=role, parts=[part])


def test_message_needs_at_least_one_part() -> None:
    with pytest.raises(ValidationError):
        Message(role="user", parts=[])


# --- Tools -----------------------------------------------------------------------------------


def test_tool_spec_holds_json_schema_parameters() -> None:
    spec = ToolSpec(
        name="get_logs",
        description="Read recent service logs.",
        parameters={"type": "object", "properties": {"limit": {"type": "integer"}}},
    )
    assert spec.parameters["type"] == "object"


@pytest.mark.parametrize("name", ["get logs", "", "x" * 65, "get.logs"])
def test_tool_name_must_be_valid_for_every_provider(name: str) -> None:
    """Both Anthropic and OpenAI require ^[a-zA-Z0-9_-]{1,64}$."""
    with pytest.raises(ValidationError):
        ToolSpec(name=name, description="d", parameters={"type": "object"})


def test_force_tool_names_a_specific_tool() -> None:
    assert ForceTool(name="submit_diagnosis").name == "submit_diagnosis"


# --- Completions and usage -------------------------------------------------------------------


def test_completion_exposes_text_and_tool_calls() -> None:
    completion = Completion(
        parts=[TextPart(text="DB looks "), TextPart(text="down."), CALL],
        stop_reason="tool_use",
        usage=Usage(input_tokens=100, output_tokens=20),
    )
    assert completion.text == "DB looks down."
    assert completion.tool_calls == [CALL]


def test_completion_without_tool_calls() -> None:
    completion = Completion(
        parts=[TextPart(text="done")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    assert completion.tool_calls == []


def test_completion_cannot_contain_tool_results() -> None:
    with pytest.raises(ValidationError):
        Completion(
            parts=[RESULT], stop_reason="end_turn", usage=Usage(input_tokens=1, output_tokens=1)
        )


def test_unknown_stop_reason_rejected() -> None:
    """Adapters must normalize provider stop reasons into the neutral set."""
    with pytest.raises(ValidationError):
        Completion(parts=[], stop_reason="length", usage=Usage(input_tokens=0, output_tokens=0))


def test_usage_adds_up_across_calls() -> None:
    total = Usage(input_tokens=100, output_tokens=10) + Usage(input_tokens=50, output_tokens=5)
    assert total == Usage(input_tokens=150, output_tokens=15)


def test_usage_rejects_negative_tokens() -> None:
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1, output_tokens=0)
