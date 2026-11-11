import pytest

from llm.base import LLMClient
from llm.scripted import RecordedCall, ScriptedLLMClient, ScriptExhaustedError
from llm.types import Completion, ForceTool, Message, TextPart, ToolCallPart, ToolSpec, Usage

USAGE = Usage(input_tokens=10, output_tokens=2)
TOOLS = [ToolSpec(name="check_health", description="d", parameters={"type": "object"})]


def done(text: str) -> Completion:
    return Completion(parts=[TextPart(text=text)], stop_reason="end_turn", usage=USAGE)


def test_satisfies_llm_client_protocol() -> None:
    assert isinstance(ScriptedLLMClient([]), LLMClient)


async def test_replays_completions_in_order() -> None:
    client = ScriptedLLMClient([done("one"), done("two")])
    first = await client.complete([Message.user("hi")], system="s", tools=TOOLS)
    second = await client.complete([Message.user("hi")], system="s", tools=TOOLS)
    assert (first.text, second.text) == ("one", "two")


async def test_records_what_it_received() -> None:
    client = ScriptedLLMClient([done("ok")])
    messages = [Message.user("alert")]

    await client.complete(messages, system="sys", tools=TOOLS, tool_choice=ForceTool(name="x"))
    messages.append(Message.user("mutated later"))  # recording must not change

    assert client.calls == [
        RecordedCall(
            messages=[Message.user("alert")],
            system="sys",
            tools=TOOLS,
            tool_choice=ForceTool(name="x"),
        )
    ]


async def test_script_step_can_be_a_function_of_the_call() -> None:
    """Lets a test react to what the loop sent, e.g. echo the forced tool."""

    def respond(call: RecordedCall) -> Completion:
        assert isinstance(call.tool_choice, ForceTool)
        return Completion(
            parts=[ToolCallPart(id="c1", name=call.tool_choice.name)],
            stop_reason="tool_use",
            usage=USAGE,
        )

    client = ScriptedLLMClient([respond])
    result = await client.complete(
        [Message.user("x")], system="s", tools=TOOLS, tool_choice=ForceTool(name="check_health")
    )
    assert result.tool_calls[0].name == "check_health"


async def test_running_past_the_script_fails_loudly() -> None:
    client = ScriptedLLMClient([done("only")])
    await client.complete([Message.user("x")], system="s", tools=TOOLS)
    with pytest.raises(ScriptExhaustedError):
        await client.complete([Message.user("x")], system="s", tools=TOOLS)
