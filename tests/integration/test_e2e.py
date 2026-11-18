"""End to end without network: real service (in-process) + real tools + loop + scripted model.

The scripted model checks that the evidence it would need actually reached it, so this test
fails if the service, the tools or the loop stop carrying the db_down signal.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from agent.cli import format_trace
from agent.loop import run_agent
from agent.models import Alert
from agent.tools import SUBMIT_TOOL, ToolBox
from llm.scripted import RecordedCall, ScriptedLLMClient
from llm.types import Completion, ToolCallPart, ToolResultPart, Usage
from service.main import create_app
from service.state import Settings

BASE = "http://svc"
ADMIN = {"X-Admin-Token": "t"}
ALERT_FILE = Path(__file__).parents[2] / "alerts" / "db_down.json"
USAGE = Usage(input_tokens=100, output_tokens=10)


@pytest.fixture
def app():
    return create_app(Settings(admin_token="t", seed=7))


@pytest.fixture
async def db_down(app) -> None:
    """Healthy traffic, then the fault, then traffic that hits it (as T10 will seed)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url=BASE) as c:
        await c.post("/orders", json={"item": "seed", "amount": 5})
        for i in range(10):
            await c.get("/orders")
            await c.get(f"/orders/{i % 2 + 1}")
        await c.post("/admin/clock/advance", json={"seconds": 1500}, headers=ADMIN)
        assert (await c.post("/admin/faults/db_down", headers=ADMIN)).status_code == 200
        for _ in range(10):
            await c.get("/orders")
        await c.post("/admin/clock/advance", json={"seconds": 300}, headers=ADMIN)


@pytest.fixture
async def toolbox(app) -> AsyncIterator[ToolBox]:
    box = ToolBox(httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE))
    yield box
    await box.aclose()


def _call(name: str, **arguments) -> Completion:
    return Completion(
        parts=[ToolCallPart(id=f"c-{name}", name=name, arguments=arguments)],
        stop_reason="tool_use",
        usage=USAGE,
    )


def _last_result(call: RecordedCall) -> str:
    part = call.messages[-1].parts[0]
    assert isinstance(part, ToolResultPart)
    return part.content


def after_health(call: RecordedCall) -> Completion:
    assert '"db":{"status":"down"' in _last_result(call)
    return _call("get_logs", level="ERROR")


def after_logs(call: RecordedCall) -> Completion:
    assert "connection refused" in _last_result(call)
    return _call(
        SUBMIT_TOOL,
        root_cause_category="dependency_down_db",
        explanation="The orders database refuses connections, so DB-backed requests fail.",
        evidence=["/health: db down", "ERROR logs: connection refused from db"],
        proposed_fix="Restore the database (check the DB host and network); no rollback needed.",
        fix_risk="low",
        confidence=0.9,
    )


async def test_scripted_agent_diagnoses_db_down(db_down, toolbox: ToolBox) -> None:
    alert = Alert.model_validate_json(ALERT_FILE.read_text())
    client = ScriptedLLMClient([_call("check_health"), after_health, after_logs])

    trace = await run_agent(alert, client=client, tools=toolbox, budget=4)

    assert trace.outcome == "submitted"
    assert trace.tool_sequence == ["check_health", "get_logs"]
    assert trace.diagnosis is not None
    assert trace.diagnosis.root_cause_category == "dependency_down_db"

    text = format_trace(trace)
    assert "step 1/4: check_health()" in text
    assert 'step 2/4: get_logs({"level": "ERROR"})' in text
    assert "outcome: submitted  steps used: 2/4" in text
    assert '"root_cause_category": "dependency_down_db"' in text


def test_alert_fixture_does_not_leak_the_root_cause() -> None:
    text = ALERT_FILE.read_text().lower()
    assert not any(word in text for word in ("db", "database", "sql", "connection"))
