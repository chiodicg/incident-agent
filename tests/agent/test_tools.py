"""Agent tools, run against the real service in-process (ASGITransport) so the tool layer and
the service API can't drift apart. Failure modes use httpx.MockTransport."""

import json
from collections.abc import AsyncIterator
from typing import get_args

import httpx
import pytest

from agent.models import ROOT_CAUSES
from agent.tools import (
    DIAGNOSTIC_TOOLS,
    SUBMIT_TOOL,
    MetricName,
    ToolBox,
    tool_specs,
)
from llm.types import ToolCallPart
from service.faults import trigger
from service.main import create_app
from service.state import Settings
from service.telemetry import METRIC_NAMES

BASE = "http://svc"


@pytest.fixture
def app():
    return create_app(Settings(admin_token="t", seed=5))


@pytest.fixture
async def toolbox(app) -> AsyncIterator[ToolBox]:
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE)
    box = ToolBox(http)
    yield box
    await box.aclose()


def call(name: str, **arguments) -> ToolCallPart:
    return ToolCallPart(id="c1", name=name, arguments=arguments)


async def generate_traffic(app) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as c:
        await c.post("/orders", json={"item": "x", "amount": 3})
        for _ in range(5):
            await c.get("/orders")


# --- Specs -----------------------------------------------------------------------------------


def test_specs_cover_four_tools_plus_submit_in_fixed_order() -> None:
    assert [s.name for s in tool_specs()] == [
        "get_logs",
        "get_metrics",
        "get_deploys",
        "check_health",
        SUBMIT_TOOL,
    ]


def test_specs_are_deterministic() -> None:
    """Prompt conditions must see byte-identical tool definitions."""
    dump = lambda: json.dumps([s.model_dump() for s in tool_specs()], sort_keys=True)  # noqa: E731
    assert dump() == dump()


def test_schemas_have_no_refs_or_titles() -> None:
    text = json.dumps([s.parameters for s in tool_specs()])
    assert "$ref" not in text
    assert '"title"' not in text


def test_submit_schema_lists_every_root_cause() -> None:
    submit = next(s for s in tool_specs() if s.name == SUBMIT_TOOL)
    props = submit.parameters["properties"]
    assert props["root_cause_category"]["enum"] == list(ROOT_CAUSES)
    assert set(submit.parameters["required"]) >= {"root_cause_category", "proposed_fix"}


def test_descriptions_carry_no_ordering_or_cost_hints() -> None:
    """Guidance lives only in the per-condition system prompt."""
    banned = [
        "use this first",
        "before ",
        "after ",
        "cheap",
        "expensive",
        "should",
        "start with",
        "prefer",
        "always",
    ]
    for spec in tool_specs():
        lowered = spec.description.lower()
        assert not [w for w in banned if w in lowered], spec.name


def test_agent_metric_names_match_the_service() -> None:
    assert set(get_args(MetricName)) == set(METRIC_NAMES)


# --- Against the real service ----------------------------------------------------------------


async def test_get_logs_returns_recent_requests(app, toolbox: ToolBox) -> None:
    await generate_traffic(app)
    result = await toolbox.run(call("get_logs", limit=2))
    assert not result.is_error
    body = json.loads(result.content)
    assert body["count"] == 2
    assert result.call_id == "c1"


async def test_get_logs_level_filter(app, toolbox: ToolBox) -> None:
    await generate_traffic(app)
    trigger(app.state.svc, "db_down")
    await generate_traffic(app)

    body = json.loads((await toolbox.run(call("get_logs", level="ERROR"))).content)
    assert body["count"] > 0
    assert all(line["level"] == "ERROR" for line in body["logs"])


async def test_get_metrics_with_name_filter(app, toolbox: ToolBox) -> None:
    await generate_traffic(app)
    result = await toolbox.run(call("get_metrics", names=["error_rate", "p95_latency_ms"]))
    assert set(json.loads(result.content)["metrics"]) == {"error_rate", "p95_latency_ms"}


async def test_get_deploys(toolbox: ToolBox) -> None:
    result = await toolbox.run(call("get_deploys", limit=1))
    assert [d["version"] for d in json.loads(result.content)["deploys"]] == ["v1.3.2"]


async def test_check_health_503_is_data_not_error(app, toolbox: ToolBox) -> None:
    trigger(app.state.svc, "db_down")
    result = await toolbox.run(call("check_health"))
    assert not result.is_error
    assert json.loads(result.content)["components"]["db"]["status"] == "down"


async def test_full_metrics_fit_in_default_budget(app, toolbox: ToolBox) -> None:
    """The most verbose call shouldn't need truncation, or the agent loses data silently."""
    await generate_traffic(app)
    result = await toolbox.run(call("get_metrics"))
    assert "truncated" not in result.content


# --- Agent mistakes become readable error results --------------------------------------------


async def test_unknown_tool(toolbox: ToolBox) -> None:
    result = await toolbox.run(call("restart_service"))
    assert result.is_error
    assert "get_logs" in result.content


@pytest.mark.parametrize(
    "arguments",
    [{"limit": 500}, {"level": "LOUD"}, {"surprise": 1}],
    ids=["over-limit", "bad-enum", "extra-field"],
)
async def test_invalid_arguments(toolbox: ToolBox, arguments: dict) -> None:
    result = await toolbox.run(call("get_logs", **arguments))
    assert result.is_error
    assert result.content.startswith("Invalid arguments for get_logs")


async def test_submit_is_not_a_toolbox_tool(toolbox: ToolBox) -> None:
    with pytest.raises(ValueError):
        await toolbox.run(call(SUBMIT_TOOL))


# --- Service failures ------------------------------------------------------------------------


def mock_box(handler, **kwargs) -> ToolBox:
    return ToolBox(
        httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE), **kwargs
    )


async def test_http_error_from_ops_endpoint_is_error_result() -> None:
    box = mock_box(lambda req: httpx.Response(500, json={"detail": "log backend unavailable"}))
    result = await box.run(call("get_logs"))
    assert result.is_error
    assert "HTTP 500" in result.content
    assert "log backend unavailable" in result.content


async def test_transport_failure_is_error_result() -> None:
    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=req)

    result = await mock_box(boom).run(call("check_health"))
    assert result.is_error
    assert "ConnectTimeout" in result.content


async def test_long_results_are_truncated_with_hint() -> None:
    box = mock_box(lambda req: httpx.Response(200, text="x" * 500), max_chars=100)
    result = await box.run(call("get_logs"))
    assert len(result.content) < 200
    assert "truncated 400 characters" in result.content


# --- Read-only guarantee ---------------------------------------------------------------------


async def test_tools_never_touch_admin_or_send_credentials() -> None:
    seen: list[httpx.Request] = []

    def record(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={})

    box = mock_box(record)
    for name in DIAGNOSTIC_TOOLS:
        await box.run(call(name))

    assert len(seen) == len(DIAGNOSTIC_TOOLS)
    assert all(req.method == "GET" for req in seen)
    assert not any(req.url.path.startswith("/admin") for req in seen)
    assert not any("x-admin-token" in req.headers for req in seen)


async def test_route_argument_cannot_redirect_the_request() -> None:
    """`route` is a query parameter, never part of the path."""
    seen: list[httpx.Request] = []
    box = mock_box(lambda req: seen.append(req) or httpx.Response(200, json={}))

    await box.run(call("get_logs", route="/../admin/reset"))
    assert seen[0].url.path == "/ops/logs"


async def test_default_log_read_fits_in_default_budget(app, toolbox: ToolBox) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE) as c:
        await c.post("/orders", json={"item": "x", "amount": 3})
        for _ in range(60):
            await c.get("/orders/1")

    result = await toolbox.run(call("get_logs"))
    assert json.loads(result.content)["count"] == 50
