"""Scenario model and seeding: deterministic, fault starts at onset, noise never gives it away."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import ValidationError

from eval.load import NOISE_LOGS, seed
from eval.scenarios import SCENARIOS, AlertSpec, Evidence, Scenario, ScheduledDeploy
from service.main import create_app
from service.state import Settings

TOKEN = "t"
BASE = "http://svc"
ALL_LOGS = {"since_minutes": 60, "limit": 200}
START = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
DB_DOWN = SCENARIOS["db_down"]


def http_for(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE)


async def seeded(scenario: Scenario, seed_value: int = 3):
    app = create_app(Settings(admin_token=TOKEN, start_time=START))
    async with http_for(app) as http:
        alert = await seed(scenario, http, admin_token=TOKEN, seed=seed_value)
    return app, alert


async def snapshot(app) -> dict:
    """Everything the agent could see, as the agent's endpoints return it."""
    async with http_for(app) as http:
        return {
            "logs": (await http.get("/ops/logs", params=ALL_LOGS)).json(),
            "metrics": (await http.get("/metrics", params={"window_minutes": 60})).json(),
            "deploys": (await http.get("/ops/deploys")).json(),
            "health": (await http.get("/health")).json(),
        }


# --- Scenario model --------------------------------------------------------------------------


def _scenario(**overrides) -> Scenario:
    base = DB_DOWN.model_dump()
    return Scenario.model_validate({**base, **overrides})


def test_unknown_fault_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown service faults"):
        _scenario(faults=("gremlins",))


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown tools"):
        _scenario(cheapest_path=("ssh_into_box",))


def test_is_decisive_needs_the_right_tool_and_match() -> None:
    assert DB_DOWN.is_decisive("check_health", '{"db":{"status":"down","detail":"x"}}')
    assert not DB_DOWN.is_decisive("check_health", '{"db":{"status":"ok"}}')
    assert not DB_DOWN.is_decisive("get_metrics", '"db":{"status":"down"')  # wrong tool


def test_every_scenario_alert_hides_the_answer() -> None:
    for s in SCENARIOS.values():
        alert_text = s.alert.model_dump_json().lower()
        assert s.true_category not in alert_text
        for e in s.decisive_evidence:
            assert e.match.lower() not in alert_text, s.name


def test_noise_never_matches_decisive_evidence() -> None:
    for s in SCENARIOS.values():
        for line in NOISE_LOGS:
            assert not s.is_decisive("get_logs", str(line)), (s.name, line)


# --- Seeding ---------------------------------------------------------------------------------


async def test_same_seed_same_telemetry_different_seed_differs() -> None:
    a, _ = await seeded(DB_DOWN, 3)
    b, _ = await seeded(DB_DOWN, 3)
    c, _ = await seeded(DB_DOWN, 4)
    assert await snapshot(a) == await snapshot(b)
    assert (await snapshot(a))["logs"] != (await snapshot(c))["logs"]


async def test_alert_fires_after_the_backfill() -> None:
    _, alert = await seeded(DB_DOWN)
    assert alert.fired_at == START + timedelta(minutes=30)
    assert alert.name == DB_DOWN.alert.name


async def test_fault_starts_at_onset_not_before() -> None:
    app, _ = await seeded(DB_DOWN)
    snap = await snapshot(app)
    assert snap["health"]["components"]["db"]["status"] == "down"

    errors = [r for r in snap["logs"]["logs"] if r["level"] == "ERROR"]
    first_error = min(datetime.fromisoformat(r["ts"]) for r in errors)
    assert first_error == START + timedelta(minutes=30 - DB_DOWN.onset_minutes)


async def test_history_has_traffic_and_noise() -> None:
    app, _ = await seeded(DB_DOWN)
    logs = (await snapshot(app))["logs"]["logs"]
    noise_msgs = {n["msg"] for n in NOISE_LOGS}
    assert any(r["msg"] in noise_msgs for r in logs)
    assert any(r.get("status") == 404 for r in logs)  # stale-link 404s
    request_rate = (await snapshot(app))["metrics"]["metrics"]["request_rate"]
    assert request_rate["min"] > 0  # steady traffic through the whole window


async def test_scheduled_deploy_is_recorded_at_its_minute() -> None:
    scenario = _scenario(
        deploys=(
            ScheduledDeploy(
                minutes_before_alert=6, version="v1.4.0", author="sam", summary="Tweak copy"
            ),
        )
    )
    app, _ = await seeded(scenario)
    latest = (await snapshot(app))["deploys"]["deploys"][0]
    assert latest["version"] == "v1.4.0"
    assert datetime.fromisoformat(latest["ts"]) == START + timedelta(minutes=24)


def test_alert_spec_fires_with_a_timestamp() -> None:
    spec = AlertSpec(name="X", severity="warning", summary="s")
    assert spec.fire(START).fired_at == START


def test_evidence_requires_a_match() -> None:
    with pytest.raises(ValidationError):
        Evidence(tool="get_logs", match="")
