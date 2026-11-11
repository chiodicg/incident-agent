from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from service.main import create_app
from service.state import Settings

TOKEN = "test-token"
ADMIN = {"X-Admin-Token": TOKEN}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Settings(admin_token=TOKEN, seed=1))) as c:
        yield c


def state(client: TestClient):
    return client.app.state.svc


# --- Domain routes ---------------------------------------------------------------------------


def test_order_lifecycle(client: TestClient) -> None:
    created = client.post("/orders", json={"item": "book", "amount": 12.5})
    assert created.status_code == 201
    order_id = created.json()["id"]

    assert client.get(f"/orders/{order_id}").json()["item"] == "book"
    assert [o["id"] for o in client.get("/orders").json()] == [order_id]

    paid = client.post("/checkout", json={"order_id": order_id})
    assert paid.status_code == 200
    assert paid.json()["status"] == "paid"
    assert client.get(f"/orders/{order_id}").json()["status"] == "paid"


def test_missing_order_is_404(client: TestClient) -> None:
    assert client.get("/orders/999").status_code == 404


def test_dependency_failure_is_503_with_component_error_log(client: TestClient) -> None:
    state(client).db.down = True

    assert client.get("/orders").status_code == 503

    errors = client.get("/ops/logs", params={"level": "ERROR"}).json()["logs"]
    assert errors[0]["component"] == "db"
    assert "connection refused" in errors[0]["msg"]


# --- Logs ------------------------------------------------------------------------------------


def test_requests_are_logged_newest_first(client: TestClient) -> None:
    client.post("/orders", json={"item": "a", "amount": 1})
    client.get("/orders")

    logs = client.get("/ops/logs").json()["logs"]
    assert [entry["route"] for entry in logs[:2]] == ["/orders", "/orders"]
    assert logs[0]["msg"].startswith("GET /orders 200")


def test_log_filters(client: TestClient) -> None:
    client.post("/orders", json={"item": "a", "amount": 1})
    client.get("/orders/1")
    client.get("/orders/42")  # 404

    by_route = client.get("/ops/logs", params={"route": "/orders/{order_id}"}).json()
    assert by_route["count"] == 2
    assert client.get("/ops/logs", params={"limit": 1}).json()["count"] == 1
    assert client.get("/ops/logs", params={"level": "ERROR"}).json()["count"] == 0


def test_log_window_follows_simulated_clock(client: TestClient) -> None:
    client.get("/orders")
    client.post("/admin/clock/advance", json={"seconds": 20 * 60}, headers=ADMIN)

    assert client.get("/ops/logs", params={"since_minutes": 15}).json()["count"] == 0
    assert client.get("/ops/logs", params={"since_minutes": 30}).json()["count"] == 1


def test_log_limit_is_capped(client: TestClient) -> None:
    assert client.get("/ops/logs", params={"limit": 500}).status_code == 422


# --- Metrics ---------------------------------------------------------------------------------


def test_metrics_report_latency_and_error_rate(client: TestClient) -> None:
    for _ in range(3):
        client.get("/orders")
    state(client).db.down = True
    client.get("/orders")

    body = client.get(
        "/metrics", params=[("names", "p95_latency_ms"), ("names", "error_rate")]
    ).json()
    assert set(body["metrics"]) == {"p95_latency_ms", "error_rate"}
    assert body["metrics"]["error_rate"]["latest"] == pytest.approx(0.25)
    assert body["metrics"]["p95_latency_ms"]["latest"] > 0
    assert len(body["metrics"]["error_rate"]["series"]) == 15


def test_all_metrics_returned_by_default(client: TestClient) -> None:
    client.post("/orders", json={"item": "a", "amount": 1})
    client.post("/checkout", json={"order_id": 1})

    metrics = client.get("/metrics").json()["metrics"]
    assert {"memory_mb", "cpu_pct", "db_pool_in_use", "cache_hit_rate"} <= set(metrics)
    assert metrics["upstream_status_counts"]["totals"] == {"200": 1}


def test_unknown_metric_is_400_and_lists_available(client: TestClient) -> None:
    resp = client.get("/metrics", params={"names": "vibes"})
    assert resp.status_code == 400
    assert "p95_latency_ms" in resp.json()["detail"]


def test_same_seed_gives_identical_telemetry() -> None:
    def run() -> dict:
        with TestClient(create_app(Settings(seed=7))) as c:
            c.post("/orders", json={"item": "a", "amount": 1})
            for _ in range(5):
                c.get("/orders/1")
            return c.get("/metrics").json()

    assert run() == run()


# --- Health ----------------------------------------------------------------------------------


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["components"]) == {"app", "db", "cache", "payments_upstream", "disk"}


def test_health_reports_down_component(client: TestClient) -> None:
    state(client).db.down = True
    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["components"]["db"]["status"] == "down"


def test_agent_probes_are_not_recorded_as_traffic(client: TestClient) -> None:
    client.get("/health")
    client.get("/metrics")
    client.get("/ops/logs")
    client.get("/ops/deploys")

    assert client.get("/ops/logs").json()["count"] == 0
    assert client.get("/metrics").json()["metrics"]["request_rate"]["max"] == 0


# --- Deploys ---------------------------------------------------------------------------------


def test_deploys_newest_first_with_baseline_history(client: TestClient) -> None:
    deploys = client.get("/ops/deploys").json()["deploys"]
    assert [d["version"] for d in deploys] == ["v1.3.2", "v1.3.1", "v1.3.0"]


def test_admin_can_record_deploy(client: TestClient) -> None:
    now = client.get("/ops/deploys").json()["now"]
    resp = client.post(
        "/admin/deploys",
        headers=ADMIN,
        json={
            "version": "v1.4.0",
            "ts": now,
            "author": "sam",
            "summary": "Tune DB pool",
            "changed_config": {"DB_POOL_SIZE": 1},
        },
    )
    assert resp.status_code == 200
    latest = client.get("/ops/deploys", params={"limit": 1}).json()["deploys"][0]
    assert latest["changed_config"] == {"DB_POOL_SIZE": 1}


# --- Admin auth ------------------------------------------------------------------------------


@pytest.mark.parametrize("headers", [{}, {"X-Admin-Token": "wrong"}], ids=["missing", "wrong"])
def test_admin_requires_token(client: TestClient, headers: dict) -> None:
    assert client.post("/admin/reset", headers=headers).status_code == 401


def test_admin_disabled_when_no_token_configured() -> None:
    with TestClient(create_app(Settings(admin_token=None))) as c:
        assert c.post("/admin/reset", headers={"X-Admin-Token": ""}).status_code == 401


def test_reset_restores_clean_state(client: TestClient) -> None:
    client.post("/orders", json={"item": "a", "amount": 1})
    state(client).db.down = True
    client.post("/admin/clock/advance", json={"seconds": 60}, headers=ADMIN)

    assert client.post("/admin/reset", headers=ADMIN).status_code == 200

    assert state(client).db.down is False
    assert client.get("/orders").json() == []
    assert client.get("/ops/logs", params={"since_minutes": 60}).json()["count"] == 1
    assert len(client.get("/ops/deploys").json()["deploys"]) == 3
