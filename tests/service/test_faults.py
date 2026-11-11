"""Fault signature tests: each fault's decisive evidence is visible through the ops endpoints."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from service.main import create_app
from service.state import Settings

ADMIN = {"X-Admin-Token": "t"}


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(Settings(admin_token="t", seed=3))) as c:
        yield c


def traffic(client: TestClient, n: int = 10) -> None:
    client.post("/orders", json={"item": "seed", "amount": 5})
    for i in range(n):
        client.get("/orders")
        client.get(f"/orders/{i % 2 + 1}")


def test_fault_list_and_unknown_fault(client: TestClient) -> None:
    listing = client.get("/admin/faults", headers=ADMIN).json()
    assert "db_down" in listing["available"]
    assert listing["active"] == []
    assert client.post("/admin/faults/nope", headers=ADMIN).status_code == 404


def test_faults_require_admin_token(client: TestClient) -> None:
    assert client.post("/admin/faults/db_down").status_code == 401
    assert client.get("/admin/faults").status_code == 401


def test_reset_clears_faults(client: TestClient) -> None:
    client.post("/admin/faults/db_down", headers=ADMIN)
    client.post("/admin/reset", headers=ADMIN)

    assert client.get("/admin/faults", headers=ADMIN).json()["active"] == []
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/orders").status_code == 200


def test_db_down_signature(client: TestClient) -> None:
    traffic(client, 5)
    client.post("/admin/clock/advance", json={"seconds": 120}, headers=ADMIN)
    assert client.post("/admin/faults/db_down", headers=ADMIN).json()["active"] == ["db_down"]
    traffic(client, 5)

    health = client.get("/health")
    assert health.status_code == 503
    assert health.json()["components"]["db"]["status"] == "down"

    errors = client.get("/ops/logs", params={"level": "ERROR"}).json()["logs"]
    assert errors and all(e["component"] == "db" for e in errors)
    assert "connection refused" in errors[0]["msg"]

    error_rate = client.get("/metrics", params={"names": "error_rate"}).json()["metrics"]
    assert error_rate["error_rate"]["latest"] > 0.4
