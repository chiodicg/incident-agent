"""In-process fake dependencies with switchable failure knobs.

Faults (service/faults.py) work by flipping these knobs. Calls charge *simulated* latency to
the current request instead of sleeping, so the service stays fast and deterministic.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from typing import Any, Literal

from service.telemetry import MetricStore, add_latency

HealthStatus = Literal["ok", "degraded", "down"]


class DependencyError(Exception):
    """A dependency failed; surfaces to clients as a 503."""

    def __init__(self, component: str, message: str) -> None:
        super().__init__(f"{component}: {message}")
        self.component = component
        self.message = message


@dataclass(frozen=True)
class ComponentHealth:
    status: HealthStatus
    detail: str = ""


class Database:
    def __init__(self, rng: random.Random, metrics: MetricStore, path: str = ":memory:") -> None:
        self._rng = rng
        self._metrics = metrics
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS orders ("
            " id INTEGER PRIMARY KEY, item TEXT NOT NULL, amount REAL NOT NULL,"
            " status TEXT NOT NULL DEFAULT 'pending')"
        )
        # Fault knobs
        self.down = False
        self.extra_latency_ms = 0.0
        self.base_latency_ms = 4.0
        self.pool_size = 10

    def _call(self) -> None:
        if self.down:
            raise DependencyError("db", "connection refused (orders-db:5432)")
        latency = self.base_latency_ms + self.extra_latency_ms + self._rng.uniform(0, 3)
        add_latency(latency)
        self._metrics.observe("db_query_ms", latency)
        self._metrics.observe("db_pool_in_use", self._rng.randint(1, 3))

    def list_orders(self, limit: int = 20) -> list[dict[str, Any]]:
        self._call()
        rows = self._conn.execute("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def get_order(self, order_id: int) -> dict[str, Any] | None:
        self._call()
        row = self._conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        return dict(row) if row else None

    def create_order(self, item: str, amount: float) -> dict[str, Any]:
        self._call()
        cur = self._conn.execute("INSERT INTO orders (item, amount) VALUES (?, ?)", (item, amount))
        self._conn.commit()
        return {"id": cur.lastrowid, "item": item, "amount": amount, "status": "pending"}

    def mark_paid(self, order_id: int) -> None:
        self._call()
        self._conn.execute("UPDATE orders SET status = 'paid' WHERE id = ?", (order_id,))
        self._conn.commit()

    def health(self) -> ComponentHealth:
        if self.down:
            return ComponentHealth("down", "connection refused (orders-db:5432)")
        return ComponentHealth("ok", f"pool_size={self.pool_size}")


class Cache:
    def __init__(self, metrics: MetricStore) -> None:
        self._metrics = metrics
        self._data: dict[str, Any] = {}
        self._hits = 0
        self._lookups = 0
        self.down = False

    def get(self, key: str) -> Any | None:
        self._lookups += 1
        value = None if self.down else self._data.get(key)
        if value is not None:
            self._hits += 1
        add_latency(0.5)
        self._metrics.observe("cache_hit_rate", self._hits / self._lookups)
        return value

    def set(self, key: str, value: Any) -> None:
        if not self.down:
            self._data[key] = value

    def invalidate(self, key: str) -> None:
        self._data.pop(key, None)

    def health(self) -> ComponentHealth:
        if self.down:
            return ComponentHealth("down", "connection refused (cache:6379)")
        return ComponentHealth("ok")


class PaymentUpstream:
    """Third-party payment provider."""

    def __init__(self, rng: random.Random, metrics: MetricStore) -> None:
        self._rng = rng
        self._metrics = metrics
        self.base_latency_ms = 80.0
        self.extra_latency_ms = 0.0
        self.forced_status: int | None = None  # e.g. 401, 429, 503

    def charge(self, order_id: int, amount: float) -> str:
        add_latency(self.base_latency_ms + self.extra_latency_ms + self._rng.uniform(0, 20))
        status = self.forced_status or 200
        self._metrics.observe_upstream(status)
        if status != 200:
            raise DependencyError("payments", f"upstream returned HTTP {status}")
        return f"ch_{order_id:06d}"

    def health(self) -> ComponentHealth:
        if self.forced_status:
            return ComponentHealth("degraded", f"last status {self.forced_status}")
        return ComponentHealth("ok")


class Host:
    """The machine the service runs on: memory, CPU, disk."""

    def __init__(self, rng: random.Random, metrics: MetricStore) -> None:
        self._rng = rng
        self._metrics = metrics
        self.memory_mb = 310.0
        self.memory_limit_mb = 1024.0
        self.cpu_pct = 18.0
        self.disk_used_pct = 61.0

    def sample(self) -> None:
        """Record host gauges; called once per handled request."""
        self._metrics.observe("memory_mb", self.memory_mb + self._rng.uniform(-5, 5))
        self._metrics.observe("cpu_pct", max(0.0, self.cpu_pct + self._rng.uniform(-4, 4)))

    def write_audit(self, record: dict[str, Any]) -> None:
        if self.disk_used_pct >= 100:
            raise DependencyError("disk", "No space left on device: /var/lib/app/audit.log")
        add_latency(1.0)

    def disk_health(self) -> ComponentHealth:
        if self.disk_used_pct >= 100:
            return ComponentHealth("down", "/var/lib/app 100% used")
        if self.disk_used_pct >= 90:
            return ComponentHealth("degraded", f"/var/lib/app {self.disk_used_pct:.0f}% used")
        return ComponentHealth("ok", f"/var/lib/app {self.disk_used_pct:.0f}% used")
