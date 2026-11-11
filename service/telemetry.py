"""Simulated clock, structured logs, and metrics for the service.

Everything is timestamped by an injected clock rather than wall time, so the eval can
backfill history and runs are reproducible.
"""

from __future__ import annotations

import contextvars
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Protocol

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
LEVEL_ORDER: dict[str, int] = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

MAX_LOG_LIMIT = 200
MAX_BUCKETS = 15


class Clock(Protocol):
    def now(self) -> datetime: ...


class SimClock:
    """Time that only moves when advanced, so telemetry is deterministic."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("clock cannot go backwards")
        self._now += timedelta(seconds=seconds)


# --- Per-request simulated cost -------------------------------------------------------------


@dataclass
class RequestCost:
    latency_ms: float = 0.0


_request_cost: contextvars.ContextVar[RequestCost | None] = contextvars.ContextVar(
    "request_cost", default=None
)


def start_request_cost() -> RequestCost:
    cost = RequestCost()
    _request_cost.set(cost)
    return cost


def add_latency(ms: float) -> None:
    """Charge simulated latency to the current request (no-op outside a request)."""
    cost = _request_cost.get()
    if cost is not None:
        cost.latency_ms += ms


# --- Logs -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class LogRecord:
    ts: datetime
    level: LogLevel
    message: str
    route: str | None = None
    component: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ts": self.ts.isoformat(), "level": self.level, "msg": self.message}
        if self.route:
            out["route"] = self.route
        if self.component:
            out["component"] = self.component
        out.update(self.fields)
        return out


class LogStore:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._records: list[LogRecord] = []

    def emit(
        self,
        level: LogLevel,
        message: str,
        *,
        route: str | None = None,
        component: str | None = None,
        ts: datetime | None = None,
        **fields: Any,
    ) -> None:
        self._records.append(
            LogRecord(ts or self._clock.now(), level, message, route, component, fields)
        )

    def query(
        self,
        *,
        level: LogLevel | None = None,
        route: str | None = None,
        since_minutes: float = 15,
        limit: int = 50,
    ) -> list[LogRecord]:
        """Newest first. `level` is a minimum severity (WARNING returns warnings and errors)."""
        cutoff = self._clock.now() - timedelta(minutes=since_minutes)
        min_level = LEVEL_ORDER[level] if level else 0
        matches = [
            r
            for r in reversed(self._records)  # newest-inserted first breaks timestamp ties
            if r.ts > cutoff
            and r.ts <= self._clock.now()
            and LEVEL_ORDER[r.level] >= min_level
            and (route is None or r.route == route)
        ]
        matches.sort(key=lambda r: r.ts, reverse=True)
        return matches[: min(limit, MAX_LOG_LIMIT)]


# --- Metrics --------------------------------------------------------------------------------

GAUGES: dict[str, str] = {
    "memory_mb": "MB",
    "cpu_pct": "%",
    "db_pool_in_use": "connections",
    "db_query_ms": "ms",
    "cache_hit_rate": "ratio",
}
DERIVED: dict[str, str] = {
    "p95_latency_ms": "ms",
    "error_rate": "ratio",
    "request_rate": "requests/min",
    "upstream_status_counts": "count",
}
METRIC_NAMES: tuple[str, ...] = (*DERIVED, *GAUGES)


class UnknownMetricError(ValueError):
    pass


@dataclass(frozen=True)
class RequestSample:
    ts: datetime
    route: str
    method: str
    status: int
    latency_ms: float


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


class MetricStore:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._gauges: dict[str, list[tuple[datetime, float]]] = {name: [] for name in GAUGES}
        self._requests: list[RequestSample] = []
        self._upstream: list[tuple[datetime, int]] = []

    def observe(self, name: str, value: float, *, ts: datetime | None = None) -> None:
        if name not in GAUGES:
            raise UnknownMetricError(name)
        self._gauges[name].append((ts or self._clock.now(), value))

    def observe_request(
        self,
        route: str,
        method: str,
        status: int,
        latency_ms: float,
        *,
        ts: datetime | None = None,
    ) -> None:
        self._requests.append(
            RequestSample(ts or self._clock.now(), route, method, status, latency_ms)
        )

    def observe_upstream(self, status: int, *, ts: datetime | None = None) -> None:
        self._upstream.append((ts or self._clock.now(), status))

    def summary(self, names: list[str] | None = None, window_minutes: float = 15) -> dict[str, Any]:
        names = list(names) if names else list(METRIC_NAMES)
        unknown = [n for n in names if n not in METRIC_NAMES]
        if unknown:
            raise UnknownMetricError(
                f"unknown metrics {unknown}; available: {', '.join(METRIC_NAMES)}"
            )

        now = self._clock.now()
        start = now - timedelta(minutes=window_minutes)
        bucket_minutes = max(1, math.ceil(window_minutes / MAX_BUCKETS))
        n_buckets = math.ceil(window_minutes / bucket_minutes)
        width = timedelta(minutes=bucket_minutes)

        def bucket_of(ts: datetime) -> int | None:
            if not (start < ts <= now):
                return None
            # Buckets are aligned to `now`, so the last bucket always ends at the current time.
            return n_buckets - 1 - int((now - ts) / width)

        def bucket_time(i: int) -> str:
            return (now - width * (n_buckets - 1 - i)).isoformat()

        requests: list[list[RequestSample]] = [[] for _ in range(n_buckets)]
        for r in self._requests:
            if (i := bucket_of(r.ts)) is not None:
                requests[i].append(r)

        metrics: dict[str, Any] = {}
        for name in names:
            if name == "p95_latency_ms":
                values = [p95([r.latency_ms for r in b]) if b else None for b in requests]
            elif name == "error_rate":
                values = [sum(r.status >= 500 for r in b) / len(b) if b else None for b in requests]
            elif name == "request_rate":
                values = [len(b) / bucket_minutes for b in requests]
            elif name == "upstream_status_counts":
                metrics[name] = self._upstream_summary(bucket_of, bucket_time, n_buckets)
                continue
            else:
                gauge: list[list[float]] = [[] for _ in range(n_buckets)]
                for ts, v in self._gauges[name]:
                    if (i := bucket_of(ts)) is not None:
                        gauge[i].append(v)
                values = [sum(b) / len(b) if b else None for b in gauge]
            metrics[name] = _series(values, bucket_time, GAUGES.get(name) or DERIVED[name])

        return {
            "now": now.isoformat(),
            "window_minutes": window_minutes,
            "bucket_minutes": bucket_minutes,
            "metrics": metrics,
        }

    def _upstream_summary(self, bucket_of: Any, bucket_time: Any, n_buckets: int) -> dict:
        buckets: list[dict[str, int]] = [{} for _ in range(n_buckets)]
        totals: dict[str, int] = {}
        for ts, status in self._upstream:
            if (i := bucket_of(ts)) is not None:
                key = str(status)
                buckets[i][key] = buckets[i].get(key, 0) + 1
                totals[key] = totals.get(key, 0) + 1
        return {
            "unit": "count",
            "totals": totals,
            "series": [{"t": bucket_time(i), "counts": b} for i, b in enumerate(buckets)],
        }


def _series(values: list[float | None], bucket_time: Any, unit: str) -> dict[str, Any]:
    present = [v for v in values if v is not None]
    return {
        "unit": unit,
        "latest": _round(present[-1]) if present else None,
        "min": _round(min(present)) if present else None,
        "max": _round(max(present)) if present else None,
        "mean": _round(sum(present) / len(present)) if present else None,
        "series": [{"t": bucket_time(i), "value": _round(v)} for i, v in enumerate(values)],
    }


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 3)
