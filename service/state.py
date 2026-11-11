"""Service settings and the mutable state behind one app instance."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel

from service.deps import Cache, Database, Host, PaymentUpstream
from service.telemetry import LogStore, MetricStore, SimClock

DEFAULT_START = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)


class Settings(BaseModel):
    admin_token: str | None = None
    seed: int = 0
    start_time: datetime = DEFAULT_START
    db_path: str = ":memory:"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            admin_token=os.environ.get("ADMIN_TOKEN") or None,
            seed=int(os.environ.get("SERVICE_SEED", "0")),
        )


class Deploy(BaseModel):
    version: str
    ts: datetime
    author: str
    summary: str
    changed_config: dict[str, Any] = {}


@dataclass
class DeployHistory:
    deploys: list[Deploy] = field(default_factory=list)

    def record(self, deploy: Deploy) -> None:
        self.deploys.append(deploy)

    def recent(self, limit: int = 5) -> list[Deploy]:
        return sorted(self.deploys, key=lambda d: d.ts, reverse=True)[:limit]


def baseline_deploys(now: datetime) -> list[Deploy]:
    """Unremarkable history, so the deploy tool never returns an empty, suspicious list."""
    return [
        Deploy(
            version="v1.3.0",
            ts=now - timedelta(days=6, hours=3),
            author="priya",
            summary="Add order status endpoint",
        ),
        Deploy(
            version="v1.3.1",
            ts=now - timedelta(days=3, hours=7),
            author="marco",
            summary="Bump httpx; tidy logging",
        ),
        Deploy(
            version="v1.3.2",
            ts=now - timedelta(days=1, hours=2),
            author="priya",
            summary="Copy changes on checkout errors",
        ),
    ]


@dataclass
class ServiceState:
    settings: Settings
    clock: SimClock
    rng: random.Random
    logs: LogStore
    metrics: MetricStore
    deploys: DeployHistory
    db: Database
    cache: Cache
    payments: PaymentUpstream
    host: Host
    active_faults: set[str] = field(default_factory=set)


def build_state(settings: Settings) -> ServiceState:
    clock = SimClock(settings.start_time)
    rng = random.Random(settings.seed)
    logs = LogStore(clock)
    metrics = MetricStore(clock)
    return ServiceState(
        settings=settings,
        clock=clock,
        rng=rng,
        logs=logs,
        metrics=metrics,
        deploys=DeployHistory(baseline_deploys(clock.now())),
        db=Database(rng, metrics, settings.db_path),
        cache=Cache(metrics),
        payments=PaymentUpstream(rng, metrics),
        host=Host(rng, metrics),
    )
