"""Scenario definitions: one per fault in the catalog (SPEC.md, Fault Catalog).

A scenario says how to put the service into a fault state (see eval/load.py), what alert the
agent receives, and what counts as the right answer. Ground truth lives here and is never
shown to the agent.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from agent.models import Alert, RootCause
from agent.tools import DIAGNOSTIC_TOOLS
from service.faults import FAULTS
from service.state import Deploy

BACKFILL_MINUTES = 30  # simulated history before the alert fires


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AlertSpec(_Frozen):
    """The alert without its timestamp; `fired_at` is the simulated time when seeding ends."""

    name: str
    severity: str
    summary: str
    labels: dict[str, str] = {}

    def fire(self, at: datetime) -> Alert:
        return Alert(fired_at=at, **self.model_dump())


class Evidence(_Frozen):
    """A tool result containing `match` shows the root cause. Used for 'steps to root cause'."""

    tool: str
    match: str = Field(min_length=1)


class ScheduledDeploy(_Frozen):
    """A deploy recorded during the backfill, `minutes_before_alert` before the alert fires."""

    minutes_before_alert: int = Field(ge=1, le=BACKFILL_MINUTES)
    version: str
    author: str
    summary: str
    changed_config: dict[str, str] = {}

    def at(self, ts: datetime) -> Deploy:
        return Deploy(ts=ts, **self.model_dump(exclude={"minutes_before_alert"}))


class Scenario(_Frozen):
    name: str
    faults: tuple[str, ...]  # service fault names, triggered together at onset (15 has two)
    onset_minutes: int = Field(default=5, ge=1, le=BACKFILL_MINUTES)  # before the alert
    deploys: tuple[ScheduledDeploy, ...] = ()
    alert: AlertSpec
    true_category: RootCause
    reference_explanation: str  # for the judges
    secondary_issue: str | None = None  # a real but non-causal problem (fault 15); judges only
    decisive_evidence: tuple[Evidence, ...] = Field(min_length=1)  # any one counts
    cheapest_path: tuple[str, ...] = Field(min_length=1)  # tool names, for step efficiency

    def model_post_init(self, _context: object) -> None:
        unknown = [f for f in self.faults if f not in FAULTS]
        if unknown:
            raise ValueError(f"{self.name}: unknown service faults {unknown}")
        tools = {e.tool for e in self.decisive_evidence} | set(self.cheapest_path)
        if bad := tools - set(DIAGNOSTIC_TOOLS):
            raise ValueError(f"{self.name}: unknown tools {sorted(bad)}")

    def is_decisive(self, tool: str, result: str) -> bool:
        return any(e.tool == tool and e.match in result for e in self.decisive_evidence)


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario(
            name="db_down",
            faults=("db_down",),
            alert=AlertSpec(
                name="HighErrorRate",
                severity="critical",
                summary="5xx error rate above 20% on the orders service for 5 minutes",
                labels={"service": "orders"},
            ),
            true_category="dependency_down_db",
            reference_explanation=(
                "The orders database refuses connections. Every request that needs the DB "
                "fails with 503 (connection refused), so the error rate jumps when the DB goes "
                "down. The app, cache, payments upstream and disk are healthy, and no recent "
                "deploy is involved. Fix: restore the database or its connectivity; a rollback "
                "of the service would not help."
            ),
            decisive_evidence=(
                Evidence(tool="check_health", match='"db":{"status":"down"'),
                Evidence(tool="get_logs", match="db: connection refused"),
            ),
            cheapest_path=("check_health",),
        ),
    ]
}
