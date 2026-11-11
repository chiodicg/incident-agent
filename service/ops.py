"""Operational endpoints.

`ops_router` is what the agent's read-only tools call. `admin_router` mutates the service
(reset, clock, deploys, faults) and requires the admin token, which the agent never has.
"""

from __future__ import annotations

import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from service.state import Deploy, ServiceState, build_state
from service.telemetry import LogLevel, UnknownMetricError

# Paths whose requests are not recorded as traffic, so the agent's probes can't alter
# the evidence it is reading.
UNTRACKED_PREFIXES = ("/ops", "/metrics", "/health", "/admin", "/docs", "/openapi.json")


def get_state(request: Request) -> ServiceState:
    return request.app.state.svc


State = Annotated[ServiceState, Depends(get_state)]

ops_router = APIRouter()


@ops_router.get("/ops/logs")
def read_logs(
    state: State,
    level: LogLevel | None = None,
    route: str | None = None,
    since_minutes: Annotated[float, Query(gt=0, le=24 * 60)] = 15,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> dict[str, Any]:
    records = state.logs.query(level=level, route=route, since_minutes=since_minutes, limit=limit)
    return {"count": len(records), "logs": [r.to_dict() for r in records]}


@ops_router.get("/ops/deploys")
def read_deploys(state: State, limit: Annotated[int, Query(ge=1, le=50)] = 5) -> dict[str, Any]:
    return {
        "now": state.clock.now().isoformat(),
        "deploys": [d.model_dump(mode="json") for d in state.deploys.recent(limit)],
    }


@ops_router.get("/metrics")
def read_metrics(
    state: State,
    names: Annotated[list[str] | None, Query()] = None,
    window_minutes: Annotated[float, Query(gt=0, le=24 * 60)] = 15,
) -> dict[str, Any]:
    try:
        return state.metrics.summary(names, window_minutes)
    except UnknownMetricError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@ops_router.get("/health")
def health(state: State) -> JSONResponse:
    components = {
        "app": {"status": "ok", "detail": f"version {state.deploys.recent(1)[0].version}"},
        "db": vars(state.db.health()),
        "cache": vars(state.cache.health()),
        "payments_upstream": vars(state.payments.health()),
        "disk": vars(state.host.disk_health()),
    }
    statuses = {c["status"] for c in components.values()}
    overall = "down" if "down" in statuses else "degraded" if "degraded" in statuses else "ok"
    return JSONResponse(
        status_code=503 if overall == "down" else 200,
        content={"status": overall, "components": components},
    )


# --- Admin ----------------------------------------------------------------------------------


def require_admin(request: Request, x_admin_token: Annotated[str | None, Header()] = None) -> None:
    expected = request.app.state.svc.settings.admin_token
    if not expected or not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="admin token required")


admin_router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


@admin_router.post("/reset")
def reset(request: Request) -> dict[str, str]:
    request.app.state.svc = build_state(request.app.state.svc.settings)
    return {"status": "reset", "now": request.app.state.svc.clock.now().isoformat()}


class AdvanceClock(BaseModel):
    seconds: float = Field(gt=0, le=7 * 24 * 3600)


@admin_router.post("/clock/advance")
def advance_clock(body: AdvanceClock, state: State) -> dict[str, str]:
    state.clock.advance(body.seconds)
    return {"now": state.clock.now().isoformat()}


@admin_router.post("/deploys")
def record_deploy(deploy: Deploy, state: State) -> dict[str, str]:
    state.deploys.record(deploy)
    return {"status": "recorded", "version": deploy.version}
