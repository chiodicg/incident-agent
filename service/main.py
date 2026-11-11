"""The patient: a small order/checkout service with injectable faults.

Run: uv run uvicorn service.main:app --port 8000
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from service.deps import DependencyError
from service.ops import UNTRACKED_PREFIXES, State, admin_router, ops_router
from service.state import Settings, build_state
from service.telemetry import start_request_cost

APP_OVERHEAD_MS = 6.0


class NewOrder(BaseModel):
    item: str = Field(min_length=1, max_length=100)
    amount: float = Field(gt=0)


class CheckoutRequest(BaseModel):
    order_id: int


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="orders-service")
    app.state.svc = build_state(settings or Settings.from_env())

    @app.middleware("http")
    async def record_traffic(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path.startswith(UNTRACKED_PREFIXES):
            return await call_next(request)

        state = request.app.state.svc
        cost = start_request_cost()
        try:
            response = await call_next(request)
        except Exception as exc:  # unhandled bug in a route -> 500, like a real app
            state.logs.emit(
                "ERROR",
                f"unhandled exception: {type(exc).__name__}: {exc}",
                route=_route_template(request),
            )
            response = JSONResponse(status_code=500, content={"detail": "internal error"})

        route = _route_template(request)
        latency = APP_OVERHEAD_MS + state.rng.uniform(0, 4) + cost.latency_ms
        state.metrics.observe_request(route, request.method, response.status_code, latency)
        state.host.sample()
        state.logs.emit(
            "INFO",
            f"{request.method} {route} {response.status_code} {latency:.0f}ms",
            route=route,
            status=response.status_code,
            latency_ms=round(latency, 1),
        )
        return response

    @app.exception_handler(DependencyError)
    async def dependency_failed(request: Request, exc: DependencyError) -> JSONResponse:
        request.app.state.svc.logs.emit(
            "ERROR",
            f"dependency failure: {exc}",
            route=_route_template(request),
            component=exc.component,
        )
        return JSONResponse(status_code=503, content={"detail": f"{exc.component} unavailable"})

    @app.get("/orders")
    async def list_orders(state: State) -> list[dict]:
        return state.db.list_orders()

    @app.get("/orders/{order_id}")
    async def get_order(order_id: int, state: State) -> dict:
        key = f"order:{order_id}"
        if (cached := state.cache.get(key)) is not None:
            return cached
        order = state.db.get_order(order_id)
        if order is None:
            raise HTTPException(status_code=404, detail="order not found")
        state.cache.set(key, order)
        return order

    @app.post("/orders", status_code=201)
    async def create_order(body: NewOrder, state: State) -> dict:
        order = state.db.create_order(body.item, body.amount)
        state.host.write_audit({"event": "order_created", **order})
        return order

    @app.post("/checkout")
    async def checkout(body: CheckoutRequest, state: State) -> dict:
        order = state.db.get_order(body.order_id)
        if order is None:
            raise HTTPException(status_code=404, detail="order not found")
        charge_id = state.payments.charge(order["id"], order["amount"])
        state.db.mark_paid(order["id"])
        state.cache.invalidate(f"order:{order['id']}")
        state.host.write_audit({"event": "order_paid", "order_id": order["id"]})
        return {"order_id": order["id"], "status": "paid", "charge_id": charge_id}

    app.include_router(ops_router)
    app.include_router(admin_router)
    return app


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", request.url.path)


app = create_app()
