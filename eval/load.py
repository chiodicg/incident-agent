"""Put the service into a scenario's state: seeded history, noise, the fault, then a live burst.

    uv run python -m eval.load db_down --seed 3        # against $SERVICE_URL, prints the alert

Timeline (simulated clock, starting at the service's start time):

    reset(seed) ── 30 min of traffic + noise ─────────────────── live burst ── alert fires
                              ▲ deploys at their minute
                              ▲ faults triggered `onset_minutes` before the alert

Everything goes through the service's HTTP API (admin endpoints for setup, normal routes for
traffic), so the symptoms emerge from real request handling, and the same code seeds an
in-process app in tests or a running server. Requests are sent one at a time: with the same
seed, the service records identical telemetry.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
from dataclasses import dataclass
from datetime import datetime

import httpx
from dotenv import load_dotenv

from agent.models import Alert
from eval.scenarios import BACKFILL_MINUTES, SCENARIOS, Scenario

REQUESTS_PER_MINUTE = (6, 10)
NOISE_PROBABILITY_PER_MINUTE = 0.35
KNOWN_ORDER_IDS = range(1, 6)

# Background noise: present in every scenario so no single log query gives the answer away.
# Deliberately unrelated to every fault; a test checks none matches any decisive evidence.
NOISE_LOGS: tuple[dict[str, str], ...] = (
    {"level": "WARNING", "msg": "client disconnected before response was sent", "route": "/orders"},
    {"level": "WARNING", "msg": "deprecated header X-Client-Version used by client"},
    {"level": "WARNING", "msg": "request body above soft size limit (12 KB)", "route": "/orders"},
    {
        "level": "WARNING",
        "msg": "payment status poll retried (attempt 2/3), then succeeded",
        "component": "payments_upstream",
    },
    {"level": "INFO", "msg": "config reloaded: no changes"},
    {"level": "WARNING", "msg": "slow request: GET /orders took 212ms (threshold 200ms)"},
    {"level": "INFO", "msg": "cache: evicted 14 expired keys", "component": "cache"},
)


@dataclass
class _Admin:
    http: httpx.AsyncClient
    token: str
    now: datetime | None = None

    async def post(self, path: str, json: dict | None = None) -> dict:
        response = await self.http.post(path, json=json, headers={"X-Admin-Token": self.token})
        response.raise_for_status()
        body = response.json()
        if "now" in body:
            self.now = datetime.fromisoformat(body["now"])
        return body


async def seed(
    scenario: Scenario, http: httpx.AsyncClient, *, admin_token: str, seed: int
) -> Alert:
    """Reset the service and replay the scenario. Returns the alert the agent should receive."""
    rng = random.Random(seed)  # traffic and noise; the service has its own RNG, reseeded below
    admin = _Admin(http, admin_token)
    await admin.post("/admin/reset", {"seed": seed})

    for minute in range(BACKFILL_MINUTES):
        minutes_before_alert = BACKFILL_MINUTES - minute
        for deploy in scenario.deploys:
            if deploy.minutes_before_alert == minutes_before_alert:
                payload = deploy.at(admin.now).model_dump(mode="json")
                await admin.post("/admin/deploys", payload)
        if minutes_before_alert == scenario.onset_minutes:
            for fault in scenario.faults:
                await admin.post(f"/admin/faults/{fault}")
        await _traffic(http, rng)
        if rng.random() < NOISE_PROBABILITY_PER_MINUTE:
            await admin.post("/admin/logs", rng.choice(NOISE_LOGS))
        await admin.post("/admin/clock/advance", {"seconds": 60})

    await _traffic(http, rng)  # live burst at alert time
    assert admin.now is not None
    return scenario.alert.fire(admin.now)


async def _traffic(http: httpx.AsyncClient, rng: random.Random) -> None:
    """One simulated minute of user traffic. Status codes are whatever the service returns."""
    for _ in range(rng.randint(*REQUESTS_PER_MINUTE)):
        roll = rng.random()
        if roll < 0.40:
            await http.get("/orders")
        elif roll < 0.75:
            await http.get(f"/orders/{rng.choice(KNOWN_ORDER_IDS)}")
        elif roll < 0.85:
            item = rng.choice(["book", "lamp", "mug", "chair"])
            await http.post("/orders", json={"item": item, "amount": rng.randint(5, 80)})
        elif roll < 0.95:
            await http.post("/checkout", json={"order_id": rng.choice(KNOWN_ORDER_IDS)})
        else:
            await http.get(f"/orders/{rng.randint(9000, 9999)}")  # stale link: a normal 404


# --- CLI ------------------------------------------------------------------------------------


async def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="eval.load", description="Seed a running service.")
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--service-url", default=os.environ.get("SERVICE_URL"))
    parser.add_argument("--alert-out", help="also write the alert JSON here (for agent.cli)")
    args = parser.parse_args(argv)

    token = os.environ.get("ADMIN_TOKEN")
    if not token:
        parser.error("ADMIN_TOKEN is not set")
    async with httpx.AsyncClient(base_url=args.service_url or "http://localhost:8000") as http:
        alert = await seed(SCENARIOS[args.scenario], http, admin_token=token, seed=args.seed)
    alert_json = alert.model_dump_json(indent=2)
    print(alert_json)
    if args.alert_out:
        with open(args.alert_out, "w") as f:
            f.write(alert_json + "\n")


if __name__ == "__main__":
    asyncio.run(main())
