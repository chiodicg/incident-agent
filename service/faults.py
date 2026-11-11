"""Fault registry: named, switchable failure modes for the service.

A fault only flips knobs on the fake dependencies; the symptoms (errors, latency, health) then
emerge from normal request handling. Clearing faults is done with /admin/reset.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from service.state import ServiceState


@dataclass(frozen=True)
class Fault:
    name: str
    description: str
    apply: Callable[[ServiceState], None]


def _db_down(state: ServiceState) -> None:
    state.db.down = True


FAULTS: dict[str, Fault] = {
    f.name: f
    for f in [
        Fault("db_down", "Orders database refuses connections.", _db_down),
    ]
}


class UnknownFaultError(KeyError):
    pass


def trigger(state: ServiceState, name: str) -> None:
    try:
        fault = FAULTS[name]
    except KeyError as exc:
        raise UnknownFaultError(name) from exc
    fault.apply(state)
    state.active_faults.add(name)
