"""Agent-facing domain types: the alert it receives and the diagnosis it must produce."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, get_args

from pydantic import BaseModel, Field

# A Literal (not an Enum) so the JSON Schema inlines the values; some providers handle
# `$ref`/`$defs` in tool schemas poorly.
RootCause = Literal[
    "bad_deploy_config",
    "bad_deploy_code",
    "memory_leak",
    "dependency_down_db",
    "db_lock_contention",
    "upstream_latency",
    "expired_credential",
    "upstream_rate_limit",
    "cache_failure",
    "disk_full",
    "cpu_saturation",
    "bad_data",
    "no_incident",
    "other",  # escape hatch: always scored incorrect, but better than a forced wrong label
]
ROOT_CAUSES: tuple[str, ...] = get_args(RootCause)


class Alert(BaseModel):
    name: str  # e.g. "HighLatencyP95"
    severity: Literal["info", "warning", "critical"]
    summary: str  # e.g. "p95 latency > 2s on /orders for 5m"
    fired_at: datetime
    labels: dict[str, str] = {}


class Diagnosis(BaseModel):
    """What the agent submits via the submit_diagnosis tool. The proposed fix is never run."""

    root_cause_category: RootCause = Field(
        description="The single category that best explains the alert."
    )
    explanation: str = Field(description="What happened and why, in a few sentences.")
    evidence: list[str] = Field(
        min_length=1,
        description="Specific observations from tool results that support the diagnosis.",
    )
    proposed_fix: str = Field(
        description="The remediation you recommend. It will NOT be executed automatically."
    )
    fix_risk: Literal["low", "medium", "high"] = Field(
        description="Risk of applying the proposed fix."
    )
    confidence: float = Field(ge=0, le=1, description="Confidence in the diagnosis, 0 to 1.")
