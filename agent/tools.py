"""The agent's read-only diagnostic tools.

Tool descriptions are deliberately neutral: they say what a tool returns, never when to use it
or how expensive it is. Ordering guidance belongs only in the per-condition system prompt, so
the eval compares guidance rather than tool wording. These specs are identical across
conditions.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.models import Diagnosis
from llm.types import ToolCallPart, ToolResultPart, ToolSpec

DEFAULT_MAX_CHARS = 8000  # default get_logs (50 lines) is ~7k chars and must fit
DEFAULT_TIMEOUT_S = 10.0
SUBMIT_TOOL = "submit_diagnosis"

MetricName = Literal[
    "p95_latency_ms",
    "error_rate",
    "request_rate",
    "upstream_status_counts",
    "memory_mb",
    "cpu_pct",
    "db_pool_in_use",
    "db_query_ms",
    "cache_hit_rate",
]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetLogsArgs(_Args):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = Field(
        default=None, description="Minimum severity; WARNING returns warnings and errors."
    )
    route: str | None = Field(
        default=None, description="Only logs for this route template, e.g. /orders/{order_id}."
    )
    since_minutes: float = Field(default=15, gt=0, le=1440, description="Look-back window.")
    limit: int = Field(default=50, ge=1, le=200, description="Max log lines, newest first.")


class GetMetricsArgs(_Args):
    names: list[MetricName] | None = Field(
        default=None, description="Metrics to return; omit for all metrics."
    )
    window_minutes: float = Field(default=15, gt=0, le=1440, description="Look-back window.")


class GetDeploysArgs(_Args):
    limit: int = Field(default=5, ge=1, le=50, description="Number of most recent deploys.")


class CheckHealthArgs(_Args):
    pass


class _Tool(BaseModel):
    name: str
    description: str
    args: type[_Args]
    path: str


DIAGNOSTIC_TOOLS: dict[str, _Tool] = {
    t.name: t
    for t in [
        _Tool(
            name="get_logs",
            description=(
                "Read the service's structured JSON logs, newest first. Each line has a "
                "timestamp, level, message, and where relevant the route, component, HTTP "
                "status and latency."
            ),
            args=GetLogsArgs,
            path="/ops/logs",
        ),
        _Tool(
            name="get_metrics",
            description=(
                "Read time-series metrics for the service, bucketed over a look-back window, "
                "with latest/min/max/mean per metric. Covers request latency, error rate, "
                "request rate, upstream status codes, memory, CPU, DB pool, DB query time "
                "and cache hit rate."
            ),
            args=GetMetricsArgs,
            path="/metrics",
        ),
        _Tool(
            name="get_deploys",
            description=(
                "Read the service's recent deploy history, newest first: version, time, "
                "author, summary and any changed configuration keys."
            ),
            args=GetDeploysArgs,
            path="/ops/deploys",
        ),
        _Tool(
            name="check_health",
            description=(
                "Read the service health endpoint: overall status and per-component status "
                "for the app, database, cache, payments upstream and disk."
            ),
            args=CheckHealthArgs,
            path="/health",
        ),
    ]
}

# /health legitimately answers 503 when a component is down; that is data, not a tool failure.
DATA_STATUSES: dict[str, set[int]] = {"/health": {503}}


def tool_specs() -> list[ToolSpec]:
    """The four diagnostic tools plus submit_diagnosis, in a fixed order."""
    specs = [
        ToolSpec(name=t.name, description=t.description, parameters=schema_for(t.args))
        for t in DIAGNOSTIC_TOOLS.values()
    ]
    specs.append(
        ToolSpec(
            name=SUBMIT_TOOL,
            description=(
                "Submit your final diagnosis and end the investigation. The proposed fix is "
                "recorded for a human to review; it is never executed."
            ),
            parameters=schema_for(Diagnosis),
        )
    )
    return specs


class ToolBox:
    """Executes diagnostic tool calls against the service over HTTP.

    It holds no admin credentials and only knows the fixed read-only paths above, so the agent
    cannot reach /admin or change the service.
    """

    def __init__(self, http: httpx.AsyncClient, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self._http = http
        self._max_chars = max_chars

    @classmethod
    def for_url(cls, base_url: str, **kwargs: Any) -> ToolBox:
        return cls(httpx.AsyncClient(base_url=base_url, timeout=DEFAULT_TIMEOUT_S), **kwargs)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def run(self, call: ToolCallPart) -> ToolResultPart:
        """Never raises for agent mistakes or service failures: they become error results the
        agent can read and react to."""
        if call.name == SUBMIT_TOOL:
            raise ValueError("submit_diagnosis is handled by the agent loop, not the ToolBox")

        tool = DIAGNOSTIC_TOOLS.get(call.name)
        if tool is None:
            available = ", ".join(DIAGNOSTIC_TOOLS)
            return _error(call, f"Unknown tool {call.name!r}. Available: {available}.")

        try:
            args = tool.args.model_validate(call.arguments)
        except ValidationError as exc:
            return _error(
                call, f"Invalid arguments for {call.name}: {format_validation_error(exc)}"
            )

        params = args.model_dump(exclude_none=True)
        try:
            response = await self._http.get(tool.path, params=params)
        except httpx.HTTPError as exc:
            return _error(call, f"Request to {tool.name} failed: {type(exc).__name__}: {exc}")

        body = _pretty(response)
        ok = response.is_success or response.status_code in DATA_STATUSES.get(tool.path, set())
        if not ok:
            message = f"HTTP {response.status_code} from {tool.name}:\n{body}"
            return _error(call, self._truncate(message))
        return ToolResultPart(call_id=call.id, content=self._truncate(body))

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_chars:
            return text
        dropped = len(text) - self._max_chars
        return (
            f"{text[: self._max_chars]}\n...[truncated {dropped} characters; "
            "use the tool's filters to narrow the result]"
        )


def schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema without pydantic's auto-generated titles (noise in the prompt)."""
    return _strip_titles(model.model_json_schema())


def _strip_titles(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            k: _strip_titles(v)
            for k, v in node.items()
            # Drop schema `title` keywords, but keep a *property* that happens to be named title.
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def _pretty(response: httpx.Response) -> str:
    try:
        return json.dumps(response.json(), separators=(",", ":"))  # compact: saves context
    except ValueError:
        return response.text


def format_validation_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, e['loc'])) or '(root)'}: {e['msg']}" for e in exc.errors()
    )


def _error(call: ToolCallPart, message: str) -> ToolResultPart:
    return ToolResultPart(call_id=call.id, content=message, is_error=True)
