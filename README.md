# incident-agent

An LLM incident responder for a running service. It investigates an alert with read-only tools
(logs, metrics, deploy history, health check) under a step budget, forms a root-cause hypothesis,
and proposes — never executes — a fix. An eval harness seeds 17 faults and compares models,
prompt conditions and step budgets.

See [SPEC.md](SPEC.md) for the design. Work in progress.

## Quickstart

```bash
uv sync
cp .env.example .env   # add API keys
uv run pytest
```
