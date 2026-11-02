# Incident responder for a running service.

- Deploy a small FastAPI app with deliberate faults you can trigger.
- Implement the agent connected to the app.
- The agent gets an alert, decides whether to read logs, query metrics, check recent deploys or hit a health endpoint, forms a hypothesis, and proposes (but does not execute) a fix. 
- Tools: logs, metrics, deploy history, health check. The decision: diagnostic order under a step budget. 
- Eval: a dozen seeded faults, scored on correct root cause, steps taken and whether it ever proposed something destructive.

## Scope
- This is for portfolio/learning
- The small fastAPI app can be built by code agents, but I want to do pair programming with agentic workflow.