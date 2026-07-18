# Workflow Run Tracking

Generic durable state tracking for agents and automated workflows built on the LiteLLM proxy.

## The Problem

Agents like [shin-builder](https://github.com/BerriAI/shin-builder) run multi-stage pipelines (triage → plan → implement → PR). Their task state and conversation history lived in memory — a process restart lost everything.

## Three-Table Design

```
WorkflowRun      one instance of work (header + materialized status)
WorkflowEvent    append-only state transitions (source of truth for replay)
WorkflowMessage  conversation inbox/outbox (full content, not truncated)
```

**WorkflowEvent is the source of truth.** `WorkflowRun.status` is a materialized cache updated automatically when events are appended. If you need to debug a run, replay its events.

## API

All endpoints require a valid LiteLLM API key (`Authorization: Bearer sk-...`).

### Runs

```
POST   /v1/workflows/runs                  Create a run
GET    /v1/workflows/runs                  List runs (?workflow_type=&status=)
GET    /v1/workflows/runs/{run_id}         Get run + latest event
PATCH  /v1/workflows/runs/{run_id}         Update status / metadata / output
```

### Events

```
POST   /v1/workflows/runs/{run_id}/events  Append event (auto-updates run status)
GET    /v1/workflows/runs/{run_id}/events  Full event log (ordered by sequence)
```

### Messages

```
POST   /v1/workflows/runs/{run_id}/messages  Append message
GET    /v1/workflows/runs/{run_id}/messages  Conversation history (ordered by sequence)
```

## Quick Start

```bash
# Create a run
curl -X POST http://localhost:4000/v1/workflows/runs \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{"workflow_type": "shin-builder", "metadata": {"title": "Fix login bug"}}'

# {"run_id": "abc-123", "session_id": "xyz-456", "status": "pending", ...}

# Mark step started (sets status → running)
curl -X POST http://localhost:4000/v1/workflows/runs/abc-123/events \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{"event_type": "step.started", "step_name": "grill", "data": {"claude_session_id": "sess-789"}}'

# Store a conversation message
curl -X POST http://localhost:4000/v1/workflows/runs/abc-123/messages \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "What is the expected behavior?", "session_id": "sess-789"}'

# Restart recovery: fetch active runs and resume from last event's data.claude_session_id
curl "http://localhost:4000/v1/workflows/runs?status=running,paused&workflow_type=shin-builder" \
  -H "Authorization: Bearer sk-1234"
```

## Status Auto-Update Rules

When you append an event, the run's status is updated automatically:

| event_type      | run.status |
|-----------------|------------|
| `step.started`  | `running`  |
| `step.failed`   | `failed`   |
| `hook.waiting`  | `paused`   |
| `hook.received` | `running`  |

Set `status = completed` explicitly via PATCH when the workflow finishes.

## Linking to Spend Logs

`WorkflowRun.session_id` is generated automatically (UUID). Pass it as the `x-litellm-session-id` header when making completions through the proxy:

```python
headers = {"x-litellm-session-id": run.session_id}
```

All spend log entries for this run are then tagged automatically. Query cost per run:

```
POST /ui/spend_logs/view_session_spend_logs?session_id={run.session_id}
```

## Sequence Numbers

Sequence numbers on events and messages are assigned server-side (`MAX + 1` per run). Callers never supply them. This guarantees ordering even under concurrent writes.

## Using from shin-builder

Replace the in-memory `tasks.py` dict with calls to these endpoints:

```python
import httpx

class WorkflowRunClient:
    def __init__(self, base_url: str, api_key: str):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def create_task(self, title: str, **metadata) -> dict:
        r = await self._client.post("/v1/workflows/runs", json={
            "workflow_type": "shin-builder",
            "metadata": {"title": title, **metadata},
        })
        r.raise_for_status()
        return r.json()

    async def list_active_tasks(self) -> list:
        r = await self._client.get(
            "/v1/workflows/runs",
            params={"workflow_type": "shin-builder", "status": "running,paused"},
        )
        r.raise_for_status()
        return r.json()["runs"]

    async def transition(self, run_id: str, step_name: str, event_type: str, data: dict = None):
        r = await self._client.post(f"/v1/workflows/runs/{run_id}/events", json={
            "event_type": event_type,
            "step_name": step_name,
            "data": data or {},
        })
        r.raise_for_status()

    async def append_message(self, run_id: str, role: str, content: str, session_id: str = None):
        r = await self._client.post(f"/v1/workflows/runs/{run_id}/messages", json={
            "role": role, "content": content, "session_id": session_id,
        })
        r.raise_for_status()
```

On startup, call `list_active_tasks()` to restore in-flight runs. The last `step.started` event's `data.claude_session_id` gives you the `--resume` ID.
