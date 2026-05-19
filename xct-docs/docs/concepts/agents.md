---
sidebar_position: 3
---

# Agents

An **agent** is an A2A-protocol-compliant entity registered with the proxy.
It has an agent card (declared via `agent_card_params`), an invocation URL,
optional streaming support, and versioned history.

## Discovery

```http
GET /v1/agents?q=research&category=writing&supports_streaming=true&cursor=&limit=50
```

Returns AgentResponse rows scoped to the caller. Filters are AND-joined;
ordering is `agent_id` ASC for stable cursor pagination.

The agent **card** itself follows the A2A spec and is served at:

```
GET /v1/agents/{agent_id}/.well-known/agent-card.json
```

## Invocation

Standard A2A JSON-RPC 2.0:

```http
POST /v1/a2a/{agent_id}/message/send
Content-Type: application/json
Authorization: Bearer <token>

{
  "jsonrpc": "2.0",
  "id": "req-1",
  "method": "message/send",
  "params": { "message": { "role": "user", "parts": [{ "text": "..." }] } }
}
```

### Streaming

`method: "message/stream"` plus `Accept: text/event-stream` (SDK does this
automatically when `stream=True`):

```
event: a2a.message
data: {"role":"assistant","content":"Hello"}

event: a2a.message
data: {"role":"assistant","content":" world"}
```

NDJSON is the fallback when the client doesn't send the Accept header —
existing A2A SDK clients keep working unchanged.

## Health checks

Per-agent `health_check_enabled` + `health_check_timeout_ms` fields in
`agent_card_params` control the sweep at `GET /v1/agents?health_check=true`.
Agents that fail emit a `agent.healthcheck.failed` webhook (see
[Subscribe to webhooks](../recipes/subscribe-webhook.md)).

## Versioning

Every PUT / PATCH on an agent records a snapshot in
`LiteLLM_AgentVersionTable`. Read history with
`GET /v1/agents/{id}/versions` and roll back with
`POST /v1/agents/{id}/rollback {"version_number": N}` (admin only).
Rollback itself writes a new version row tagged `is_rollback=true` so
the audit trail stays linear.

## Read scope (S3-01 hotfix)

Non-admin callers see: explicit grants from their object_permission **∪**
public agents (`litellm.public_agent_groups`) **∪** agents they own
(`created_by == user_id`). Empty grants do NOT fall back to "see
everything" — historical bug, regression-tested.

## Marketplace import

The dashboard's marketplace UI pulls agent definitions from an external
gateway (`/v1/xct-marketplace/config` returns the URL). Admins click
**Import** → `POST /v1/agents` creates the row locally.

## See also

- **[Recipe: stream an agent invocation](../recipes/stream-agent.md)**
- **[Recipe: import a marketplace agent](../recipes/import-marketplace-agent.md)**
