import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# A2A Agent Authentication Headers

Forward authentication credentials (Bearer tokens, API keys, etc.) from clients to backend A2A agents.

## Overview

When LiteLLM proxies a request to a backend A2A agent, the agent may require its own authentication headers. There are three ways to supply them:

| Method | Who configures | How it works |
|---|---|---|
| **Static headers** | Admin (UI / API) | Always sent, regardless of client request |
| **Forward client headers** | Admin (UI / API) | Header names to extract from client request and forward |
| **Convention-based** | Client (no admin config) | Client sends `x-a2a-{agent_name}-{header}` — automatically routed |

All three methods can be combined. **Static headers always win** on key conflicts.

---

## Method 1 — Static Headers

Admin-configured headers that are always sent to the backend agent. Use this for server-to-server tokens or internal credentials that clients should never see or override.

<Tabs>
<TabItem value="ui" label="UI">

1. Go to **Agents** in the LiteLLM dashboard.
2. Create or edit an agent.
3. Open the **Authentication Headers** panel.
4. Under **Static Headers**, click **Add Static Header** and fill in the header name and value.

</TabItem>
<TabItem value="api" label="REST API">

```bash
curl -X POST http://localhost:4000/v1/agents \
  -H "Authorization: Bearer sk-admin" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "my-agent",
    "agent_card_params": { ... },
    "static_headers": {
      "Authorization": "Bearer internal-server-token",
      "X-Internal-Service": "litellm-proxy"
    }
  }'
```

To update an existing agent:

```bash
curl -X PATCH http://localhost:4000/v1/agents/{agent_id} \
  -H "Authorization: Bearer sk-admin" \
  -H "Content-Type: application/json" \
  -d '{
    "static_headers": {
      "Authorization": "Bearer new-token"
    }
  }'
```

</TabItem>
</Tabs>

**Client call — no special headers needed:**

```bash
curl -X POST http://localhost:4000/a2a/my-agent \
  -H "Authorization: Bearer sk-client-key" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0", "id": "1", "method": "message/send",
    "params": { "message": { "role": "user", "parts": [{"kind": "text", "text": "Hello"}], "messageId": "msg-1" } }
  }'
```

The backend agent receives `Authorization: Bearer internal-server-token` without the client ever knowing the value.

---

## Method 2 — Forward Client Headers

Admin specifies a list of header **names**. When the client sends a request that includes those headers, LiteLLM extracts their values and forwards them to the backend agent. The client controls the values; the admin controls which headers are eligible to be forwarded.

<Tabs>
<TabItem value="ui" label="UI">

1. Go to **Agents** in the LiteLLM dashboard.
2. Create or edit an agent.
3. Open the **Authentication Headers** panel.
4. Under **Forward Client Headers**, type header names and press **Enter** (e.g. `x-api-key`, `Authorization`).

</TabItem>
<TabItem value="api" label="REST API">

```bash
curl -X POST http://localhost:4000/v1/agents \
  -H "Authorization: Bearer sk-admin" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "my-agent",
    "agent_card_params": { ... },
    "extra_headers": ["x-api-key", "x-user-token"]
  }'
```

</TabItem>
</Tabs>

**Client call — include the forwarded headers:**

```bash
curl -X POST http://localhost:4000/a2a/my-agent \
  -H "Authorization: Bearer sk-client-key" \
  -H "x-api-key: user-secret-value" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

The backend agent receives `x-api-key: user-secret-value`.

:::note
Header name matching is **case-insensitive**. If the client sends `X-API-Key` and `extra_headers` lists `x-api-key`, they match.
:::

---

## Method 3 — Convention-Based Forwarding

Clients can forward headers to a specific agent without any admin pre-configuration by using the naming convention:

```
x-a2a-{agent_name_or_id}-{header_name}: value
```

LiteLLM parses these headers automatically and routes them to the matching agent only.

**Examples:**

| Client header sent | Agent name/ID | Forwarded as |
|---|---|---|
| `x-a2a-my-agent-authorization: Bearer tok` | `my-agent` | `authorization: Bearer tok` |
| `x-a2a-my-agent-x-api-key: secret` | `my-agent` | `x-api-key: secret` |
| `x-a2a-abc123-authorization: Bearer tok` | agent ID `abc123` | `authorization: Bearer tok` |

```bash
curl -X POST http://localhost:4000/a2a/my-agent \
  -H "Authorization: Bearer sk-client-key" \
  -H "x-a2a-my-agent-authorization: Bearer agent-specific-token" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

The `x-a2a-other-agent-authorization` header sent in the same request is **not** forwarded to `my-agent` — it is silently ignored.

:::tip Matches both agent name and agent ID
Both the human-readable name (e.g. `my-agent`) and the UUID (e.g. `abc123-...`) are valid. Use whichever is convenient for the client.
:::

---

## Merge Precedence

When multiple methods supply the same header name, **static headers win**:

```
dynamic (forwarded/convention)  →  merged  ←  static (overlays, wins)
```

Example:

| Source | `Authorization` value |
|---|---|
| Client sends (via `extra_headers` or convention) | `Bearer client-token` |
| Admin-configured `static_headers` | `Bearer server-token` |
| **What the backend agent receives** | **`Bearer server-token`** |

This ensures admin-controlled credentials cannot be overridden by client requests.

---

## Combining All Three Methods

```bash
# Register agent with static + forwarded headers
curl -X POST http://localhost:4000/v1/agents \
  -H "Authorization: Bearer sk-admin" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "my-agent",
    "agent_card_params": { ... },
    "static_headers": {
      "X-Internal-Token": "secret123"
    },
    "extra_headers": ["x-user-id"]
  }'

# Client call using all three mechanisms
curl -X POST http://localhost:4000/a2a/my-agent \
  -H "Authorization: Bearer sk-client-key" \
  -H "x-user-id: user-42" \
  -H "x-a2a-my-agent-x-request-id: req-abc" \
  -H "Content-Type: application/json" \
  -d '{ ... }'
```

The backend agent receives:

```
X-Internal-Token: secret123          ← static header (always)
x-user-id: user-42                   ← forwarded (in extra_headers)
x-request-id: req-abc                ← convention-based (x-a2a-my-agent-*)
X-LiteLLM-Trace-Id: <uuid>           ← LiteLLM internal
X-LiteLLM-Agent-Id: <agent-id>       ← LiteLLM internal
```

---

## Header Isolation

Each agent invocation uses an isolated HTTP connection. Headers configured for agent A are **never** sent to agent B, even if both agents are running and receiving requests simultaneously.

---

## API Reference

### `POST /v1/agents` / `PATCH /v1/agents/{agent_id}`

| Field | Type | Description |
|---|---|---|
| `static_headers` | `object` | `{"Header-Name": "value"}` — always forwarded |
| `extra_headers` | `string[]` | Header names to extract from client request and forward |

### Agent Response

Both fields are returned in `GET /v1/agents` and `GET /v1/agents/{agent_id}`:

```json
{
  "agent_id": "...",
  "agent_name": "my-agent",
  "static_headers": { "X-Internal-Token": "secret123" },
  "extra_headers": ["x-user-id"],
  ...
}
```

:::caution
`static_headers` values are stored in the database and returned by the API. Treat them as you would any credential — do not store sensitive long-lived tokens here if your API is publicly accessible. Consider using short-lived tokens or environment-injected secrets instead.
:::
