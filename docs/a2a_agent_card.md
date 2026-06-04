# A2A Agent Card

LiteLLM can proxy [A2A-compatible agents](https://a2a-protocol.org/latest/specification/), exposing them to your clients through LiteLLM with virtual keys, team scoping, observability, and a unified agent card.

This page documents which A2A agent card fields LiteLLM supports today, how invocation works, and what to expect from the proxied agent card served at `/a2a/{agent_id}/.well-known/agent.json`.

For provider-specific setup, see:

- [Register a LangGraph Platform agent](./providers/langgraph#register-a-langgraph-platform-agent)

## Agent card support

The fields below mirror the A2A v1.0 specification ([§4.4 Agent Discovery Objects](https://a2a-protocol.org/latest/specification/)). A ✅ means the field is present in the agent card LiteLLM serves to clients; a ❌ means the field is not.

### AgentCard (§4.4.1)

| Field | Supported |
|---|---|
| `name` | ✅ |
| `description` | ✅ |
| `supportedInterfaces` | ✅ |
| `provider` | ✅ |
| `version` | ✅ |
| `documentationUrl` | ✅ |
| `capabilities` | ✅ |
| `securitySchemes` | ✅ |
| `securityRequirements` | ✅ |
| `defaultInputModes` | ✅ |
| `defaultOutputModes` | ✅ |
| `skills` | ✅ |
| `signatures` | ❌ |
| `iconUrl` | ✅ |

### AgentProvider (§4.4.2)

| Field | Supported |
|---|---|
| `url` | ✅ |
| `organization` | ✅ |

### AgentCapabilities (§4.4.3)

| Field | Supported |
|---|---|
| `streaming` | ✅ |
| `pushNotifications` | ❌ |
| `extensions` | ❌ |
| `extendedAgentCard` | ❌ |

### AgentExtension (§4.4.4)

| Field | Supported |
|---|---|
| `uri` | ❌ |
| `description` | ❌ |
| `required` | ❌ |
| `params` | ❌ |

### AgentSkill (§4.4.5)

| Field | Supported |
|---|---|
| `id` | ✅ |
| `name` | ✅ |
| `description` | ✅ |
| `tags` | ✅ |
| `examples` | ✅ |
| `inputModes` | ✅ |
| `outputModes` | ✅ |
| `securityRequirements` | ❌ |

### AgentInterface (§4.4.6)

| Field | Supported |
|---|---|
| `url` | ✅ |
| `protocolBinding` | ✅ |
| `tenant` | ❌ |
| `protocolVersion` | ✅ |

### AgentCardSignature (§4.4.7)

| Field | Supported |
|---|---|
| `protected` | ❌ |
| `signature` | ❌ |
| `header` | ❌ |

## How A2A on LiteLLM works

When you register an A2A agent in LiteLLM:

1. You provide a base URL (and, for some providers, an assistant identifier).
2. LiteLLM fetches the upstream agent card from the agent's `/.well-known/agent-card.json` (or the provider-specific equivalent).
3. You review the parsed card in the LiteLLM UI and choose which skills and fields to expose.
4. LiteLLM saves the curated card and serves it at:

    ```
    GET /a2a/{agent_id}/.well-known/agent.json
    ```

5. Clients invoke the agent at:

    ```
    POST /a2a/{agent_id}
    ```

    using A2A JSON-RPC 2.0 (see [Supported A2A methods](#supported-a2a-methods) below).

## Supported A2A methods

All methods below are accepted on `POST /a2a/{agent_id}` (and `POST /a2a/{agent_id}/message/send` for `message/send`). LiteLLM also accepts the PascalCase aliases from the A2A SDK (for example `GetTask` → `tasks/get`).

| Method | Supported | How LiteLLM handles it |
|---|---|---|
| `message/send` | ✅ | Routed through LiteLLM A2A SDK (`asend_message`) — logging, guardrails, cost tracking |
| `message/stream` | ✅ | Routed through LiteLLM streaming handler — NDJSON/SSE response |
| `tasks/get` | ✅ | JSON-RPC forwarded to the agent's `agent_card_params.url` |
| `tasks/list` | ✅ | JSON-RPC forwarded to upstream |
| `tasks/cancel` | ✅ | JSON-RPC forwarded to upstream |
| `tasks/resubscribe` | ✅ | JSON-RPC forwarded to upstream (streaming/SSE) |
| `tasks/pushNotificationConfig/set` | ✅ | JSON-RPC forwarded to upstream |
| `tasks/pushNotificationConfig/get` | ✅ | JSON-RPC forwarded to upstream |
| `tasks/pushNotificationConfig/list` | ✅ | JSON-RPC forwarded to upstream |
| `tasks/pushNotificationConfig/delete` | ✅ | JSON-RPC forwarded to upstream |
| `agent/getAuthenticatedExtendedCard` | ✅ | JSON-RPC forwarded to upstream; `result.url` rewritten to the proxy |

### PascalCase aliases (SDK)

| SDK / alias name | Wire method |
|---|---|
| `GetTask` | `tasks/get` |
| `ListTasks` | `tasks/list` |
| `CancelTask` | `tasks/cancel` |
| `SubscribeToTask` | `tasks/resubscribe` |
| `CreateTaskPushNotificationConfig` | `tasks/pushNotificationConfig/set` |
| `GetTaskPushNotificationConfig` | `tasks/pushNotificationConfig/get` |
| `ListTaskPushNotificationConfigs` | `tasks/pushNotificationConfig/list` |
| `DeleteTaskPushNotificationConfig` | `tasks/pushNotificationConfig/delete` |
| `GetExtendedAgentCard` | `agent/getAuthenticatedExtendedCard` |

### Requirements

- **Task and push-notification methods** require `agent_card_params.url` pointing at a real A2A JSON-RPC server. LiteLLM forwards the request body unchanged (aside from auth headers).
- **Completion-bridge-only agents** (for example LangGraph/Bedrock AgentCore with `custom_llm_provider` and no `url`) support `message/send` and `message/stream` only. Task APIs return an error if no upstream URL is configured.
- **`message/send` / `message/stream` only:** LiteLLM may strip LiteLLM-specific keys from `params` (for example `guardrails`). Task method `params` are forwarded as-is so A2A fields like `id` are preserved.

### Example: two-step task flow

```bash title="1. Send a message"
curl -X POST "http://localhost:4000/a2a/my-agent" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "r1",
    "method": "message/send",
    "params": {
      "message": {
        "kind": "message",
        "role": "user",
        "messageId": "m1",
        "parts": [{"kind": "text", "text": "Hello"}]
      }
    }
  }'
```

Use `result.id` from the response as the task id:

```bash title="2. Poll task status"
curl -X POST "http://localhost:4000/a2a/my-agent" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "r2",
    "method": "tasks/get",
    "params": {"id": "<task-id-from-step-1>"}
  }'
```

---

## Skill routing

Clients invoke a specific skill by including `skillId` in the message metadata:

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "method": "message/send",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "user",
      "parts": [{"kind": "text", "text": "..."}],
      "metadata": {"skillId": "triage_ticket"}
    }
  }
}
```

LiteLLM forwards the entire message envelope, including metadata, to the upstream agent unchanged. The upstream agent is responsible for reading `skillId` and routing internally.

## Editing the agent card

You can edit supported fields from the agent detail page in the LiteLLM UI. Use the **Re-sync from upstream** button to pick up new skills or capabilities the upstream agent has added since registration; it shows a diff and lets you accept changes selectively.

## Related documentation

- [Register a LangGraph Platform agent](./providers/langgraph#register-a-langgraph-platform-agent)
- [A2A Protocol Specification (v1.0)](https://a2a-protocol.org/latest/specification/)