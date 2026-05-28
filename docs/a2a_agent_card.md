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

    using standard A2A JSON-RPC (`message/send`, `message/stream`).

## Supported A2A methods

| Method | Supported |
|---|---|
| `message/send` | ✅ |
| `message/stream` | ✅ |
| `tasks/get` | ❌ |
| `tasks/cancel` | ❌ |
| `tasks/list` | ❌ |
| `tasks/resubscribe` | ❌ |
| `tasks/pushNotificationConfig/set` | ❌ |
| `tasks/pushNotificationConfig/get` | ❌ |
| `tasks/pushNotificationConfig/list` | ❌ |
| `tasks/pushNotificationConfig/delete` | ❌ |
| `agent/getAuthenticatedExtendedCard` | ❌ |

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