import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Gateway Auth Reference

LiteLLM exposes two gateway surfaces that share most authentication and authorization primitives but diverge in a few important places. This page is the side-by-side reference: which header does what, where the two surfaces are symmetric, and where they're not. Each section links out to the dedicated page for the deep dive.

| Surface | Endpoints | Dedicated docs |
|---|---|---|
| **MCP Gateway** | `/mcp`, `/{server}/mcp`, `/toolset/{name}/mcp`, `/sse`, `/v1/mcp/...`, `/mcp-rest/...` | [MCP Overview](./mcp) |
| **A2A Agent Gateway** | `/a2a/{agent_id}`, `/a2a/{agent_id}/message/send`, `/v1/agents/...` | [A2A Overview](./a2a) |

---

## 1. Client → LiteLLM (authenticating the caller)

Both surfaces accept the same LiteLLM Virtual Key headers and the same identification headers. The one place they diverge: the MCP **ASGI** routes (the streamable MCP endpoints at `/mcp`, `/{name}/mcp`, `/toolset/{name}/mcp`, `/sse`) bypass the standard FastAPI auth dependency and do not parse the vendor-specific auth aliases (`API-Key`, `x-api-key`, `x-goog-api-key`, `Ocp-Apim-Subscription-Key`) or `x-litellm-tags`. The MCP **REST/management** routes (`/v1/mcp/...`, `/mcp-rest/...`) and **all** A2A routes accept the full header set.

| Header | Purpose | MCP ASGI | MCP REST + A2A |
|---|---|---|---|
| `x-litellm-api-key: Bearer sk-...` | Preferred LiteLLM Virtual Key header. Use whenever the inbound `Authorization` header may carry a different token (OAuth passthrough, OBO, A2A per-user forwarding). | ✓ | ✓ |
| `Authorization: Bearer sk-...` | Standard fallback. Stripped of the `Bearer ` prefix before lookup. | ✓ | ✓ |
| `API-Key`, `x-api-key`, `x-goog-api-key`, `Ocp-Apim-Subscription-Key` | Vendor-specific aliases (Azure, Anthropic, Google AI Studio, Azure APIM). | — | ✓ |
| `x-litellm-end-user-id` | End-user identification. Layers per-end-user budgets, MCP access intersection, and audit log entries on top of the key. `x-litellm-customer-id` is an accepted alias. | ✓ | ✓ |
| `x-litellm-trace-id` | Cross-request correlation ID. Falls back to `x-litellm-session-id` or any matching `x-<vendor>-session-id` header. | ✓ | ✓ |
| `x-litellm-session-id` | Session grouping. Same parse path as trace-id, lower priority. | ✓ | ✓ |
| `x-litellm-tags` | Comma-separated tags for spend-log labeling and tag-based routing. Body field `tags` takes precedence. | — (not parsed on MCP ASGI) | ✓ |
| `x-litellm-mcp-debug: true` | Returns masked diagnostic response headers (`x-mcp-debug-*`). See [MCP OAuth — Debugging](./mcp_oauth#debugging-oauth). | ✓ | — |
| `x-mcp-servers` | Scope a request to specific MCP servers (comma-separated). | ✓ | — |

---

## 2. LiteLLM → Backend (authenticating the gateway to the agent or MCP server)

This is the section where MCP and A2A diverge most. MCP has a first-class `auth_type` field on each server registration. **A2A has no `auth_type` field at all** — the outbound auth mode is inferred from what's present in `litellm_params`.

### MCP — `auth_type` enum

Nine values. The MCP server's outbound `Authorization` header (or per-request SigV4 signature) is determined by `auth_type`. See [MCP Overview — Add HTTP MCP Server](./mcp#add-http-mcp-server) for the full table.

| `auth_type` | Mechanism | Dedicated docs |
|---|---|---|
| `none` | No auth header added | — |
| `api_key` / `bearer_token` / `basic` / `authorization` / `token` | Static header, sent verbatim per call | [MCP Overview](./mcp) |
| `oauth2` | PKCE (interactive) or M2M `client_credentials`. Discriminated by `oauth2_flow`. | [MCP OAuth](./mcp_oauth) |
| `oauth2_token_exchange` | RFC 8693 On-Behalf-Of (OBO) — exchange the caller's bearer token for a scoped MCP token | [MCP OBO Auth](./mcp_obo_auth) |
| `aws_sigv4` | Per-request SigV4 signature using a dedicated MCP-side credential chain | [MCP AWS SigV4](./mcp_aws_sigv4) |

### A2A — auth mode inferred from `litellm_params`

There is no `auth_type` field on an agent. The provider handler picks the auth mechanism from the contents of `litellm_params`:

| Mode | When it fires | Send to backend |
|---|---|---|
| **Bearer / JWT** | `litellm_params.api_key` is set | `Authorization: Bearer <api_key>` |
| **SigV4** (AgentCore only) | `litellm_params.api_key` is unset | Per-request SigV4 via the full AWS credential chain. See [Bedrock AgentCore — A2A Gateway Authentication](./providers/bedrock_agentcore#a2a-gateway-authentication). |
| **Provider-native** | `litellm_params.custom_llm_provider` matches a non-Bedrock provider (Vertex AI Agent Engine, LangGraph, Azure AI Foundry, Pydantic AI) | The provider's normal auth path |

The dual JWT-vs-SigV4 mode is specific to AgentCore. Other A2A providers (Vertex, LangGraph, Azure Foundry) use the provider's own credential conventions — see the relevant provider page under [Providers](./providers).

### Zero-trust add-on (MCP only)

If the MCP server needs to **cryptographically verify** the request came through LiteLLM, layer the [MCP JWT Signer](./mcp_zero_trust) guardrail on top. It signs every outbound tool call with a short-lived RS256 JWT and publishes a JWKS endpoint the MCP server can verify against. This is a guardrail (`guardrail: mcp_jwt_signer`, `mode: pre_mcp_call`), not an `auth_type` — it composes with any `auth_type`.

---

## 3. Per-user header passthrough

Both surfaces let clients forward credentials destined for a specific backend server/agent without admin pre-configuration. The conventions look symmetric but parse differently — be precise when copy-pasting.

| Surface | Prefix | Parse rule | Match against | Example |
|---|---|---|---|---|
| **MCP** | `x-mcp-` | Format: `x-mcp-{server_alias}-{header_name}` | Server's `alias`, then `server_name` (case-insensitive) | `x-mcp-github-authorization: Bearer ghp_...` → server `github`, header `Authorization` |
| **A2A** | `x-a2a-` | Format: `x-a2a-{agent_name_or_id}-{header_name}`; matched against agent's UUID and human-readable name (both tried) | Agent's UUID **and** human-readable name (both tried) | `x-a2a-my-agent-x-api-key: secret` → agent `my-agent`, header `x-api-key` |

Both surfaces also support admin-controlled alternatives that compose with the user passthrough:

| Mechanism | MCP | A2A | Notes |
|---|---|---|---|
| `static_headers: {K: V}` | ✓ | ✓ | Always sent. **Wins over user passthrough** on key conflicts. |
| `extra_headers: [name, name, ...]` | ✓ | ✓ | Admin-allowlist of client header names to forward verbatim. |
| `x-<surface>-<id>-<header>` convention | ✓ (`x-mcp-`) | ✓ (`x-a2a-`) | Client-driven, no admin config needed. |

See [MCP Overview — Forwarding Custom Headers](./mcp#forwarding-custom-headers-to-mcp-servers) and [A2A Agent Authentication Headers](./a2a_agent_headers) for the full mechanics.

---

## 4. Authorization — RBAC and access groups

Both surfaces use the `object_permission` model with intersection-style resolution, but at different depths today. MCP resolves across five levels; A2A across two. The detailed flowcharts and tables live on the dedicated pages:

- [MCP Permission Hierarchy](./mcp_control#permission-hierarchy)
- [A2A Agent Permission Management — How It Works](./a2a_agent_permissions#how-it-works)

| Level | MCP field | A2A field |
|---|---|---|
| **Key** | `object_permission.mcp_servers`, `object_permission.mcp_access_groups`, `object_permission.mcp_tool_permissions` | `object_permission.agents`, `object_permission.agent_access_groups` |
| **Team** | Same | Same (inheritance-first: if the key has no list, it inherits the team's) |
| **End user** | Same (via `x-litellm-end-user-id`) | — not resolved today |
| **Agent** | Same (via `x-litellm-agent-id`) | — not applicable (the agent is the target) |
| **Org** | Same — acts as a **ceiling** | — not resolved today |

| Concern | MCP | A2A |
|---|---|---|
| Per-server / per-agent allowlist | `object_permission.mcp_servers` | `object_permission.agents` |
| Access groups (tag-based grants) | `object_permission.mcp_access_groups` | `object_permission.agent_access_groups` |
| Per-server tool-level allowlist | `object_permission.mcp_tool_permissions: {server_id: [tool, ...]}` | n/a (tools live inside the agent) |
| Server-registration allowlist (admin-static) | `allowed_tools` / `disallowed_tools` on the MCP server | n/a |
| Param-level allowlist | `allowed_params: {tool_name: [param, ...]}` on the MCP server | n/a |
| Reject behaviour | `list_tools` filters out hidden servers; `call_tool` returns error | `GET /v1/agents` filters; `POST /a2a/{agent_id}` returns HTTP **403** |

---

## 5. Trace IDs and identity propagation

`x-litellm-trace-id` is **accepted** on every request and threaded through logging on both surfaces. A few A2A-specific extras:

| Setting | Scope | Behaviour |
|---|---|---|
| `require_trace_id_on_calls_to_agent: true` | Per-agent, on the agent's `litellm_params` | Reject inbound `/a2a/{agent_id}` calls missing `x-litellm-trace-id` (or `x-litellm-session-id` fallback) with **HTTP 400**. See [A2A Overview — Trace ID enforcement](./a2a#trace-id-enforcement-optional-per-agent). |
| `require_trace_id_on_calls_by_agent: true` | Per-agent, on the agent's `litellm_params` | Reverse direction — when a key **owned by** that agent makes outbound calls, require a trace ID on those. |

**Sub-agent identity propagation** — when LiteLLM dispatches a downstream call as part of an A2A invocation, it forwards `X-LiteLLM-Trace-Id` and `X-LiteLLM-Agent-Id` to maintain trace continuity and spend attribution. The original virtual key and end-user identity are **not** auto-forwarded. Use `extra_headers` or the `x-a2a-{agent_name_or_id}-{header}` convention to thread identity explicitly. See [A2A Overview — Sub-agent identity propagation](./a2a#sub-agent-identity-propagation).

---

## 6. Guardrails on the gateway path

| Concern | MCP | A2A |
|---|---|---|
| Pre-call input guardrails (Presidio, Bedrock, Lakera, Aporia, etc.) | `mode: pre_mcp_call` | Standard chat-completion guardrails apply to the underlying LLM calls the agent makes |
| During-call intervention | `mode: during_mcp_call` | — |
| Zero-trust JWT signing | [`mcp_jwt_signer` guardrail](./mcp_zero_trust) | — (not applicable to A2A today) |
| Documentation | [MCP Guardrails](./mcp_guardrail), [MCP Zero Trust](./mcp_zero_trust) | Standard [guardrails docs](./proxy/guardrails) apply via the agent's underlying model calls |

---

## 7. Cheatsheet — what header does what

For copy-paste, the high-frequency request headers across both surfaces:

```http
# Always (LiteLLM-side auth and identification)
x-litellm-api-key: Bearer sk-...
# or
Authorization: Bearer sk-...

x-litellm-end-user-id: user-42
x-litellm-trace-id: 8f4a-2b1c-d3e5-...

# MCP — server scoping / per-user passthrough
x-mcp-servers: github,zapier
x-mcp-github-authorization: Bearer ghp_<user-token>     # user passthrough to github_mcp
x-litellm-mcp-debug: true                                # diagnostic response headers

# A2A — per-user passthrough
x-a2a-my-agent-authorization: Bearer <user-token>        # caller's token to my-agent
x-a2a-my-agent-x-api-key: <user-key>                     # additional per-agent header
```

For the deep dives, follow the cross-links above into the dedicated pages.
