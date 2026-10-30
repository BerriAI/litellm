# Gateway catalog pagination

Tools, prompts, resources, and resource templates retain upstream page boundaries. Follow the gateway's opaque `nextCursor` until it is absent. Each request checks the caller's current access and tool permissions. A cursor can be replayed and can be sent to another replica serving the same configuration

Set the same nonempty `LITELLM_SALT_KEY` on every replica. Pagination state uses authenticated encryption with purpose-separated HKDF keys derived only from this value. The master key is never a fallback. Complete single-page lists and direct tool calls work without a salt key; a listing that needs continuation returns an actionable configuration error instead of truncated results. Some SDKs automatically list tools when validating a tool-call response, so those SDK calls also require a salt when that listing is paginated

Cursors expire ten minutes after the first page. Continuations do not extend that deadline. Rotating the salt invalidates all outstanding cursors; clients must start a new listing. During a rolling key change, replicas with different keys cannot accept each other's cursors. Coordinate the change across the fleet

A changed registry, caller scope, or available upstream revision requires a fresh listing. When an upstream exposes a string or integer `_meta.revision`, subsequent pages must retain it. Otherwise consistency follows that upstream's own cursor guarantees. Repeated upstream cursors and the existing upstream page limit stop traversal with an explicit error

Listing failures retain per-server outcome metadata. An incomplete upstream catalog cannot establish a bare tool-name route. Complete initial pages retain legacy bare-name routing. Use the server-prefixed names returned by the gateway for paginated catalogs

For a scoped rollout, keep the previous source build serving the control pool and send only selected clients to a separate candidate pool. All candidate replicas must share configuration and salt. Verify page one on one candidate replica and continuation on another, plus a fresh listing and tool call on the control pool. Do not mirror tool calls between pools

For rollback, stop sending new requests to the candidate pool, drain its in-flight operations, and return selected clients to the control pool. Clients must discard candidate cursors and start a fresh listing when crossing versions; older gateways do not validate these cursors. Keep the registry-revision migration installed when rolling back pagination. Verify a fresh listing and a tool call after switching pools

## Elicitation

Elicitation is disabled by default. Enable it per upstream in `config.yaml`; `allow_elicitation` is currently YAML-only and is not editable through the Admin UI or database-backed server API

```yaml
mcp_servers:
  interactive:
    url: https://mcp.example.com/mcp
    transport: http
    allow_elicitation: true
    timeout: 60
```

The downstream MCP client must advertise the requested form or URL capability during initialization. Use the gateway's legacy SSE endpoint (`/mcp/sse`) for the verified interactive form and URL relay path. The current Streamable HTTP endpoint (`/mcp`) can lose initialization state before a tool call, so even a client that advertised support receives an explicit elicitation error instead of an input request. Successful interactive relay over Streamable HTTP is not currently verified. Stateless calls and LLM tool bridges with no downstream MCP client also receive explicit errors

The relay wait uses the upstream server's existing `timeout` setting, or `LITELLM_MCP_CLIENT_TIMEOUT` (60 seconds by default). The enclosing tool call also retains its existing timeout. Unsupported modes, disconnects and relay failures return errors, never a fabricated user decline. Actual user accept, decline and cancel responses are preserved; cancellation stops the relay


## Guardrail blocks during managed tool execution

Set `on_mcp_block` in a guardrail's `litellm_params`, alongside its provider and mode. It applies to non-streaming requests to `/v1/chat/completions`, `/v1/responses`, and `/v1/messages` when LiteLLM automatically executes MCP tools (`require_approval: never`).

```yaml
guardrails:
  - guardrail_name: mcp-input-filter
    litellm_params:
      guardrail: litellm_content_filter
      mode: pre_mcp_call
      on_mcp_block: raise
      default_on: true
      blocked_words:
        - keyword: confidential-marker
          action: BLOCK
```

| Value | When this guardrail blocks an MCP tool call |
| --- | --- |
| `tool_error` (default) | Return the refusal as tool-result text to the LLM and continue the managed request. |
| `raise` | Stop the managed request with HTTP 400, attributed to the blocking guardrail. Do not execute later tools in that batch or make the model follow-up call. |

`pre_mcp_call` checks the tool name and arguments before execution. `during_mcp_call` checks input concurrently with execution, so the tool may already have run when a refusal arrives. HTTP 400 does not undo tool side effects. Model rounds completed before the block still count toward usage and spend.

The policy belongs to the blocking guardrail. Among parallel refusals, a guardrail configured with `raise` takes precedence. A policy pipeline's `allow` or `next` action still controls whether a detected violation becomes a refusal.

This setting does not change ordinary LLM guardrails, post-MCP output guardrails, tool discovery, or direct MCP protocol/REST error handling. Streaming retains the existing tool-result behavior. Provider outages and authentication failures retain their existing handling rather than being converted into content-policy HTTP 400 responses.
