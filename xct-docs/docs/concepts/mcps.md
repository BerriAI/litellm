---
sidebar_position: 4
---

# MCPs

An **MCP** is a Model Context Protocol server — a remote process that
exposes a set of tools. The proxy proxies tool calls, tracks per-tool
spend, handles OAuth / OBO / PKCE-passthrough, and applies a 6-level
permission cascade.

## The 6-level cascade (from S3 backend)

Access resolution stops at the first match:

1. **Key-level** — explicit `mcp_servers` on the VerificationToken
2. **Team-level** — inherited via `team.object_permission`
3. **Org-level**
4. **User-level**
5. **Agent-level** (for keys minted via A2A invocation)
6. **Toolset-level** — named bundles of `(server, tool)` pairs

OAuth credentials and BYOK credentials share `LiteLLM_MCPUserCredentials`
(distinguished by `type` field in the JSON payload).

## Discovery

```python
tools = xct.mcp.list_tools()    # GET /v1/mcp/tools
```

Returns OpenAI-shape function definitions that you can pass straight to
`xct.chat.completions.create(tools=...)`. The proxy handles tool execution.

## Invocation modes

### Through `/v1/chat/completions`
```python
reply = xct.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    tools=tools,                 # MCP tools from .list_tools()
    extra_headers={"x-mcp-servers": "github,search"},  # narrow to specific MCPs
)
```

The proxy intercepts tool calls in the response, invokes the MCP server,
threads results back into the conversation.

### Directly via `/mcp-rest/`
```http
POST /mcp-rest/tools/call
{ "tool_name": "github.search_issues", "arguments": {"q": "..."} }
```
Same auth model; same tool name format `{server_alias}.{tool_name}`.

### URL-namespaced
For a single server or access group:
```
POST /github/mcp/v1/chat/completions      ← only github MCP tools available
POST /research-tools,coding-tools/mcp/v1/chat/completions   ← access-group bundle
```

## Auth modes (`auth_type`)

| Type | Notes |
|---|---|
| `none` | Public MCP |
| `api_key` | Single static key from `credentials` JSON |
| `bearer_token` | `Authorization: Bearer ...` |
| `basic` | HTTP basic |
| `oauth2` | Stored OAuth flow (use `/v1/mcp/server/{id}/oauth/...` to walk through it) |
| `oauth2_token_exchange` | RFC 8693 OBO — `subject_token` is the caller's JWT |
| `delegate_auth_to_upstream` | PKCE-passthrough — proxy doesn't see the secret |

## Approval workflow

Non-admin keys can `POST /v1/mcp/server/register` with their own MCP
proposal; admins review at `/v1/mcp/server/submissions`. Until approved,
the row stays `approval_status="pending_review"` and isn't routable.

## See also

- **[Recipe: use an MCP tool in chat](../recipes/use-mcp-tool.md)**
- Upstream docs on MCP: https://docs.litellm.ai/docs/mcp
