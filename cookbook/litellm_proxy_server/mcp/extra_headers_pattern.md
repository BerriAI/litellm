# MCP `extra_headers` Pattern

LiteLLM MCP servers support forwarding selected client request headers to upstream MCP servers. This is an **allowlist** pattern: only header names listed in `extra_headers` are forwarded.

## When to use

| Scenario | Config |
|----------|--------|
| Pass caller's bearer token to upstream | `extra_headers: ["Authorization"]` + `auth_type: none` |
| Forward tenant or trace headers | `extra_headers: ["X-Tenant-Id", "X-Request-Id"]` |
| Operator-fixed secrets | Use `static_headers` instead (always sent, overrides caller) |

## Proxy config example

```yaml
mcp_servers:
  github_mcp:
    server_name: github-mcp
    alias: github
    url: https://api.githubcopilot.com/mcp
    transport: http
    auth_type: none
    extra_headers:
      - Authorization
      - X-GitHub-Api-Version
    static_headers:
      X-LiteLLM-Gateway: "true"
```

## Header precedence

When a tool call reaches an OpenAPI-backed or HTTP MCP server:

1. **`static_headers`** (operator-configured) — highest precedence
2. **`extra_headers`** (forwarded from client) — fills in non-conflicting names
3. **BYOK / OAuth tokens** — override `Authorization` when stored per-user

Caller-forwarded headers **cannot** override `static_headers` with the same name (case-insensitive).

## Client request example

```bash
curl -X POST "http://localhost:4000/mcp-rest/tools/call" \
  -H "Authorization: Bearer sk-litellm-key" \
  -H "Authorization-MCP-Github: Bearer ghp_xxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "server_id": "github",
    "name": "search_repositories",
    "arguments": {"query": "litellm"}
  }'
```

For per-server auth headers, use the `Authorization-MCP-<ServerName>` pattern (see `MCPRequestHandler` in `auth/user_api_key_auth_mcp.py`).

## OAuth pass-through

When `auth_type: none` and `extra_headers` includes `Authorization`, you can enable OAuth pass-through so upstream 401 challenges reach the client:

```yaml
mcp_servers:
  upstream_mcp:
    auth_type: none
    oauth_passthrough: true
    extra_headers:
      - Authorization
```

This is distinct from `delegate_auth_to_upstream` (OAuth2 servers only).

## Type reference

```python
# litellm/types/mcp_server/mcp_server_manager.py
extra_headers: Optional[List[str]]  # header names to forward from client → upstream
static_headers: Optional[Dict[str, str]]  # operator headers baked into every call
```

## Related tests

- `tests/test_litellm/proxy/_experimental/mcp_server/test_mcp_hook_extra_headers.py`
- `tests/test_litellm/proxy/_experimental/mcp_server/test_openapi_to_mcp_generator.py` (header merge precedence)