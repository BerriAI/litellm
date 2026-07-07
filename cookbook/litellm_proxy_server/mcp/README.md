# LiteLLM Proxy MCP REST API

The LiteLLM proxy exposes MCP tools over HTTP at `/mcp-rest/*`. These endpoints let you list and call MCP tools without an MCP SDK client.

**Auth:** All routes require a LiteLLM proxy API key in the `Authorization` header:

```http
Authorization: Bearer sk-...
```

See also: [mcp_with_litellm_proxy.py](./mcp_with_litellm_proxy.py) for the Responses API + MCP integration example.

## `POST /mcp-rest/tools/call`

Call a tool on a configured MCP server.

### Request body schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `server_id` | string | **Yes** | MCP server UUID, or server name/alias (resolved server-side). |
| `name` | string | **Yes** | Tool name as returned by `/mcp-rest/tools/list`. |
| `arguments` | object | No | Tool input arguments. Defaults to `{}` when omitted. |

### Example: call a tool

```bash
curl -X POST "http://localhost:4000/mcp-rest/tools/call" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "server_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "name": "search_repositories",
    "arguments": {
      "query": "litellm",
      "limit": 5
    }
  }'
```

### Example: use server alias instead of UUID

`server_id` accepts the server's configured name or alias:

```json
{
  "server_id": "github",
  "name": "get_file_contents",
  "arguments": {
    "owner": "BerriAI",
    "repo": "litellm",
    "path": "README.md"
  }
}
```

### Error responses

| HTTP status | `error` field | When |
|-------------|---------------|------|
| 400 | `missing_parameter` | `server_id` or `name` is absent from the body. |
| 403 | `access_denied` | Key lacks permission for the requested server or tool. |
| 403 | `forbidden` | Virtual tool (`mcp_tool_search` / `mcp_tool_call`) used without `mcp_tool_search_enabled` on the key. |
| 412 | `missing_user_env_vars` | Per-user env vars are required but not configured. |
| 500 | `internal_server_error` | Unexpected server error. Check proxy logs. |

#### Missing required fields

```json
{
  "detail": {
    "error": "missing_parameter",
    "message": "server_id is required in request body"
  }
}
```

### Virtual tool search path

When `mcp_tool_search_enabled` is set on the API key, you can call virtual tools without `server_id`:

```json
{
  "name": "mcp_tool_search",
  "arguments": {
    "query": "create pull request",
    "top_k": 5
  }
}
```

```json
{
  "name": "mcp_tool_call",
  "arguments": {
    "tool_name": "github-create_pull_request",
    "arguments": {
      "owner": "BerriAI",
      "repo": "litellm",
      "title": "Docs: MCP REST body schema"
    }
  }
}
```

## `GET /mcp-rest/tools/list`

List tools available to the authenticated key.

### Query parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `server_id` | string | No | Filter to one server by UUID. |
| `mcp_server_name` | string | No | Filter by server name or alias. |
| `toolset_name` | string | No | Filter to a named toolset scope. |
| `include_disabled_tools` | boolean | No | Admin only. Return full catalog for allowlist configuration. |

### Example

```bash
curl "http://localhost:4000/mcp-rest/tools/list?mcp_server_name=github" \
  -H "Authorization: Bearer sk-1234"
```

### Response shape

```json
{
  "tools": [
    {
      "name": "search_repositories",
      "description": "Search GitHub repositories",
      "inputSchema": { "type": "object", "properties": {} },
      "mcp_info": {
        "server_name": "github-mcp",
        "server_id": "a1b2c3d4-...",
        "alias": "github"
      }
    }
  ],
  "error": null,
  "message": "Successfully retrieved tools"
}
```

## Related endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /mcp-rest/test/connection` | Test connectivity before saving a server in the UI. |
| `POST /mcp-rest/test/tools/list` | Test tool discovery with draft server config. |

Implementation reference: `litellm/proxy/_experimental/mcp_server/rest_endpoints.py`.