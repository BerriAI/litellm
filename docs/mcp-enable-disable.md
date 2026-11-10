# Enabling and disabling MCP servers

Admins can switch an MCP server off without deleting it. A disabled server keeps
its configuration, grants and access groups, but is no longer served to callers.

## Endpoints

Both require the `PROXY_ADMIN` role, are idempotent, and return the server row
including `enabled`. Unknown `server_id` returns `404`.

```
PUT /v1/mcp/server/{server_id}/disable
PUT /v1/mcp/server/{server_id}/enable
```

## The `enabled` field

- `GET /v1/mcp/server` and `GET /v1/mcp/server/{server_id}` include `enabled`.
- DB-registered servers store it as a column; existing rows default to enabled.
- Config-file (`mcp_servers:` in `config.yaml`) servers are read-only, so their
  toggle is stored as an override (`LiteLLM_MCPServerStateOverride`) keyed by
  server id. The config is not modified. If the server is later removed from
  the config, the stale override is ignored.

## Behavior when disabled

- Tool listing and tool calls (MCP endpoints, `/mcp-rest/*`, OAuth flows) treat
  the server as absent; calls to it are rejected.
- Admin listings (`GET /v1/mcp/server`, dashboard) still show it with
  `enabled: false` so it can be re-enabled.
- Changes apply live on the serving pod and propagate to other pods through the
  usual MCP cross-pod sync; no restart is needed.
- The dashboard MCP server card grid has an admin-only toggle for each server.
