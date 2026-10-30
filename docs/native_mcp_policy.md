# Native MCP policy configuration

This change adds validated configuration and storage. Defining a policy does not yet enforce requests or responses; enforcement is added by the dependent changes.

## Configure a team or key

Proxy admins can set `native_mcp_policy` through the existing `/team/new`, `/team/update`, `/key/generate`, and `/key/update` APIs. Key regeneration preserves the policy unless an authorized update changes it. The field has a validated schema and is stored in the existing `metadata.native_mcp_policy` entry. No database migration or proxy configuration flag is needed. Existing info APIs expose the stored policy in metadata.

```json
{
  "team_id": "engineering",
  "native_mcp_policy": {
    "mcp_servers": [
      {
        "name": "github",
        "url": "https://mcp.example.com/github",
        "tools": ["search_code", "get_file"]
      }
    ]
  }
}
```

Each server grant requires `name`, `url`, and `tools`. Names and URLs match exactly. Use the same complete URL in the policy and attachment, including path and query string. URLs must use HTTP or HTTPS and cannot contain credentials or fragments. There are no server-name patterns, regex rules, or registry lookups.

Use a list of exact tool names to restrict a server. Use the string `"*"` for an explicit whole-server grant:

```json
{"name": "github", "url": "https://mcp.example.com/github", "tools": "*"}
```

To preserve unrelated metadata, update the top-level `native_mcp_policy` field. Direct metadata writes are also validated and permission checked. Metadata replacement follows the existing management API behavior. Replacing metadata with a blob that omits its policy removes that policy and therefore requires proxy-admin permission. Key holders and team admins cannot broaden or remove grants through either write path.

## Policy evaluation

The shared evaluator resolves team and key grants from the authenticated identity, then checks normalized server names, URLs, and tool scopes. It performs no database reads and returns an allow/deny result. Endpoint enforcement is not connected in this change.

Team and key grants are unioned for the same server name and URL. A key can add tools but cannot revoke a team grant. A whole-server grant at either level permits every tool on that server. Grants for different URLs are never combined.

| Configuration | Behavior |
|---|---|
| Neither team nor key defines a policy | Existing behavior is preserved |
| Either defines a policy | Native MCP enforcement runs |
| `native_mcp_policy: {}` or `{"mcp_servers": []}` | Grants no native MCP access at that level |
| `native_mcp_policy: null` | Removes that level's policy; the other level still applies |
| Malformed stored policy | Requests under enforcement fail closed |
