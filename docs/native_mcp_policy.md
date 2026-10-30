# Native MCP allowlists

`native_mcp_policy` restricts provider-hosted MCP attachments on Chat Completions, Responses, and Anthropic Messages requests through the LiteLLM proxy. The same checks apply when a supported request is routed to an Anthropic model on Bedrock or Vertex. Servers do not need registration in the LiteLLM MCP gateway.

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

Team and key grants are unioned for the same server name and URL. A key can add tools but cannot revoke a team grant. A whole-server grant at either level permits every tool on that server. Grants for different URLs are never combined.

| Configuration | Behavior |
|---|---|
| Neither team nor key defines a policy | Existing behavior is preserved |
| Either defines a policy | Native MCP enforcement runs |
| `native_mcp_policy: {}` or `{"mcp_servers": []}` | Grants no native MCP access at that level |
| `native_mcp_policy: null` | Removes that level's policy; the other level still applies |
| Malformed stored policy | Requests under enforcement fail closed |

To preserve unrelated metadata, update the top-level `native_mcp_policy` field. Direct metadata writes are also validated and permission checked. Metadata replacement follows the existing management API behavior. Replacing metadata with a blob that omits its policy removes that policy and therefore requires proxy-admin permission. Key holders and team admins cannot broaden or remove grants through either write path.

## Send a bounded attachment

An OpenAI Responses attachment for the first example can offer just `search_code`:

```json
{
  "type": "mcp",
  "server_label": "github",
  "server_url": "https://mcp.example.com/github",
  "allowed_tools": ["search_code"]
}
```

Anthropic legacy connectors use `mcp_servers[].tool_configuration.allowed_tools`. The newer `mcp_toolset` shape must set `default_config.enabled` to `false` and explicitly enable permitted tools in `configs`. A default-on toolset requires a whole-server grant. Mixed legacy and new configuration for one server, duplicate server labels, and unreadable scopes are rejected.

An omitted or empty attachment allowlist is treated conservatively as unbounded and requires a whole-server grant. A disabled Anthropic connector or a default-off toolset with no enabled tools offers no tools, but still needs an approved server identity.

The proxy checks every offered native MCP server and tool. Any denial rejects the entire request with HTTP 400 and `tool_access_denied`, naming the denied server or tools. It does not strip tools or rewrite the request's tool restrictions. A caller cannot substitute another URL under an approved label.

## Response enforcement

Native MCP call records in non-streaming responses are checked against the same policy before returning the response. Response server labels are bound to the URLs from the validated request. A call with an unknown server or unreadable tool name is rejected.

Streaming response enforcement is added in the dependent change.

Response enforcement cannot undo a tool the provider already executed. Request-side restrictions are what constrain the tools the provider is authorized to execute.

## Scope

This policy covers `/v1/chat/completions`, `/v1/responses`, and `/v1/messages`, including their existing aliases. It does not add enforcement to direct provider passthrough routes such as `/bedrock/...`, stored-response retrieval, or SDK-only calls.

Ordinary function tools, provider built-ins, and local tools named `mcp__server__tool` keep their existing controls. The existing `metadata.allowed_tools` list remains independent and still applies when configured. Requests containing only `litellm_proxy` MCP attachments keep the existing MCP gateway permissions and response handling. When a native policy is active, mixing gateway and provider-native attachments in one request returns HTTP 400. Use separate requests for the two execution paths. Registries, reusable named policies, and a management UI are outside this version.
