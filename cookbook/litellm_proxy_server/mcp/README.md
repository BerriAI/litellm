# Publish MCP servers in the AI Hub

Set `litellm_settings.public_mcp_servers` to the concrete IDs of the servers you want listed in the public AI Hub. Pin `server_id` in each configuration entry so the publication list stays stable across deployments

```yaml
mcp_servers:
  documentation:
    server_id: documentation-mcp
    url: https://mcp.example.com/mcp
    transport: http
    available_on_public_internet: true

litellm_settings:
  public_mcp_hub_strict_whitelist: true
  public_mcp_servers:
    - documentation-mcp
```

Use `documentation-mcp`, the `server_id`, in the publication list. The configuration key `documentation`, display names, and aliases are not publication IDs. Database-created servers use the ID returned by `/v1/mcp/server`

The dashboard's **AI Hub > MCP Hub > Manage MCP Hub Visibility** dialog edits this same list. Its YAML example includes the selected server IDs. With database-backed configuration (`store_model_in_db: true`), a value declared in YAML is owned by that file: edit the file and reload, or remove that key from YAML to let the dashboard manage it in the database. File-backed deployments can save the list directly to their configuration file

To remove all explicit entries, save an empty selection in the dialog or configure:

```yaml
litellm_settings:
  public_mcp_hub_strict_whitelist: true
  public_mcp_servers: []
```

## Hub listing and network access

The **Hub listing** column in AI Hub identifies servers that appear in `/public/mcp_hub`. The dashboard derives this status from the current registry and publication settings. Setting `mcp_info.is_public` on a server does not publish it; that response field is derived metadata. `mcp_info.is_public_explicit` identifies registered servers included in the explicit publication list

Gateway cards and server details show **All Networks** when `available_on_public_internet` is enabled or the server is explicitly published in `public_mcp_servers`. They show **Internal Only** when both are false. The per-server flag defaults to `true`; explicit publication overrides a disabled flag for compatibility. Older proxies that omit the metadata needed to determine access show **Unknown**. These labels describe allowed client IPs; authentication and tool permissions still apply

The default `public_mcp_hub_strict_whitelist: true` lists only registered servers in `public_mcp_servers`. Legacy mode (`false`) additionally lists registered servers with `available_on_public_internet: true`. In legacy mode, clearing the explicit publication list leaves these automatically listed servers visible. Enable strict mode when the publication list should fully determine hub visibility
