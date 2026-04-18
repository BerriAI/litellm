# Management Endpoints (`litellm/proxy/management_endpoints/`)

Admin CRUD endpoints: keys, teams, users, orgs, budgets, models, MCP servers, SCIM, SSO.

## MCP OAuth / OpenAPI (backend)
- When an MCP server already has `authorization_url` stored, skip OAuth discovery (`_discovery_metadata`) — the server URL for OpenAPI MCPs is the spec file, not the API base, and fetching it causes timeouts.
- `client_id` should be optional in the `/authorize` endpoint — if the server has a stored `client_id` in credentials, use that. Never require callers to re-supply it.

## MCP Credential Storage
- OAuth credentials and BYOK credentials share the `litellm_mcpusercredentials` table, distinguished by a `"type"` field in the JSON payload (`"oauth2"` vs plain string).
- When deleting OAuth credentials, check type before deleting to avoid accidentally deleting a BYOK credential for the same `(user_id, server_id)` pair.
- Always pass the raw `expires_at` timestamp to the client — never set it to `None` for expired credentials. Let the frontend compute the "Expired" display state from the timestamp.
- Use `RecordNotFoundError` (not bare `except Exception`) when catching "already deleted" in credential delete endpoints.
