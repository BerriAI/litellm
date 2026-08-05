# Experimental MCP Server Change Guidelines

Read @../../../../CLAUDE.md and @CLAUDE.md before changing this package.

This directory owns the proxy-hosted MCP server implementation. Keep changes
inside the module that owns the behavior, and only reach outside this package
when the public type contract, database schema, dashboard, or cross-proxy route
wiring must change with it.

## File Structure

Respect the current package boundaries:

```text
litellm/proxy/_experimental/mcp_server/
  AGENTS.md
  CLAUDE.md
  server.py                  # ASGI/MCP route handling, sessions, tool calls   [PR7: 7-arm only — move BYOK/OAuth pre-fetch into resolver]
  mcp_server_manager.py      # upstream server registry, clients, tool routing  [PR7: _create_mcp_client swaps resolve_mcp_auth -> resolve_credentials]
  auth/
    user_api_key_auth_mcp.py # LiteLLM admission auth and MCP request headers
    token_exchange.py        # OAuth token exchange handling                    [unchanged; V1TokenExchangeAdapter delegates here]
    litellm_auth_handler.py  # authenticated-user adapter for MCP sessions
  outbound_credentials/      # NEW — typed upstream-credential resolution (resolve_credentials + arms)
    __init__.py              # public surface: resolve_credentials, the configs, CredError
    result.py                # Ok | Error union (pure stdlib)
    types.py                 # AuthConfig union, CredError, Subject, ServerSpec
    httpx_auth.py            # NoOpAuth, StaticHeaderAuth (every mode -> one httpx.Auth)
    resolver.py              # resolve_credentials(): exhaustive per-mode match + assert_never
    seams.py                 # injected Protocols (one per cache-touching mode)
    v1_adapters.py           # v1-backed seam bodies; delegate to auth/oauth2/db owners
    adapter.py               # to_subject / to_server_spec / raise_public (v1 <-> v2 boundary)
  discoverable_endpoints.py  # MCP OAuth metadata, authorize, token, callback
  byok_oauth_endpoints.py    # BYOK OAuth UI/API flow
  oauth_utils.py             # redirect URI and proxy base URL validation
  oauth2_token_cache.py      # OAuth2 and per-user token resolution/cache        [PR7: resolve_mcp_auth removed; cache class stays, V1OAuth2CacheAdapter delegates to async_get_token]
  db.py                      # MCP server, credential, env var, submission DB access  [unchanged; V1ByokStore delegates to _get_byok_credential / get_user_credential]
  toolset_db.py              # MCP toolset DB access
  rest_endpoints.py          # proxy REST facade for listing/calling MCP tools   [PR7: 7-arm only — pass identity + inbound token down instead of mcp_auth_header]
  openapi_to_mcp_generator.py# OpenAPI spec to MCP tool generation
  sampling_handler.py        # MCP sampling to LiteLLM completion flow
  elicitation_handler.py     # MCP elicitation relay flow
  semantic_tool_filter.py    # semantic filtering of available MCP tools
  tool_search.py             # opt-in virtual tools (mcp_tool_search + mcp_tool_call) for large catalogs
  guardrail_translation/
    handler.py               # MCP guardrail result translation
  sse_transport.py           # SSE transport implementation
  mcp_context.py             # contextvars for MCP request/session metadata
  mcp_debug.py               # debug helpers
  tool_registry.py           # in-memory MCP tool registry helpers
  cost_calculator.py         # MCP tool cost calculation
  ui_session_utils.py        # dashboard session auth context helpers
  utils.py                   # shared primitives used by several modules
```

Do not add broad catch-all modules. Prefer the existing owner above, and add a
new file only for a distinct capability that would otherwise make an existing
module materially harder to understand.

## Implementation Rules

- Preserve the boundary between LiteLLM admission auth and upstream MCP auth.
  Admission belongs in `auth/user_api_key_auth_mcp.py`; upstream token exchange,
  delegated auth, per-user OAuth, BYOK, and raw header forwarding belong in the
  dedicated OAuth/header modules.
- Treat `none`, bearer/API key, OAuth, OAuth token exchange, delegated upstream
  auth, SSE, streamable HTTP, and stdio as separate flows. Do not collapse them
  behind a single generic branch unless tests prove every mode still behaves
  correctly.
- Be especially careful with `available_on_public_internet: false` combined with
  `delegate_auth_to_upstream: true`. The local `CLAUDE.md` explains the anonymous
  upstream PKCE path that must remain intentional.
- Keep database-backed fields in sync across migrations, typed models under
  `litellm/types/mcp.py` or `litellm/types/mcp_server/`, config loading, this
  package, and dashboard state when the field is user-visible.
- Use the official MCP SDK types and established LiteLLM Pydantic models where
  they exist. Avoid untyped protocol dictionaries at package boundaries.
- Keep security-sensitive logic easy to audit. Header forwarding, IP filtering,
  public internet checks, token storage, env var interpolation, and credential
  encryption need focused tests for both allowed and rejected paths.
- Avoid adding comments to new code unless they explain non-obvious security or
  protocol behavior. Prefer clear names and small functions.
- The virtual tool path (`tool_search.py`, gated by `mcp_tool_search_enabled`)
  must mirror the normal tool flow: IP filtering, server allowlist, per-key tool
  permissions, no-accessible-server rejection, per-request auth headers, server
  scope, error to `isError` conversion, and spend logging. Reuse `_list_mcp_tools`
  and `execute_mcp_tool` rather than reimplementing any of these checks.

## Tests

Mirror this package under `tests/test_litellm/proxy/_experimental/mcp_server/`.
For regressions, extend the existing mapped test file instead of creating a new
one. Use subdirectories that match the implementation path, such as
`auth/test_token_exchange.py` for `auth/token_exchange.py` and
`guardrail_translation/test_mcp_guardrail_handler.py` for
`guardrail_translation/handler.py`.

Use `tests/mcp_tests/` only when extending an existing broader MCP integration
scenario that already lives there. Route, auth, tool listing, tool execution,
OAuth, sampling, elicitation, DB, and dashboard-session changes should have
focused coverage in the mirrored `tests/test_litellm/...` path first.
