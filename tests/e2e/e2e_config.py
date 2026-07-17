"""Generic configuration for live e2e tests against a running LiteLLM proxy.

Shared by every e2e suite under tests/e2e/. Values come from the
environment so the same tests run against localhost or a deployed proxy.
"""

import os
import uuid

PROXY_BASE_URL = os.environ.get("LITELLM_PROXY_URL", "http://localhost:4000").rstrip("/")
MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-1234")

# Control-plane (management/admin) base URL. In a split control-plane/data-plane
# deployment the LLM data plane (PROXY_BASE_URL: /chat, /embeddings, native
# passthrough) and the management API (keys, users, teams, orgs, budgets, spend,
# model info, /openapi.json) are served by *different* services. The suite drives
# both through one Transport that routes by path (see transport.SplitTransport).
# Defaults to PROXY_BASE_URL so a monolithic proxy serving everything on one URL
# behaves exactly as before.
CONTROL_PLANE_BASE_URL = os.environ.get(
    "LITELLM_CONTROL_PLANE_URL", PROXY_BASE_URL
).rstrip("/")

UI_USERNAME = os.environ.get("E2E_UI_USERNAME", "admin")
UI_PASSWORD = os.environ.get("E2E_UI_PASSWORD", MASTER_KEY)

CHEAP_ANTHROPIC_MODEL = os.environ.get("E2E_CHEAP_ANTHROPIC_MODEL", "claude-haiku-4-5")
CHEAP_OPENAI_MODEL = os.environ.get("E2E_CHEAP_OPENAI_MODEL", "gpt-5.5")

# Jaeger query API of the compose stack's OTEL trace destination (the `jaeger`
# service in docker-compose.yml maps it to host 16686). Trace-completeness tests
# read exported spans back through it.
OTEL_QUERY_URL = os.environ.get("E2E_OTEL_QUERY_URL", "http://localhost:16686").rstrip("/")

# Query URL of the compose stack's DataDog logs-intake sink (the `dd-sink`
# service records every intake POST and replays them on GET /requests).
DD_SINK_URL = os.environ.get("E2E_DD_SINK_URL", "http://localhost:9915").rstrip("/")

# The MCP upstream the mcp suite registers on the proxy: the mcp-stub compose
# service, addressed by service name on the compose network. It must be
# reachable from the proxy, not from pytest; override when the proxy under
# test runs somewhere the compose stub is not visible from.
MCP_STUB_URL = os.environ.get("E2E_MCP_STUB_URL", "http://mcp-stub:8765/mcp")

# The stub's sibling mounts (see mcp/stub/stub_server.py): a second anonymous
# upstream with a disjoint tool set, an X-API-Key-guarded upstream, a
# Bearer-guarded upstream, and the OAuth2 client_credentials token endpoint
# that mints the only token the Bearer guard accepts. Derived from MCP_STUB_URL
# so one override relocates the whole stub.
_MCP_STUB_BASE = MCP_STUB_URL.removesuffix("/mcp")
MCP_STUB_SECOND_URL = f"{_MCP_STUB_BASE}/second/mcp"
MCP_STUB_APIKEY_URL = f"{_MCP_STUB_BASE}/apikey/mcp"
MCP_STUB_OAUTH_URL = f"{_MCP_STUB_BASE}/oauth/mcp"
MCP_STUB_OAUTHUSER_URL = f"{_MCP_STUB_BASE}/oauthuser/mcp"
MCP_STUB_TOKEN_URL = f"{_MCP_STUB_BASE}/oauth/token"

# The stub's authorization endpoint is dereferenced by the browser (pytest
# playing that role from the host), never by the proxy, so unlike the URLs
# above it needs the host-visible address of the stub: the compose file
# publishes mcp-stub on host port 8765 for exactly this hop.
MCP_STUB_AUTHORIZE_BROWSER_URL = os.environ.get(
    "E2E_MCP_STUB_AUTHORIZE_BROWSER_URL", "http://localhost:8765/oauth/authorize"
)

# Deterministic test-only credentials; must mirror mcp/stub/stub_server.py.
MCP_STUB_UPSTREAM_API_KEY = "e2e-stub-upstream-api-key"
MCP_STUB_OAUTH_CLIENT_ID = "e2e-stub-oauth-client-id"
MCP_STUB_OAUTH_CLIENT_SECRET = "e2e-stub-oauth-client-secret"
MCP_STUB_OAUTH_SCOPE = "tools:read"
MCP_STUB_OAUTH_ACCESS_TOKEN = "e2e-stub-minted-access-token"
MCP_STUB_OAUTH_USER_CLIENT_ID = "e2e-stub-user-client-id"
MCP_STUB_OAUTH_USER_CLIENT_SECRET = "e2e-stub-user-client-secret"
MCP_STUB_OAUTH_USER_ACCESS_TOKEN = "e2e-stub-user-access-token"

# Writes on the proxy are eventually consistent (e.g. spend rows flush on
# proxy_batch_write_at, ~60s). Read-backs poll to this deadline, never sleep-once.
POLL_TIMEOUT = float(os.environ.get("E2E_POLL_TIMEOUT", "120"))
POLL_INTERVAL = float(os.environ.get("E2E_POLL_INTERVAL", "5"))
REQUEST_TIMEOUT = float(os.environ.get("E2E_REQUEST_TIMEOUT", "60"))


def unique_marker() -> str:
    """A short unique token per call/run, so concurrent runs and the shared
    response cache never collide on prompts, tags, or customer ids."""
    return uuid.uuid4().hex[:12]
