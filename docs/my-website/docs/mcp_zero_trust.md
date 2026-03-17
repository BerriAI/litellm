import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MCP Zero Trust Auth (JWT Signer)

The `MCPJWTSigner` guardrail signs every outbound MCP tool call with a LiteLLM-issued RS256 JWT. MCP servers validate tokens against LiteLLM's JWKS endpoint instead of trusting each upstream IdP directly.

## Architecture

```mermaid
sequenceDiagram
    participant Client
    participant LiteLLM
    participant JWKS as LiteLLM JWKS<br/>/.well-known/jwks.json
    participant MCP as MCP Server

    Client->>LiteLLM: tool call (Bearer API key / JWT)
    Note over LiteLLM: MCPJWTSigner.async_pre_call_hook()<br/>builds RS256 JWT:<br/>sub=user_id, act=team_id,<br/>scope=mcp:tools/{name}:call

    LiteLLM->>MCP: call_tool(args)<br/>Authorization: Bearer <litellm-jwt>
    MCP->>JWKS: GET /.well-known/jwks.json
    JWKS-->>MCP: RSA public key (JWKS)
    MCP->>MCP: verify JWT signature + claims
    MCP-->>LiteLLM: tool result
    LiteLLM-->>Client: response
```

### OIDC Discovery

LiteLLM publishes standard OIDC discovery so MCP servers can find the signing key automatically:

```
GET /.well-known/openid-configuration
→ { "jwks_uri": "https://<your-litellm>/.well-known/jwks.json", ... }

GET /.well-known/jwks.json
→ { "keys": [{ "kty": "RSA", "alg": "RS256", "kid": "...", "n": "...", "e": "..." }] }
```

## Setup

### 1. Enable in `config.yaml`

```yaml title="config.yaml"
guardrails:
  - guardrail_name: "mcp-jwt-signer"
    litellm_params:
      guardrail: mcp_jwt_signer
      mode: pre_mcp_call
      default_on: true
      issuer: "https://my-litellm.example.com"  # optional — defaults to request base URL
      audience: "mcp"                            # optional — default: "mcp"
      ttl_seconds: 300                           # optional — default: 300
```

### 2. (Optional) Bring your own RSA key

If unset, LiteLLM auto-generates an RSA-2048 keypair at startup (lost on restart).

```bash
# PEM string
export MCP_JWT_SIGNING_KEY="-----BEGIN RSA PRIVATE KEY-----\n..."

# Or point to a file
export MCP_JWT_SIGNING_KEY="file:///secrets/mcp-signing-key.pem"
```

### 3. Build a verified MCP server with FastMCP

[FastMCP](https://gofastmcp.com) has a built-in `JWTVerifier` that fetches LiteLLM's JWKS automatically, handles key rotation, and enforces `iss`/`aud`/`exp` — zero boilerplate.

**Install:**
```bash
pip install fastmcp PyJWT cryptography
```

**`weather_server.py`:**
```python
from fastmcp import FastMCP, Context
from fastmcp.server.auth.providers.jwt import JWTVerifier

LITELLM_BASE_URL = "https://my-litellm.example.com"

# Point JWTVerifier at LiteLLM's JWKS endpoint.
# It auto-fetches and caches the RSA public key — no key material to manage.
auth = JWTVerifier(
    jwks_uri=f"{LITELLM_BASE_URL}/.well-known/jwks.json",
    issuer=LITELLM_BASE_URL,   # must match MCPJWTSigner `issuer:` in config.yaml
    audience="mcp",            # must match MCPJWTSigner `audience:`
    algorithm="RS256",
)

mcp = FastMCP("weather-server", auth=auth)


@mcp.tool()
async def get_weather(city: str, ctx: Context) -> str:
    """Return weather for a city. Caller identity comes from the verified JWT."""
    caller = ctx.client_id  # = JWT `sub` claim (user_id or apikey hash)
    await ctx.info(f"Request from {caller}")
    return f"Weather in {city}: sunny, 72°F"


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
```

`ctx.client_id` is populated from the JWT `sub` claim after verification — you get the caller's identity for free with no extra code.

**Wire it into LiteLLM `config.yaml`:**
```yaml title="config.yaml"
mcp_servers:
  - server_name: weather
    url: http://localhost:8000/mcp
    transport: http

guardrails:
  - guardrail_name: mcp-jwt-signer
    litellm_params:
      guardrail: mcp_jwt_signer
      mode: pre_mcp_call
      default_on: true
      issuer: "https://my-litellm.example.com"
      audience: "mcp"
```

**Run and test:**
```bash
# Terminal 1 — start the MCP server
python weather_server.py

# Terminal 2 — start LiteLLM
litellm --config config.yaml

# Terminal 3 — call through LiteLLM (JWT is injected automatically)
curl -X POST http://localhost:4000/mcp/weather/call_tool \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "get_weather", "arguments": {"city": "San Francisco"}}'
```

LiteLLM signs the JWT, sends it to the weather server, and FastMCP verifies it in one round-trip. A request without a valid token gets a `401` back from FastMCP before any tool code runs.

## JWT Claims

| Claim | Value | RFC |
|-------|-------|-----|
| `iss` | LiteLLM issuer URL | RFC 7519 |
| `aud` | configured `audience` | RFC 7519 |
| `sub` | `user_api_key_dict.user_id` | RFC 8693 |
| `act.sub` | `team_id` → `org_id` → `"litellm-proxy"` | RFC 8693 delegation |
| `email` | `user_api_key_dict.user_email` (if set) | — |
| `scope` | `mcp:tools/call mcp:tools/list mcp:tools/{name}:call` | — |
| `iat`, `exp`, `nbf` | standard timing | RFC 7519 |

## Limitations

- **OpenAPI-backed MCP servers** (`spec_path` set) do not support hook header injection. When `MCPJWTSigner` is active, calls to these servers log a warning and the JWT header is skipped. Use SSE/HTTP transport MCP servers to get full JWT injection.
- The keypair is **in-memory by default** — rotated on every restart unless `MCP_JWT_SIGNING_KEY` is set. FastMCP's `JWTVerifier` automatically re-fetches JWKS on key ID miss, so rotation is handled transparently.

## Related

- [MCP Guardrails](./mcp_guardrail) — PII masking and blocking for MCP calls
- [MCP OAuth](./mcp_oauth) — upstream OAuth2 for MCP server access
- [MCP AWS SigV4](./mcp_aws_sigv4) — AWS-signed requests to MCP servers
