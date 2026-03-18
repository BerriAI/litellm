import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# MCP Zero Trust Auth (JWT Signer)

![Zero Trust MCP Gateway](/img/mcp_zero_trust_gateway.png)

The `MCPJWTSigner` guardrail signs every outbound MCP tool call with a LiteLLM-issued RS256 JWT. MCP servers validate tokens against LiteLLM's JWKS endpoint instead of trusting each upstream IdP directly.

```mermaid
sequenceDiagram
    participant Client
    participant LiteLLM
    participant JWKS as LiteLLM JWKS<br/>/.well-known/jwks.json
    participant MCP as MCP Server

    Client->>LiteLLM: tool call (Bearer API key / JWT)
    Note over LiteLLM: MCPJWTSigner signs RS256 JWT:<br/>sub=user, act=team, scope=mcp:tools/{name}:call

    LiteLLM->>MCP: call_tool(args)<br/>Authorization: Bearer <litellm-jwt>
    MCP->>JWKS: GET /.well-known/jwks.json
    JWKS-->>MCP: RSA public key
    MCP->>MCP: verify JWT signature + claims
    MCP-->>LiteLLM: tool result
    LiteLLM-->>Client: response
```

LiteLLM publishes OIDC discovery so MCP servers find the signing key automatically:

```
GET /.well-known/openid-configuration  →  { "jwks_uri": "https://<litellm>/.well-known/jwks.json" }
GET /.well-known/jwks.json             →  { "keys": [{ "kty": "RSA", "alg": "RS256", ... }] }
```

---

## Basic setup

Enable the guardrail and point your MCP server at LiteLLM's JWKS endpoint. Every tool call gets a signed JWT automatically — no code changes needed on the client side.

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
      issuer: "https://my-litellm.example.com"  # defaults to request base URL
      audience: "mcp"                            # default: "mcp"
      ttl_seconds: 300                           # default: 300
```

**Bring your own signing key** (recommended for production — auto-generated keys are lost on restart):

```bash
export MCP_JWT_SIGNING_KEY="-----BEGIN RSA PRIVATE KEY-----\n..."
# or point to a file
export MCP_JWT_SIGNING_KEY="file:///secrets/mcp-signing-key.pem"
```

**Build a verified MCP server with [FastMCP](https://gofastmcp.com):**

```python title="weather_server.py"
from fastmcp import FastMCP, Context
from fastmcp.server.auth.providers.jwt import JWTVerifier

auth = JWTVerifier(
    jwks_uri="https://my-litellm.example.com/.well-known/jwks.json",
    issuer="https://my-litellm.example.com",
    audience="mcp",
    algorithm="RS256",
)

mcp = FastMCP("weather-server", auth=auth)

@mcp.tool()
async def get_weather(city: str, ctx: Context) -> str:
    caller = ctx.client_id  # JWT `sub` — the verified user identity
    return f"Weather in {city}: sunny, 72°F (requested by {caller})"

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
```

FastMCP fetches the JWKS automatically and re-fetches on key rotation.

---

## Your users log in with Okta / Azure AD — and you want that identity in the MCP JWT

By default, `sub` in the outbound JWT is LiteLLM's internal `user_id`. If your users authenticate with a corporate IdP, the MCP server sees a LiteLLM-internal ID instead of their real identity (email, employee ID, etc.).

**With verify+re-sign**, LiteLLM validates the incoming IdP token, extracts real identity claims, and forwards them into the outbound JWT. The MCP server sees the user's actual identity without ever needing to trust the original IdP.

```yaml title="config.yaml"
guardrails:
  - guardrail_name: mcp-jwt-signer
    litellm_params:
      guardrail: mcp_jwt_signer
      mode: pre_mcp_call
      default_on: true
      issuer: "https://my-litellm.example.com"

      # Verify the incoming Bearer token against the IdP before re-signing
      access_token_discovery_uri: "https://login.microsoftonline.com/{tenant}/v2.0/.well-known/openid-configuration"
      verify_issuer: "https://login.microsoftonline.com/{tenant}/v2.0"
      verify_audience: "api://my-app"

      # Resolution order for the outbound JWT `sub` claim:
      # try the incoming token's `sub`, then fall back to LiteLLM's user_id
      end_user_claim_sources:
        - "token:sub"       # sub from the verified incoming JWT
        - "token:email"     # email from the incoming JWT
        - "litellm:user_id" # LiteLLM's internal user_id as last resort
```

If the incoming token is **opaque** (not a JWT), add an introspection endpoint:

```yaml
      token_introspection_endpoint: "https://idp.example.com/oauth2/introspect"
```

LiteLLM will POST the token to the introspection endpoint (RFC 7662) and use the returned claims.

**Supported `end_user_claim_sources` values:**

| Source | Resolves to |
|--------|-------------|
| `token:<claim>` | Any claim from the verified incoming JWT (e.g. `token:sub`, `token:email`, `token:oid`) |
| `litellm:user_id` | `UserAPIKeyAuth.user_id` |
| `litellm:email` | `UserAPIKeyAuth.user_email` |
| `litellm:end_user_id` | `UserAPIKeyAuth.end_user_id` |
| `litellm:team_id` | `UserAPIKeyAuth.team_id` |

The first source that resolves to a non-empty value wins.

---

## Your MCP server needs to enforce that callers have specific roles or attributes

Some MCP servers contain sensitive operations — you want to block the call at the LiteLLM layer if the user's IdP token doesn't carry the claims your server expects (e.g. `department`, `employee_type`, `roles`).

Use `required_claims` to reject the call with a `403` before the tool ever runs. Use `optional_claims` to forward claims that are useful but not mandatory.

```yaml title="config.yaml"
guardrails:
  - guardrail_name: mcp-jwt-signer
    litellm_params:
      guardrail: mcp_jwt_signer
      mode: pre_mcp_call
      default_on: true

      access_token_discovery_uri: "https://idp.example.com/.well-known/openid-configuration"

      # Block calls where the incoming token is missing these claims
      required_claims:
        - "sub"
        - "employee_id"   # service accounts without an employee_id are blocked

      # Forward these claims into the outbound JWT when present
      # (silently skipped if the incoming token doesn't have them)
      optional_claims:
        - "groups"
        - "department"
```

With this config, a service account JWT that lacks `employee_id` gets a `403` with a clear error — the MCP server never receives the request.

---

## You need to add metadata to every JWT

Sometimes the MCP server needs context that LiteLLM doesn't carry natively — which deployment sent the request, a tenant ID, an environment tag. Use claim operations to inject, override, or strip claims from the outbound JWT.

```yaml title="config.yaml"
guardrails:
  - guardrail_name: mcp-jwt-signer
    litellm_params:
      guardrail: mcp_jwt_signer
      mode: pre_mcp_call
      default_on: true

      # add: insert only when the key is not already in the JWT
      add_claims:
        deployment_id: "prod-us-east-1"
        tenant_id: "acme-corp"

      # set: always override, even if the claim was computed from the incoming token
      set_claims:
        env: "production"

      # remove: strip claims you don't want the MCP server to see
      remove_claims:
        - "nbf"   # some validators reject nbf; remove it if yours does
```

Operations are applied in order: `add_claims` → `set_claims` → `remove_claims`. `set_claims` always wins over `add_claims`; `remove_claims` beats both.

---

## You're using AWS Bedrock AgentCore Gateway (or a two-layer auth architecture)

Some MCP gateways split auth into two layers — one JWT for the transport channel (authenticates the connection) and a separate JWT for the MCP resource layer (authorizes tool calls). A single JWT can't serve both because they need different `aud` values and TTLs.

Use the two-token model: LiteLLM issues both JWTs in one hook and injects them into separate headers.

```yaml title="config.yaml"
guardrails:
  - guardrail_name: mcp-jwt-signer
    litellm_params:
      guardrail: mcp_jwt_signer
      mode: pre_mcp_call
      default_on: true
      issuer: "https://my-litellm.example.com"
      audience: "mcp-resource"      # for the MCP resource layer
      ttl_seconds: 300

      # Second JWT — same sub/act/scope but targeted at the transport layer
      channel_token_audience: "bedrock-agentcore-gateway"
      channel_token_ttl: 60         # shorter TTL — transport tokens should expire fast
```

LiteLLM injects:
- `Authorization: Bearer <resource-token>` (audience: `mcp-resource`, TTL: 300s)
- `x-mcp-channel-token: Bearer <channel-token>` (audience: `bedrock-agentcore-gateway`, TTL: 60s)

The gateway reads `x-mcp-channel-token` to authenticate the transport connection. The MCP server reads `Authorization` to authorize tool calls. Both tokens are signed with the same LiteLLM key so your MCP server only needs to trust one JWKS endpoint.

---

## You want to control exactly which scopes go into the JWT

By default, LiteLLM auto-generates least-privilege scopes per request:
- Tool call: `mcp:tools/call mcp:tools/{name}:call`
- List tools: `mcp:tools/call mcp:tools/list`

If your MCP server does its own scope enforcement and needs a specific format, or you want to grant a fixed set of operations regardless of which tool is being called, set `allowed_scopes` explicitly:

```yaml title="config.yaml"
guardrails:
  - guardrail_name: mcp-jwt-signer
    litellm_params:
      guardrail: mcp_jwt_signer
      mode: pre_mcp_call
      default_on: true

      # Fixed scope list — replaces auto-generation entirely
      allowed_scopes:
        - "mcp:tools/call"
        - "mcp:tools/list"
        - "mcp:admin"
```

When `allowed_scopes` is set, all JWTs (regardless of which tool is called) carry exactly those scopes.

---

## Debugging JWT rejections

If your MCP server is rejecting tokens and you're not sure why, enable `debug_headers`. LiteLLM will add an `x-litellm-mcp-debug` header to each response containing the key claims that were signed:

```yaml title="config.yaml"
guardrails:
  - guardrail_name: mcp-jwt-signer
    litellm_params:
      guardrail: mcp_jwt_signer
      mode: pre_mcp_call
      default_on: true
      debug_headers: true
```

The response will include:

```
x-litellm-mcp-debug: v=1; kid=a3f1b2c4d5e6f708; sub=alice@corp.com; iss=https://my-litellm.example.com; exp=1712345678; scope=mcp:tools/call mcp:tools/get_weather:call
```

Use this to confirm the correct `sub`, `iss`, `kid`, and `scope` are being signed. Disable in production — it leaks claim metadata into response headers.

---

## JWT Claims reference

| Claim | Value |
|-------|-------|
| `iss` | `issuer` config value (or request base URL) |
| `aud` | `audience` config value (default: `"mcp"`) |
| `sub` | Resolved via `end_user_claim_sources` (default: `user_id` → api-key hash → `"litellm-proxy"`) |
| `act.sub` | `team_id` → `org_id` → `"litellm-proxy"` (RFC 8693 delegation) |
| `email` | `user_email` from LiteLLM auth context (when available) |
| `scope` | Auto-generated per tool call, or `allowed_scopes` when configured |
| `iat`, `exp`, `nbf` | Standard timing claims (RFC 7519) |

---

## Limitations

- **OpenAPI-backed MCP servers** (`spec_path` set) do not support JWT injection. LiteLLM logs a warning and skips the header. Use SSE/HTTP transport servers to get full JWT injection.
- The keypair is **in-memory by default** and rotated on each restart unless `MCP_JWT_SIGNING_KEY` is set. FastMCP's `JWTVerifier` handles key rotation transparently via JWKS key ID matching.

---

## Related

- [MCP Guardrails](./mcp_guardrail) — PII masking and blocking for MCP calls
- [MCP OAuth](./mcp_oauth) — upstream OAuth2 for MCP server access
- [MCP AWS SigV4](./mcp_aws_sigv4) — AWS-signed requests to MCP servers
