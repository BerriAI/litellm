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

### 3. Configure your MCP server to verify tokens

Point your MCP server at LiteLLM's OIDC discovery endpoint:

```
https://<your-litellm>/.well-known/openid-configuration
```

Most JWT middleware (e.g. `python-jose`, `jsonwebtoken`, AWS Lambda authorizers) supports OIDC auto-discovery.

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

- **OpenAPI-backed MCP servers** (`spec_path` set) do not support hook header injection. Calls to these servers will fail with a 500 if `MCPJWTSigner` is active with `default_on: true`. Use SSE/HTTP transport MCP servers instead.
- The keypair is **in-memory by default** — rotated on every restart unless `MCP_JWT_SIGNING_KEY` is set. MCP servers should re-fetch JWKS on verification failure (short JWKS cache TTL recommended).

## Related

- [MCP Guardrails](./mcp_guardrail) — PII masking and blocking for MCP calls
- [MCP OAuth](./mcp_oauth) — upstream OAuth2 for MCP server access
- [MCP AWS SigV4](./mcp_aws_sigv4) — AWS-signed requests to MCP servers
