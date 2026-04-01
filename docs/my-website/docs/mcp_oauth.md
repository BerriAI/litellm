# MCP OAuth

LiteLLM supports two OAuth 2.0 flows for MCP servers:

| Flow | Use Case | How It Works |
|------|----------|--------------|
| **Interactive (PKCE)** | User-facing apps (Claude Code, Cursor) | Browser-based consent, per-user tokens |
| **Machine-to-Machine (M2M)** | Backend services, CI/CD, automated agents | `client_credentials` grant, proxy-managed tokens |

## Interactive OAuth (PKCE)

For user-facing MCP clients (Claude Code, Cursor), LiteLLM supports the full OAuth 2.0 authorization code flow with PKCE.

### Setup

```yaml title="config.yaml" showLineNumbers
mcp_servers:
  github_mcp:
    url: "https://api.githubcopilot.com/mcp"
    auth_type: oauth2
    client_id: os.environ/GITHUB_OAUTH_CLIENT_ID
    client_secret: os.environ/GITHUB_OAUTH_CLIENT_SECRET
```

[**See Claude Code Tutorial**](./tutorials/claude_responses_api#connecting-mcp-servers)

### How It Works

```mermaid
sequenceDiagram
    participant Browser as User-Agent (Browser)
    participant Client as Client
    participant LiteLLM as LiteLLM Proxy
    participant MCP as MCP Server (Resource Server)
    participant Auth as Authorization Server

    Note over Client,LiteLLM: Step 1 – Resource discovery
    Client->>LiteLLM: GET /.well-known/oauth-protected-resource/{mcp_server_name}/mcp
    LiteLLM->>Client: Return resource metadata

    Note over Client,LiteLLM: Step 2 – Authorization server discovery
    Client->>LiteLLM: GET /.well-known/oauth-authorization-server/{mcp_server_name}
    LiteLLM->>Client: Return authorization server metadata

    Note over Client,Auth: Step 3 – Dynamic client registration
    Client->>LiteLLM: POST /{mcp_server_name}/register
    LiteLLM->>Auth: Forward registration request
    Auth->>LiteLLM: Issue client credentials
    LiteLLM->>Client: Return client credentials

    Note over Client,Browser: Step 4 – User authorization (PKCE)
    Client->>Browser: Open authorization URL + code_challenge + resource
    Browser->>Auth: Authorization request
    Note over Auth: User authorizes
    Auth->>Browser: Redirect with authorization code
    Browser->>LiteLLM: Callback to LiteLLM with code
    LiteLLM->>Browser: Redirect back with authorization code
    Browser->>Client: Callback with authorization code

    Note over Client,Auth: Step 5 – Token exchange
    Client->>LiteLLM: Token request + code_verifier + resource
    LiteLLM->>Auth: Forward token request
    Auth->>LiteLLM: Access (and refresh) token
    LiteLLM->>Client: Return tokens

    Note over Client,MCP: Step 6 – Authenticated MCP call
    Client->>LiteLLM: MCP request with access token + LiteLLM API key
    LiteLLM->>MCP: MCP request with Bearer token
    MCP-->>LiteLLM: MCP response
    LiteLLM-->>Client: Return MCP response
```

**Participants**

- **Client** -- The MCP-capable AI agent (e.g., Claude Code, Cursor, or another IDE/agent) that initiates OAuth discovery, authorization, and tool invocations on behalf of the user.
- **LiteLLM Proxy** -- Mediates all OAuth discovery, registration, token exchange, and MCP traffic while protecting stored credentials.
- **Authorization Server** -- Issues OAuth 2.0 tokens via dynamic client registration, PKCE authorization, and token endpoints.
- **MCP Server (Resource Server)** -- The protected MCP endpoint that receives LiteLLM's authenticated JSON-RPC requests.
- **User-Agent (Browser)** -- Temporarily involved so the end user can grant consent during the authorization step.

**Flow Steps**

1. **Resource Discovery**: The client fetches MCP resource metadata from LiteLLM's `.well-known/oauth-protected-resource` endpoint to understand scopes and capabilities.
2. **Authorization Server Discovery**: The client retrieves the OAuth server metadata (token endpoint, authorization endpoint, supported PKCE methods) through LiteLLM's `.well-known/oauth-authorization-server` endpoint.
3. **Dynamic Client Registration**: The client registers through LiteLLM, which forwards the request to the authorization server (RFC 7591). If the provider doesn't support dynamic registration, you can pre-store `client_id`/`client_secret` in LiteLLM (e.g., GitHub MCP) and the flow proceeds the same way.
4. **User Authorization**: The client launches a browser session (with code challenge and resource hints). The user approves access, the authorization server sends the code through LiteLLM back to the client.
5. **Token Exchange**: The client calls LiteLLM with the authorization code, code verifier, and resource. LiteLLM exchanges them with the authorization server and returns the issued access/refresh tokens.
6. **MCP Invocation**: With a valid token, the client sends the MCP JSON-RPC request (plus LiteLLM API key) to LiteLLM, which forwards it to the MCP server and relays the tool response.

See the official [MCP Authorization Flow](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization#authorization-flow-steps) for additional reference.

## Machine-to-Machine (M2M) Auth

LiteLLM automatically fetches, caches, and refreshes OAuth2 tokens using the `client_credentials` grant. No manual token management required.

### Setup

You can configure M2M OAuth via the LiteLLM UI or `config.yaml`.

### UI Setup

Navigate to the **MCP Servers** page and click **+ Add New MCP Server**.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/d1f1e89c-a789-4975-8846-b15d9821984a/ascreenshot_630800e00a2e4b598baabfc25efbabd3_text_export.jpeg)

Enter a name for your server and select **HTTP** as the transport type.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/2008c9d6-6093-4121-beab-1e52c71376aa/ascreenshot_516ffd6c7b524465a253a56048c3d228_text_export.jpeg)

Paste the MCP server URL.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/b0ee8b7d-6de8-492b-8962-287987feec29/ascreenshot_b3efca82078a4c6bb1453c58161909f9_text_export.jpeg)

Under **Authentication**, select **OAuth**.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/e1597814-ff8e-40b9-9d7b-864dcdbe0910/ascreenshot_2097612712264d8f9e553f7ca9175fb0_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/f6ea5694-f28a-4bc3-9c9a-bb79f199bd65/ascreenshot_9be839f55b1b4f96bfe24030ba2c7f8d_text_export.jpeg)

Choose **Machine-to-Machine (M2M)** as the OAuth flow type. This is for server-to-server authentication using the `client_credentials` grant — no browser interaction required.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/9853310c-1d86-4628-bad1-7a391eca0e4d/ascreenshot_f302a286fa264fdd8d56db53b8f9395c_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/df64dc65-ef86-475d-adaf-12e227d5e873/ascreenshot_9e2f41d43a76435f918a00b52ffcc639_text_export.jpeg)

Fill in the **Client ID** and **Client Secret** provided by your OAuth provider.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/0de5a7bd-9898-4fc7-8843-b23dd5aac47f/ascreenshot_b9087aaa81a14b5b9c199929efc4a563_text_export.jpeg)

Enter the **Token URL** — this is the endpoint LiteLLM will call to fetch access tokens using `client_credentials`.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/0aea70f1-558c-4dca-91bc-1175fe1ddc89/ascreenshot_b3fcf8a1287e4e2d9a3d67c4a29f7bff_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/e842ef09-1fd7-47a6-909b-252d389f0abc/ascreenshot_2a87dad3624847e7ac370591d1d1aedd_text_export.jpeg)

Scroll down and review the server URL and all fields, then click **Create MCP Server**.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/0857712b-4b53-40f8-8c1f-a4c72edaa644/ascreenshot_47be3fcd5de64ed391f70c1fb74a8bfc_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/9d961765-955f-4905-a3dc-1a446aa3b2cc/ascreenshot_43fd39d014224564bc6b35aced1fb6d3_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/3825d5fa-8fd1-4e71-b090-77ff0259c3f6/ascreenshot_2509a7ebd9bf421eb0e82f2553566745_text_export.jpeg)

Once created, open the server and navigate to the **MCP Tools** tab to verify that LiteLLM can connect and list available tools.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/8107e27b-5072-4675-8fd6-89b47692b1bd/ascreenshot_f774bc76138f430d808fb4482ebfcdca_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/ce94bb7b-c81b-4396-9939-178efb2cdfce/ascreenshot_28b838ab6ae34c76858454555c4c1d79_text_export.jpeg)

Select a tool (e.g. **echo**) to test it. Fill in the required parameters and click **Call Tool**.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/c459c1d3-ec29-4211-9c28-37fbe7783bbc/ascreenshot_e9b138b3c2cc4440bb1a6f42ac7ae861_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/5438ac60-e0ac-4a79-bf6f-5594f160d3b5/ascreenshot_9133a17d26204c46bce497e74685c483_text_export.jpeg)

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/a8f6821b-3982-4b4d-9b25-70c8aff5ac31/ascreenshot_28d474d0e62545a482cff6128527883a_text_export.jpeg)

LiteLLM automatically fetches an OAuth token behind the scenes and calls the tool. The result confirms the M2M OAuth flow is working end-to-end.

![](https://colony-recorder.s3.amazonaws.com/files/2026-02-10/c3924549-a949-48d1-ac67-ab4c30475859/ascreenshot_8f6eca9d717f45478d50a881bd244bb3_text_export.jpeg)

### Config.yaml Setup

```yaml title="config.yaml" showLineNumbers
mcp_servers:
  my_mcp_server:
    url: "https://my-mcp-server.com/mcp"
    auth_type: oauth2
    client_id: os.environ/MCP_CLIENT_ID
    client_secret: os.environ/MCP_CLIENT_SECRET
    token_url: "https://auth.example.com/oauth/token"
    scopes: ["mcp:read", "mcp:write"]  # optional
```

### How It Works

1. On first MCP request, LiteLLM POSTs to `token_url` with `grant_type=client_credentials`
2. The access token is cached in-memory with TTL = `expires_in - 60s`
3. Subsequent requests reuse the cached token
4. When the token expires, LiteLLM fetches a new one automatically

```mermaid
sequenceDiagram
    participant Client as Client
    participant LiteLLM as LiteLLM Proxy
    participant Auth as Authorization Server
    participant MCP as MCP Server

    Client->>LiteLLM: MCP request + LiteLLM API key
    LiteLLM->>Auth: POST /oauth/token (client_credentials)
    Auth->>LiteLLM: access_token (expires_in: 3600)
    LiteLLM->>MCP: MCP request + Bearer token
    MCP-->>LiteLLM: MCP response
    LiteLLM-->>Client: MCP response

    Note over LiteLLM: Token cached for subsequent requests
    Client->>LiteLLM: Next MCP request
    LiteLLM->>MCP: MCP request + cached Bearer token
    MCP-->>LiteLLM: MCP response
    LiteLLM-->>Client: MCP response
```

### Test with Mock Server

Use [BerriAI/mock-oauth2-mcp-server](https://github.com/BerriAI/mock-oauth2-mcp-server) to test locally:

```bash title="Terminal 1 - Start mock server" showLineNumbers
pip install fastapi uvicorn
python mock_oauth2_mcp_server.py  # starts on :8765
```

```yaml title="config.yaml" showLineNumbers
mcp_servers:
  test_oauth2:
    url: "http://localhost:8765/mcp"
    auth_type: oauth2
    client_id: "test-client"
    client_secret: "test-secret"
    token_url: "http://localhost:8765/oauth/token"
```

```bash title="Terminal 2 - Start proxy and test" showLineNumbers
litellm --config config.yaml --port 4000

# List tools
curl http://localhost:4000/mcp-rest/tools/list \
  -H "Authorization: Bearer sk-1234"

# Call a tool
curl http://localhost:4000/mcp-rest/tools/call \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{"name": "echo", "arguments": {"message": "hello"}}'
```

### Config Reference

| Field | Required | Description |
|-------|----------|-------------|
| `auth_type` | Yes | Must be `oauth2` |
| `client_id` | Yes | OAuth2 client ID. Supports `os.environ/VAR_NAME` |
| `client_secret` | Yes | OAuth2 client secret. Supports `os.environ/VAR_NAME` |
| `token_url` | Yes | Token endpoint URL |
| `scopes` | No | List of scopes to request |

## Debugging OAuth

When the LiteLLM proxy is hosted remotely and you cannot access server logs, enable **debug headers** to get masked authentication diagnostics in the HTTP response.

### Enable Debug Mode

Add the `x-litellm-mcp-debug: true` header to your MCP client request.

**Claude Code:**

```bash
claude mcp add --transport http litellm_proxy http://proxy.example.com/atlassian_mcp/mcp \
  --header "x-litellm-api-key: Bearer sk-..." \
  --header "x-litellm-mcp-debug: true"
```

**curl:**

```bash
curl -X POST http://localhost:4000/atlassian_mcp/mcp \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-..." \
  -H "x-litellm-mcp-debug: true" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

### Reading the Debug Response Headers

The response includes these headers (all sensitive values are masked):

| Header | Description |
|--------|-------------|
| `x-mcp-debug-inbound-auth` | Which inbound auth headers were present. |
| `x-mcp-debug-oauth2-token` | The OAuth2 token (masked). Shows `SAME_AS_LITELLM_KEY` if the LiteLLM key is leaking. |
| `x-mcp-debug-auth-resolution` | Which auth method was used: `oauth2-passthrough`, `m2m-client-credentials`, `per-request-header`, `static-token`, or `no-auth`. |
| `x-mcp-debug-outbound-url` | The upstream MCP server URL. |
| `x-mcp-debug-server-auth-type` | The `auth_type` configured on the server. |

**Example — healthy OAuth2 passthrough:**

```
x-mcp-debug-inbound-auth: x-litellm-api-key=Bearer****1234; authorization=Bearer****ef01
x-mcp-debug-oauth2-token: Bearer****ef01
x-mcp-debug-auth-resolution: oauth2-passthrough
x-mcp-debug-outbound-url: https://mcp.atlassian.com/v1/mcp
x-mcp-debug-server-auth-type: oauth2
```

**Example — LiteLLM key leaking (misconfigured):**

```
x-mcp-debug-inbound-auth: authorization=Bearer****1234
x-mcp-debug-oauth2-token: Bearer****1234 (SAME_AS_LITELLM_KEY - likely misconfigured)
x-mcp-debug-auth-resolution: oauth2-passthrough
x-mcp-debug-outbound-url: https://mcp.atlassian.com/v1/mcp
x-mcp-debug-server-auth-type: oauth2
```

### Common Issues

#### LiteLLM API key leaking to the MCP server

**Symptom:** `x-mcp-debug-oauth2-token` shows `SAME_AS_LITELLM_KEY`.

The `Authorization` header carries the LiteLLM API key instead of an OAuth2 token. The OAuth2 flow never ran because the client already had an `Authorization` header set.

**Fix:** Move the LiteLLM key to `x-litellm-api-key`:

```bash
# WRONG — blocks OAuth2 discovery
claude mcp add --transport http my_server http://proxy/mcp/server \
    --header "Authorization: Bearer sk-..."

# CORRECT — LiteLLM key in dedicated header, Authorization free for OAuth2
claude mcp add --transport http my_server http://proxy/mcp/server \
    --header "x-litellm-api-key: Bearer sk-..."
```

#### No OAuth2 token present

**Symptom:** `x-mcp-debug-oauth2-token` shows `(none)` and `x-mcp-debug-auth-resolution` shows `no-auth`.

Check that:
1. The `Authorization` header is NOT set as a static header in the client config.
2. The MCP server in LiteLLM config has `auth_type: oauth2`.
3. The `.well-known/oauth-protected-resource` endpoint returns valid metadata.

#### M2M token used instead of user token

**Symptom:** `x-mcp-debug-auth-resolution` shows `m2m-client-credentials`.

The server has `client_id`/`client_secret`/`token_url` configured so LiteLLM is fetching a machine-to-machine token instead of using the per-user OAuth2 token. To use per-user tokens, remove the client credentials from the server config.
