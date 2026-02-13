import Image from '@theme/IdealImage';

# MCP Troubleshooting Guide

When LiteLLM acts as an MCP proxy, traffic normally flows `Client → LiteLLM Proxy → MCP Server`, while OAuth-enabled setups add an authorization server for metadata discovery.

For provisioning steps, transport options, and configuration fields, refer to [mcp.md](./mcp.md).

## Locate the Error Source

Pin down where the failure occurs before adjusting settings so you do not mix symptoms from separate hops.

### LiteLLM UI / Playground Errors (LiteLLM → MCP)
Failures shown on the MCP creation form or within the MCP Tool Testing Playground mean the LiteLLM proxy cannot reach the MCP server. Typical causes are misconfiguration (transport, headers, credentials), MCP/server outages, network/firewall blocks, or inaccessible OAuth metadata.

<Image 
  img={require('../img/mcp_tool_testing_playground.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

<br/>

**Actions**
- Capture LiteLLM proxy logs alongside MCP-server logs (see [Error Log Example](./mcp_troubleshoot#error-log-example-failed-mcp-call)) to inspect the request/response pair and stack traces.
- From the LiteLLM server, run Method 2 ([`curl` smoke test](./mcp_troubleshoot#curl-smoke-test)) against the MCP endpoint to confirm basic connectivity.

### Client Traffic Issues (Client → LiteLLM)
If only real client requests fail, determine whether LiteLLM ever reaches the MCP hop.

#### MCP Protocol Sessions
Clients such as IDEs or agent runtimes speak the MCP protocol directly with LiteLLM.

**Actions**
- Inspect LiteLLM access logs (see [Access Log Example](./mcp_troubleshoot#access-log-example-successful-mcp-call)) to verify the client request reached the proxy and which MCP server it targeted.
- Review LiteLLM error logs (see [Error Log Example](./mcp_troubleshoot#error-log-example-failed-mcp-call)) for TLS, authentication, or routing errors that block the request before the MCP call starts.
- Use the [MCP Inspector](./mcp_troubleshoot#mcp-inspector) to confirm the MCP server is reachable outside of the failing client.

#### Responses/Completions with Embedded MCP Calls
During `/responses` or `/chat/completions`, LiteLLM may trigger MCP tool calls mid-request. An error could occur before the MCP call begins or after the MCP responds.

**Actions**
- Check LiteLLM request logs (see [Access Log Example](./mcp_troubleshoot#access-log-example-successful-mcp-call)) to see whether an MCP attempt was recorded; if not, the problem lies in `Client → LiteLLM`.
- Validate MCP connectivity with the [MCP Inspector](./mcp_troubleshoot#mcp-inspector) to ensure the server responds.
- Reproduce the same MCP call via the LiteLLM Playground to confirm LiteLLM can complete the MCP hop independently.

<Image 
  img={require('../img/mcp_playground.png')}
  style={{width: '80%', display: 'block', margin: '0'}}
/>

### OAuth Metadata Discovery
LiteLLM performs metadata discovery per the MCP spec ([section 2.3](https://modelcontextprotocol.info/specification/draft/basic/authorization/#23-server-metadata-discovery)). When OAuth is enabled, confirm the authorization server exposes the metadata URL and that LiteLLM can fetch it.

**Actions**
- Use `curl <metadata_url>` (or similar) from the LiteLLM host to ensure the discovery document is reachable and contains the expected authorization/token endpoints.
- Record the exact metadata URL, requested scopes, and any static client credentials so support can replay the discovery step if needed.

## Debug Headers (Client-Side Diagnostics)

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

### Response Headers

When debug mode is enabled, LiteLLM returns these response headers (all sensitive values are masked):

| Header | Description | Example |
|--------|-------------|---------|
| `x-mcp-debug-inbound-auth` | Which inbound auth headers were present and how they were classified. | `x-litellm-api-key=Bearer****1234` |
| `x-mcp-debug-oauth2-token` | The OAuth2 token extracted from the `Authorization` header (masked). Shows `(none)` if absent, or flags `SAME_AS_LITELLM_KEY` when the LiteLLM key is leaking to the MCP server. | `Bearer****ef01` or `(none)` |
| `x-mcp-debug-auth-resolution` | Which auth priority was used for the outbound MCP call. | `oauth2-passthrough`, `m2m-client-credentials`, `per-request-header`, `static-token`, or `no-auth` |
| `x-mcp-debug-outbound-url` | The upstream MCP server URL that will receive the request. | `https://mcp.atlassian.com/v1/mcp` |
| `x-mcp-debug-server-auth-type` | The `auth_type` configured on the MCP server. | `oauth2`, `bearer_token`, or `(none)` |

### Common Issues

#### LiteLLM API key leaking to the MCP server

**Symptom:** `x-mcp-debug-oauth2-token` shows `SAME_AS_LITELLM_KEY`.

This means the `Authorization` header carries the LiteLLM API key and it's being forwarded to the upstream MCP server instead of an OAuth2 token. The OAuth2 flow never ran because the client already had an `Authorization` header set.

**Fix:** Move the LiteLLM key to `x-litellm-api-key` so the `Authorization` header is free for OAuth2 discovery:

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

This means the client didn't go through the OAuth2 flow. Check that:
1. The `Authorization` header is NOT set as a static header in the client config.
2. The `.well-known/oauth-protected-resource` endpoint returns valid metadata.
3. The MCP server in LiteLLM config has `auth_type: oauth2`.

#### M2M token used instead of user token

**Symptom:** `x-mcp-debug-auth-resolution` shows `m2m-client-credentials`.

This means the server has `client_id`/`client_secret`/`token_url` configured and LiteLLM is fetching a machine-to-machine token instead of using the per-user OAuth2 token. If you want per-user tokens, remove the client credentials from the server config.

### Debugging with Claude Code

Claude Code has a built-in debug mode for MCP connections:

```bash
# Start Claude Code with MCP debug logging
claude --debug

# Then use /mcp to inspect MCP server status
/mcp
```

This shows the MCP connection lifecycle, including OAuth2 discovery, token exchange, and transport errors on the client side.

### Debugging with MCP Python SDK

Enable verbose logging in the MCP Python SDK to see HTTP-level details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# MCP SDK will log HTTP requests, headers (excluding secrets), and responses
```

## Verify Connectivity

Run lightweight validations before impacting production traffic.

### MCP Inspector
Use the MCP Inspector when you need to test both `Client → LiteLLM` and `Client → MCP` communications in one place; it makes isolating the failing hop straightforward.

1. Execute `npx @modelcontextprotocol/inspector` on your workstation.
2. Configure and connect:
   - **Transport Type:** choose the transport the client uses (Streamable HTTP for LiteLLM).
   - **URL:** the endpoint under test (LiteLLM MCP URL for `Client → LiteLLM`, or the MCP server URL for `Client → MCP`).
   - **Custom Headers:** e.g., `Authorization: Bearer <LiteLLM API Key>`.
3. Open the **Tools** tab and click **List Tools** to verify the MCP alias responds.

### `curl` Smoke Test
`curl` is ideal on servers where installing the Inspector is impractical. It replicates the MCP tool call LiteLLM would make—swap in the domain of the system under test (LiteLLM or the MCP server).

```bash
curl -X POST https://your-target-domain.example.com/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

Add `-H "Authorization: Bearer <LiteLLM API Key>"` when the target is a LiteLLM endpoint that requires authentication. Adjust the headers, or payload to target other MCP methods. Matching failures between `curl` and LiteLLM confirm that the MCP server or network/OAuth layer is the culprit.

## Review Logs

Well-scoped logs make it clear whether LiteLLM reached the MCP server and what happened next.

### Access Log Example (successful MCP call)
```text
INFO:     127.0.0.1:57230 - "POST /everything/mcp HTTP/1.1" 200 OK
```

### Error Log Example (failed MCP call)
```text
07:22:00 - LiteLLM:ERROR: client.py:224 - MCP client list_tools failed - Error Type: ExceptionGroup, Error: unhandled errors in a TaskGroup (1 sub-exception), Server: http://localhost:3001/mcp, Transport: MCPTransport.http
  httpcore.ConnectError: All connection attempts failed
ERROR:LiteLLM:MCP client list_tools failed - Error Type: ExceptionGroup, Error: unhandled errors in a TaskGroup (1 sub-exception)...
  httpx.ConnectError: All connection attempts failed
```
