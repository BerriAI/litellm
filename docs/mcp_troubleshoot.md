import Image from '@theme/IdealImage';

# MCP Troubleshooting Guide

When LiteLLM acts as an MCP proxy, traffic normally flows `Client → LiteLLM Proxy → MCP Server`, while OAuth-enabled setups add an authorization server for metadata discovery.

For provisioning steps, transport options, and configuration fields, refer to [mcp.md](./mcp.md).

## Quick Start: Debug with One Command

The fastest way to debug MCP issues is to enable **debug headers**. Run this curl against your LiteLLM proxy and check the response headers:

```bash
curl -si -X POST http://localhost:4000/{your_mcp_server}/mcp \
  -H "Content-Type: application/json" \
  -H "x-litellm-api-key: Bearer sk-YOUR_KEY" \
  -H "x-litellm-mcp-debug: true" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
  2>&1 | grep -i "x-mcp-debug"
```

This returns masked diagnostic headers that tell you exactly what's happening with authentication:

```
x-mcp-debug-inbound-auth: x-litellm-api-key=Bearer****1234
x-mcp-debug-oauth2-token: Bearer****ef01
x-mcp-debug-auth-resolution: oauth2-passthrough
x-mcp-debug-outbound-url: https://mcp.atlassian.com/v1/mcp
x-mcp-debug-server-auth-type: oauth2
```

If you see `SAME_AS_LITELLM_KEY` in `x-mcp-debug-oauth2-token`, your LiteLLM API key is leaking to the MCP server instead of an OAuth2 token. See [Debugging OAuth](./mcp_oauth#debugging-oauth) for the fix and other common issues.

For Claude Code, add the debug header to your MCP config:

```bash
claude mcp add --transport http my_server http://localhost:4000/my_mcp/mcp \
  --header "x-litellm-api-key: Bearer sk-..." \
  --header "x-litellm-mcp-debug: true"
```

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
- From the LiteLLM server, run a [`curl` smoke test](./mcp_troubleshoot#curl-smoke-test) against the MCP endpoint to confirm basic connectivity.

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

## Debugging OAuth

For detailed OAuth2 debugging — including debug header reference, common misconfigurations, and example output — see [Debugging OAuth](./mcp_oauth#debugging-oauth).

### MCP OAuth: Connect returns `{"detail":"invalid_request"}` {#mcp-oauth-invalid-request}

**Symptom.** Clicking **Connect** on an MCP OAuth server in the LiteLLM UI returns:

```
HTTP/1.1 400 Bad Request
{"detail":"invalid_request"}
```

The proxy logs (with verbose logging) show a line like `MCP OAuth: rejecting redirect_uri ... as invalid_request. Computed proxy base=...`.

**Cause.** The `/v1/mcp/server/oauth/{server_id}/authorize` endpoint validates that the browser-supplied `redirect_uri` (`https://llm.example.com/ui/mcp/oauth/callback`) shares scheme + host + port with the proxy's own public origin. Behind a TLS-terminating ingress (Kubernetes, ALB, nginx, Cloudflare, etc.) the proxy resolves to its internal address (`http://<pod-ip>:4000`) by default, so the same-origin check rejects.

**Diagnostic.** Compare what the proxy advertises as its origin to what the browser sees:

```bash
curl -sS https://llm.example.com/.well-known/oauth-authorization-server | jq .issuer
```

The `issuer` value should equal the origin the user types into their browser (`https://llm.example.com`). If it returns an internal hostname or `http://...`, the proxy's resolved origin is wrong.

**Fixes**, in order of preference:

1. **Set `PROXY_BASE_URL`** (recommended). Operator declares the proxy's true public origin out of band, no header trust required:

   ```bash
   PROXY_BASE_URL=https://llm.example.com
   ```

   Full origin only: scheme + host (+ port if non-default), no trailing slash, no path. See [Reverse proxy and ingress configuration](./mcp_oauth#reverse-proxy-and-ingress-configuration).

2. **Trust `X-Forwarded-*` from your ingress.** Set both keys in `general_settings`:

   ```yaml title="config.yaml" showLineNumbers
   general_settings:
     use_x_forwarded_for: true
     mcp_trusted_proxy_ranges:
       - "10.0.0.0/8"      # your ingress / load-balancer CIDR(s)
   ```

   `use_x_forwarded_for` alone is not enough — without `mcp_trusted_proxy_ranges`, the proxy refuses to honor `X-Forwarded-*` because it cannot tell a trusted reverse proxy from a direct attacker. Verify that your ingress sends `X-Forwarded-Proto`, `X-Forwarded-Host`, and (when running on a non-default port) `X-Forwarded-Port`.

3. **Fix the ingress.** If the ingress is stripping or rewriting `X-Forwarded-*`, no proxy setting will help — restore the headers at the ingress layer.

If the `redirect_uri` legitimately lives on a sister domain you control (e.g. an internal web app registering as an OAuth client of the MCP proxy), allowlist its origin via `MCP_TRUSTED_REDIRECT_ORIGINS`. See [Allowing additional first-party redirect_uri origins](./mcp_oauth#allowing-additional-first-party-redirect_uri-origins).

## Verify Connectivity

Run lightweight validations before impacting production traffic.

### MCP Inspector
Use the MCP Inspector when you need to test both `Client → LiteLLM` and `Client → MCP` communications in one place; it makes isolating the failing hop straightforward.

1. Execute `npx @modelcontextprotocol/inspector` on your workstation.
2. Configure and connect:
   - **Transport Type:** choose the transport the client uses (Streamable HTTP for LiteLLM).
   - **URL:** the endpoint under test (LiteLLM MCP URL for `Client → LiteLLM`, or the MCP server URL for `Client → MCP`).
   - **Custom Headers:** e.g., `x-litellm-api-key: Bearer <LiteLLM API Key>`.
3. Open the **Tools** tab and click **List Tools** to verify the MCP alias responds.

### `curl` Smoke Test
`curl` is ideal on servers where installing the Inspector is impractical. It replicates the MCP tool call LiteLLM would make—swap in the domain of the system under test (LiteLLM or the MCP server).

```bash
curl -X POST https://your-target-domain.example.com/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

Add `-H "x-litellm-api-key: Bearer <LiteLLM API Key>"` when the target is a LiteLLM endpoint that requires authentication. Adjust the headers or payload to target other MCP methods. Matching failures between `curl` and LiteLLM confirm that the MCP server or network/OAuth layer is the culprit.

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
