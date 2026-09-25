# LiteLLM MCP Client

LiteLLM MCP Client allows you to use MCP tools with LiteLLM

Install the optional dependencies with `pip install 'litellm[mcp]'`, then use the existing public imports:

```python
from litellm.experimental_mcp_client import call_openai_tool, load_mcp_tools
from litellm.experimental_mcp_client.client import MCPClient

client = MCPClient(server_url="https://mcp.example.com/mcp")
```

Core `import litellm` works without the MCP extra. Importing the experimental MCP client without its MCP or HTTPX2 dependency raises an error with this installation command

## MCP Python SDK compatibility

The `mcp` and `proxy` extras require MCP Python SDK 2.2 or newer within the 2.x release line. Installing core LiteLLM without these extras does not require MCP

Existing MCP SDK1 clients can continue connecting to the gateway over the supported legacy MCP protocols. The client and gateway can use different SDK versions in separate Python environments. Modern protocol advertisement remains disabled during the Phase 0 upgrade. An initialize body requesting `2026-07-28` falls back to the supported legacy version `2025-11-25`; an explicit `MCP-Protocol-Version: 2026-07-28` HTTP header is rejected with HTTP 400

Code sharing the gateway's Python environment must support SDK2. Its Python API has breaking changes, including renamed imports and snake_case model attributes such as `input_schema`, `is_error`, and `structured_content`. This also applies to callers consuming SDK objects returned by LiteLLM's experimental MCP client. MCP JSON fields retain their protocol spelling, such as `inputSchema` and `isError`

Upgrade SDK1-dependent libraries before installing them alongside `litellm[mcp]` or `litellm[proxy]`, or keep those clients in a separate environment and connect over the network. For example, `langchain-mcp-adapters==0.2.1` uses SDK1 Python APIs and is tested as a separate legacy client, not as a shared SDK2 dependency

The shared unit-test workflow runs the MCP integration suite once, with SDK2 in the gateway environment and an isolated SDK1 peer. Keep the SDK1 list/call compatibility test while SDK1 clients are supported; remove it when that support is explicitly retired and the client migration is documented

See the official [SDK migration guide](https://py.sdk.modelcontextprotocol.io/migration/) for Python API changes

## Custom HTTP clients and authentication

MCP HTTP and SSE transports now use `httpx2`. Custom authentication passed through `aws_auth` or `resolved_auth` must implement `httpx2.Auth`. Integrations that override the client's HTTP client factory or customize its event hooks must use `httpx2.AsyncClient`, request, response, timeout and transport types

HTTPX1 clients, auth objects and hooks are not adapted by a compatibility shim. Migrate those integrations to HTTPX2 before upgrading. Ordinary `MCPClient` construction and LiteLLM's existing helper imports remain supported; this does not restore SDK1 Python imports or camelCase SDK model attributes in the shared Python environment

## HTTP redirects

For streamable HTTP POST requests, the MCP SDK follows method-preserving redirects such as HTTP 307/308 within the configured endpoint's origin. Redirects to another path on the same scheme, host and port work. The SDK also permits an HTTP-to-HTTPS upgrade on the same host using the default ports

Redirects to a different origin are rejected before the destination receives a request or credentials. Configure the final MCP endpoint URL directly if the server redirects to a different host or port. Setting the HTTP client's `follow_redirects` option does not override the SDK's policy

### Protocol version selection

`MCPClient(protocol_version="2025-06-18", ...)` offers that exact legacy revision and rejects an upstream that selects a different revision before listing or calling tools. The default `"auto"` preserves legacy initialization and accepts only `2024-11-05`, `2025-03-26`, `2025-06-18`, and `2025-11-25`; it does not probe or fall back to modern discovery

The gateway can restrict its enabled revisions and pin each YAML upstream independently:

```yaml
general_settings:
  mcp_advertised_versions: ["2025-06-18", "2025-11-25"]
mcp_servers:
  example:
    url: https://example.com/mcp
    transport: http
    protocol_version: "2025-06-18"
```

For database-backed servers, set `mcp_info.protocol_version` in the existing MCP server create/update API. YAML also accepts this metadata setting; a top-level `protocol_version` takes precedence. Omitted settings preserve the defaults. Empty gateway version lists, unknown revisions, and modern revision settings are rejected. A handshake selecting a disabled revision fails explicitly; configure every revision your clients require

`2026-07-28` is represented in capability data but remains disabled for public serving. Apps and Tasks are not advertised. Discovery results use the caller's existing access checks, are private and uncached, and do not enable capabilities merely because the SDK knows their schemas
