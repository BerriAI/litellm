# LiteLLM MCP Client

LiteLLM MCP Client allows you to use MCP tools with LiteLLM

## MCP Python SDK compatibility

The `mcp` and `proxy` extras require MCP Python SDK 2.2 or newer within the 2.x release line. Installing core LiteLLM without these extras does not require MCP

Existing MCP SDK1 clients can continue connecting to the gateway over the supported legacy MCP protocols. The client and gateway can use different SDK versions in separate Python environments. Modern protocol advertisement remains disabled during the Phase 0 upgrade. An initialize body requesting `2026-07-28` falls back to the supported legacy version `2025-11-25`; an explicit `MCP-Protocol-Version: 2026-07-28` HTTP header is rejected with HTTP 400

Code sharing the gateway's Python environment must support SDK2. Its Python API has breaking changes, including renamed imports and snake_case model attributes such as `input_schema`, `is_error`, and `structured_content`. This also applies to callers consuming SDK objects returned by LiteLLM's experimental MCP client. MCP JSON fields retain their protocol spelling, such as `inputSchema` and `isError`

Upgrade SDK1-dependent libraries before installing them alongside `litellm[mcp]` or `litellm[proxy]`, or keep those clients in a separate environment and connect over the network. For example, `langchain-mcp-adapters==0.2.1` uses SDK1 Python APIs and is tested as a separate legacy client, not as a shared SDK2 dependency

The shared unit-test workflow runs the MCP integration suite once, with SDK2 in the gateway environment and an isolated SDK1 peer. Keep the SDK1 list/call compatibility test while SDK1 clients are supported; remove it when that support is explicitly retired and the client migration is documented

See the official [SDK migration guide](https://py.sdk.modelcontextprotocol.io/migration/) for Python API changes

## HTTP redirects

For streamable HTTP POST requests, the MCP SDK follows method-preserving redirects such as HTTP 307/308 within the configured endpoint's origin. Redirects to another path on the same scheme, host and port work. The SDK also permits an HTTP-to-HTTPS upgrade on the same host using the default ports

Redirects to a different origin are rejected before the destination receives a request or credentials. Configure the final MCP endpoint URL directly if the server redirects to a different host or port. Setting the HTTP client's `follow_redirects` option does not override the SDK's policy
