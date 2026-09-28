# MCP gateway

Native MCP routing using the official [Rust SDK](https://github.com/modelcontextprotocol/rust-sdk), pinned to `rmcp 3.4.1`. The crate exposes a mountable Axum router and an SDK `ServerHandler`

The HTTP router serves `/mcp`, `/{server}/mcp`, `/mcp/{server}`, `/mcp/{server}/mcp`, legacy `/mcp/sse` and `/mcp/sse/messages`, `/mcp/enabled`, and `/mcp-rest/tools/list` and `/mcp-rest/tools/call`. Streamable HTTP supports stateless requests and legacy initialize/session flows. The caller supplies the shutdown token, allowed hosts, allowed origins, and server identity through `HttpConfig`

`NativeGateway` aggregates upstream tools, prompts, resources, and resource templates. Tool and prompt calls resolve registered server prefixes, including prefixes containing hyphens. Resource names receive prefixes while their URIs remain unchanged, and reads require exactly one selected server. Tool allowlists apply to discovery and execution. Upstream pagination is drained with cycle detection. Aggregate tool listings retain healthy servers and expose sanitized outcomes under `_meta["litellm.ai/server_outcomes"]`, while scoped REST listings preserve upstream failures

## Integration

Construct `Server` entries from connected SDK peers and pass a `Registry` to `NativeGateway::new`. Keep their `RunningService` handles alive in the host and cancel them during shutdown. SDK peers can use stdio, Streamable HTTP, or another SDK transport. Connection establishment, outbound credentials, HTTP client pools, and reconnection remain owned by the host

Mount `router(operations, config)` behind the gateway's authentication middleware before merging it with other endpoint routers. `Registry` is an authorized static catalog, not an authenticator. Production hosts must authenticate all requests, including initialize, SSE GET, session POST, and DELETE. Authentication middleware may insert `SessionOwner` into request extensions to bind sessions to a stable principal across credential rotation. Otherwise sessions bind to a digest of the presented credentials and requested server scope

Implement `ServerResolver` to select authorized peers and tool permissions for each request. It receives HTTP headers and trusted extensions through `Context.parts`, plus the downstream MCP request context when available. Path and `x-mcp-servers` scopes can narrow that selection, never broaden it. Outbound headers are not copied automatically

Wrap `Operations` to integrate guardrails, spend logging, virtual tools, and other policy. Both MCP and REST execution use this interface. Database management, OAuth/BYOK services, toolsets, OpenAPI conversion, and semantic search are integration responsibilities in this stage, rather than new implementations in this crate

`RelayClient::for_request` forwards sampling, elicitation, roots, and progress to the originating downstream client. Create it for that request's upstream connection and retain the connection until execution completes. Do not share this relay across callers. Sampling can instead be handled by an injected SDK `ClientHandler` backed by inference

## Local interoperability check

From the repository root, this loopback-only example runs the existing Python MCP fixture through the SDK's stdio transport. It has no admission authentication and is intended for local testing

```sh
cargo run --manifest-path litellm-rust/Cargo.toml -p litellm-gateway-mcp --example stdio_proxy -- \
  .venv/bin/python -c 'import runpy; runpy.run_path("tests/mcp_tests/mcp_e2e_upstream_server.py")["mcp"].run(transport="stdio")'
```

```sh
curl -sS http://127.0.0.1:4000/mcp-rest/tools/call \
  -H 'Content-Type: application/json' \
  -d '{"server_id":"local","name":"add","arguments":{"a":2,"b":3}}'
```

```json
{"content":[{"type":"text","text":"5"}],"structuredContent":{"result":5},"isError":false}
```

```sh
curl -sS http://127.0.0.1:4000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'MCP-Protocol-Version: 2025-11-25' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"local-multiply","arguments":{"a":6,"b":7}}}'
```

```text
data: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"42"}],"structuredContent":{"result":42},"isError":false}}
```
