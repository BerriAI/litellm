# Forwarding gateway

`Forward` is a reverse proxy in front of the Python proxy on loopback. Python must see each request as if the client had connected to it directly, because every Python route (inference, MCP, pass-through, UI, WebSocket) is served behind it without Rust knowing which one it is

- Serve HTTP/1.1 only, as uvicorn does. The supervisor never puts Rust in front of a Python server that speaks HTTP/2
- Never branch on a route or path. A behavior one endpoint needs is a protocol behavior every endpoint gets
- Forward the method, request target, headers, and body byte for byte within the [URL contract](FORWARDING.md#url-fragments-and-literal-hashes). An unescaped `#` starts a fragment and is omitted with everything after it, while `%23` is preserved. Only drop hop-by-hop headers (RFC 9110 section 7.6.1) and set `x-litellm-rust-peer`, which replaces any client copy
- Stream both bodies. Never buffer, cap, or time out a body, since uploads and provider streams have no upper bound
- A client disconnect closes the upstream connection at every stage, because that is how Python learns to stop a provider stream
- An upstream that dies mid-response fails the client body. Never end it cleanly, or SSE clients treat a truncated stream as complete
- Never invent a Host. Python builds absolute URLs from it, so a missing one must never become the private upstream address
- A request Python could not have received unchanged (HTTP/1.1 without Host, a transfer coding other than chunked, a target without a path) gets the status uvicorn would have given, from Rust
- Pooled upstream connections are retired well before Python's keep-alive timeout, so a request never lands on a connection Python is closing
- Idle client connections close after Python's keep-alive timeout, as uvicorn closed them, so a client that vanishes cannot hold a socket forever. Only the wait for the next request is timed, never an exchange in progress
- WebSocket is the one protocol switch Python serves. Upgrades are relayed as raw bytes after the 101 and drained on shutdown

## Layout

Every forwarding decision has one home. `rewrite.rs` owns what Python and the client receive, as pure functions over requests and responses. `Rejection` in `error.rs` is the only place that picks a status for a request Rust answers itself. `settings.rs` parses the environment into typed values, so an `Upstream` is a loopback origin by construction. `server.rs` accepts client connections and owns their timeouts, since `axum::serve` exposes none, and `connections.rs` tracks every connection and relay for shutdown. `forward.rs` and `websocket.rs` only move requests, bodies and upgraded connections and never touch a header themselves. `connector.rs` keeps hyper reading Python's answer after Python stops reading the request

## Environment

`Settings` reads only these variables, and only when `LITELLM_FORWARD_UPSTREAM` is set. The Python supervisor in `litellm/proxy/rust_forward/` sets every one of them from the values Python actually runs with, so all of them are required and a missing or invalid one fails startup naming it. The crate never reads Python's own settings (`HOST`, `PORT`, `KEEPALIVE_TIMEOUT`, `GRACEFUL_SHUTDOWN_TIMEOUT`) and keeps no copy of Python's defaults, since Python can get those values from CLI flags the environment never shows

| Variable | Meaning |
|---|---|
| `LITELLM_FORWARD_UPSTREAM` | Python's loopback origin, e.g. `http://127.0.0.1:53211` |
| `LITELLM_FORWARD_HOST` | public bind host |
| `LITELLM_FORWARD_PORT` | public bind port |
| `LITELLM_FORWARD_UPSTREAM_KEEPALIVE` | seconds uvicorn keeps an idle connection open. Pooled connections are retired at half of it, and idle client connections are closed after it |
| `LITELLM_FORWARD_SHUTDOWN_DRAIN` | seconds to drain on shutdown, already including Python's graceful timeout and teardown headroom |

## Tests

Test through real sockets with `tests/support`, never `oneshot`. `Gateway` runs the forwarder on a listener like production, `recording` is a well-behaved upstream that captures what Python would receive, and `raw_upstream` and `Raw` drive bytes by hand for framing, timing and disconnects a real server would not produce. Waits measured in minutes use paused tokio time instead of real sleeps

The forwarder sees HTTP/1.1 messages, not routes, so the tests are split into one file per traffic shape uvicorn can emit: `exchange` (plain request and response), `framing` (requests whose framing a parser could read two ways), `slow_head` (minutes before response headers), `stream` (incremental response bodies), `idle_stream` (long-lived streams that go quiet), `upload`, `download`, `websocket`, `client_identity`, `lifecycle` (disconnects, pooling, idle timeouts, shutdown) and `roundtrip` (the forwarding invariant for generated messages). A new route never gets its own test. Add its behavior to the shape it already has, and when it needs a wire behavior none of the files covers, add a new shape file for it
